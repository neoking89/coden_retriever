"""Query execution helpers for the coding agent.

Contains logic for running queries with fallback handling and error reporting.
Extracted from CodingAgent to follow Single Responsibility Principle.
"""

import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage

from ..config_loader import get_config
from ..mcp.tool_filter import display_filtered_tools
from .compaction import maybe_compact_history
from .debug_logger import DebugLogger
from .permission_toolset import PermissionToolsetWrapper
from .react_loop import parse_messages_to_steps
from .response_renderer import AnswerRenderer, StreamRenderer
from .rich_console import (
    console,
    format_exception_message,
    print_error,
    print_session_baseline,
    print_steps_rich,
    print_token_usage,
    print_warning,
)
from .stream_handler import StreamEventHandler
from .text_tool_fallback import handle_fallback_iteration
from .token_usage import derive_baseline_from_first_turn, get_active_context_usage

if TYPE_CHECKING:
    from .interactive_loop import CommandContext, InteractiveLoop

# Multiplier for request limit relative to max_steps.
# Allows for retries and intermediate requests within the step budget.
REQUEST_LIMIT_MULTIPLIER = 2

# Default retry limit for malformed tool calls (concatenated names/args)
# that cause 400 errors from the provider API.
# Kept at 2 (not DEFAULT_MAX_RETRIES=5) because this path only fires when no
# CommandContext is available — a narrow code path that should fail fast rather
# than retry aggressively. Overridden by the configurable max_retries setting.
DEFAULT_TOOL_ERROR_RETRIES: int = 2

# HTTP 400 Bad Request -- the status code providers return for malformed tool calls.
HTTP_BAD_REQUEST = 400

# The error type string providers use for invalid request payloads.
INVALID_REQUEST_ERROR_TYPE = "invalid_request_error"

# Prompt hint appended on retry to steer the model toward single tool calls.
SEQUENTIAL_TOOL_HINT = (
    "\n\nIMPORTANT: Call exactly ONE tool at a time. "
    "Do not combine multiple tool calls into a single request."
)

# Keywords that must appear in the error message body to confirm a tool-call 400,
# preventing over-broad retries on unrelated 400s (e.g. invalid model name, context overflow).
TOOL_CALL_ERROR_KEYWORDS: frozenset[str] = frozenset({"tool", "function", "tool_call"})


class QueryExecutor:
    """Handles query execution with streaming, fallback, and error handling."""

    def __init__(self, max_steps: int, model_str: str, max_retries: int = DEFAULT_TOOL_ERROR_RETRIES):
        self.max_steps = max_steps
        self.model_str = model_str
        self.max_retries = max_retries

    async def execute(
        self,
        agent: Agent,
        prompt: str,
        debug_logger: DebugLogger,
        loop: "InteractiveLoop",
        context: Optional["CommandContext"] = None,
    ) -> None:
        """Execute a single query with streaming and fallback handling."""
        debug_logger.log_user_prompt(prompt)
        await self._apply_tool_filtering(context, prompt, debug_logger)

        wall_start: datetime = datetime.now()
        mono_start: float = time.monotonic()

        current_history = loop.history
        all_messages = []
        fallback_iterations = 0
        tool_error_retries = 0
        accumulated_usage = RunUsage()

        while fallback_iterations < self.max_steps:
            try:
                result, stream_handler = await self._run_single_iteration(
                    agent, prompt, current_history, debug_logger
                )
            except ModelHTTPError as e:
                if self._is_tool_call_error(e) and tool_error_retries < self.max_retries:
                    tool_error_retries += 1
                    print_warning(
                        f"Model produced an invalid tool call "
                        f"(retry {tool_error_retries}/{self.max_retries})"
                    )
                    prompt = self._add_sequential_tool_hint(prompt)
                    continue
                raise

            tool_error_retries = 0

            if result is None:
                raise RuntimeError("Agent run failed without exception")

            accumulated_usage += result.usage()
            all_messages = result.all_messages()
            steps = parse_messages_to_steps(all_messages)
            total_tool_calls = sum(1 for step in steps if step.action is not None)
            answer_text = str(result.output) if result.output else stream_handler.get_streamed_text()

            fallback_result = await self._handle_fallback(
                context, answer_text, total_tool_calls, all_messages,
                fallback_iterations, debug_logger
            )

            if fallback_result.should_continue:
                if fallback_result.steps:
                    print_steps_rich(fallback_result.steps)
                fallback_iterations += fallback_result.tool_call_count
                current_history = fallback_result.updated_history
                # Preserve the sequential-tool hint across fallback continuation
                # prompts so that a model which previously triggered a retry does
                # not lose the hint when the prompt is replaced by the fallback path.
                new_prompt = fallback_result.continuation_prompt
                prompt = self._add_sequential_tool_hint(new_prompt) if SEQUENTIAL_TOOL_HINT in prompt else new_prompt
                continue

            outcome = await maybe_compact_history(
                all_messages, loop.tree, get_config().agent,
                debug_logger=debug_logger,
            )
            if outcome.happened:
                all_messages = outcome.messages
                steps = parse_messages_to_steps(all_messages)

            self._finalize_response(
                debug_logger, loop, all_messages, steps,
                total_tool_calls, answer_text,
                wall_start=wall_start,
                mono_start=mono_start,
            )
            if outcome.bottom_line:
                console.print(outcome.bottom_line)
                console.print()
            return

        print_warning("Reached max fallback iterations")
        if all_messages:
            loop.update_history(all_messages)

    @staticmethod
    def _is_tool_call_error(e: ModelHTTPError) -> bool:
        """Check if the error is a 400 caused by a malformed tool call.

        Some models (especially via Ollama) concatenate multiple tool names
        into a single call, which the provider API rejects. The keyword check
        on the message body prevents retrying unrelated 400s (e.g. invalid
        model name, context-window overflow, auth errors returning 400).
        """
        if e.status_code != HTTP_BAD_REQUEST:
            return False
        body = e.body if isinstance(e.body, dict) else {}
        if body.get("type") != INVALID_REQUEST_ERROR_TYPE:
            return False
        message = str(body.get("message", "")).lower()
        return any(kw in message for kw in TOOL_CALL_ERROR_KEYWORDS)

    @staticmethod
    def _add_sequential_tool_hint(prompt: str) -> str:
        """Append a hint for sequential tool usage if not already present."""
        if SEQUENTIAL_TOOL_HINT in prompt:
            return prompt
        return f"{prompt}{SEQUENTIAL_TOOL_HINT}"

    async def _apply_tool_filtering(
        self, context: Optional["CommandContext"], user_input: str,
        debug_logger: DebugLogger,
    ) -> None:
        """Apply LLM-based tool routing if enabled.

        Streams the router's LLM output to the console and logs the
        routing result, consistent with how the main agent displays output.
        """
        if not context or not context.dynamic_tool_filtering:
            return

        if not context.llm_filter:
            return

        console.print("[dim]>> Routing tools...[/dim]")

        stream_handler = StreamEventHandler(debug_logger)
        with StreamRenderer() as renderer:
            stream_handler.on_update = renderer.update
            filter_result = await context.llm_filter.set_filter_for_query(
                user_input,
                event_stream_handler=stream_handler.handle_events,
            )
            stream_handler.flush_remaining()

        if filter_result is not None:
            display_filtered_tools(filter_result, console)
            debug_logger.log_tool_routing(
                query=user_input,
                selected_tools=[
                    t.metadata.name for t in filter_result.domain_tools
                ],
                total_domain_tools=len(context.llm_filter.tool_router.domain_tools)
                if context.llm_filter.tool_router
                else 0,
            )

    async def _run_single_iteration(
        self, agent: Agent, prompt: str, history, debug_logger: DebugLogger
    ):
        """Run a single agent iteration with streaming."""
        stream_handler = StreamEventHandler(debug_logger)

        with StreamRenderer() as renderer:
            stream_handler.on_update = renderer.update

            result = None
            try:
                result = await agent.run(
                    prompt,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=self.max_steps * REQUEST_LIMIT_MULTIPLIER),
                    event_stream_handler=stream_handler.handle_events,
                )
            finally:
                stream_handler.flush_remaining(error_occurred=result is None)

        return result, stream_handler

    async def _handle_fallback(
        self,
        context: Optional["CommandContext"],
        answer_text: str,
        total_tool_calls: int,
        all_messages,
        step_number_start: int,
        debug_logger: DebugLogger,
    ):
        """Handle text-based tool call fallback for models without native tool support."""
        ask_permission = None
        if context and context.toolset:
            if isinstance(context.toolset, PermissionToolsetWrapper):
                ask_permission = context.toolset.ask_permission_for_fallback

        server = context.server if context else None
        return await handle_fallback_iteration(
            server=server,
            answer_text=answer_text,
            total_tool_calls=total_tool_calls,
            all_messages=all_messages,
            step_number_start=step_number_start,
            debug_logger=debug_logger,
            ask_permission=ask_permission,
        )

    def _finalize_response(
        self,
        debug_logger: DebugLogger,
        loop: "InteractiveLoop",
        all_messages,
        steps,
        total_tool_calls: int,
        answer_text: str,
        *,
        wall_start: datetime,
        mono_start: float,
    ) -> None:
        """Finalize and display the agent response."""
        debug_logger.log_message_history(all_messages)
        loop.update_history(all_messages)

        if steps:
            print_steps_rich(steps)

        if total_tool_calls >= self.max_steps:
            debug_logger.log_max_steps_reached(total_tool_calls, self.max_steps)
            print_warning("Reached max steps limit, answer may be incomplete")

        debug_logger.log_model_response(answer_text, is_final=True)

        if answer_text:
            AnswerRenderer().render(answer_text)
        else:
            console.print()
            print_warning("No response text generated")
            console.print()

        elapsed_seconds: float = time.monotonic() - mono_start
        wall_end: datetime = datetime.now()
        self._display_token_usage(
            loop,
            all_messages,
            elapsed_seconds=elapsed_seconds,
            wall_start=wall_start,
            wall_end=wall_end,
        )

    @staticmethod
    def _display_token_usage(
        loop: "InteractiveLoop",
        all_messages,
        *,
        elapsed_seconds: float,
        wall_start: datetime,
        wall_end: datetime,
    ) -> None:
        """Print retained active-context size, plus a one-time measured baseline."""
        if not loop.baseline_shown:
            baseline = derive_baseline_from_first_turn(all_messages)
            if baseline is not None:
                tool_count = len(loop.context.available_tools or [])
                print_session_baseline(baseline_tokens=baseline, tool_count=tool_count)
                loop.baseline_shown = True

        context_usage = get_active_context_usage(all_messages)
        print_token_usage(
            context_tokens=context_usage.tokens,
            estimated=context_usage.estimated,
            elapsed_seconds=elapsed_seconds,
            wall_start=wall_start,
            wall_end=wall_end,
        )


class ErrorHandler:
    """Handles error reporting for model and generic errors."""

    def __init__(self, model_str: str):
        self.model_str = model_str

    def handle_model_error(self, e: ModelHTTPError, debug_logger: DebugLogger) -> None:
        """Handle model HTTP errors with helpful suggestions."""
        debug_logger.log_error(e, context="Query execution")

        body = e.body if isinstance(e.body, dict) else {}
        error_message = body.get("message", str(e))

        if QueryExecutor._is_tool_call_error(e):
            print_error(f"Model request failed (HTTP {e.status_code}): {error_message}")
            print_warning(
                f"The model '{self.model_str}' may not support tool calling properly. "
                "Maybe try a different model with better tool support."
            )
        else:
            print_error(f"Model error (HTTP {e.status_code}): {error_message}")

        self._suggest_debug_log(debug_logger)

    def handle_generic_error(self, e: Exception, debug_logger: DebugLogger) -> None:
        """Handle generic errors with debug log suggestions."""
        debug_logger.log_error(e, context="Query execution")
        print_error(format_exception_message(e))
        self._suggest_debug_log(debug_logger)

    def _suggest_debug_log(self, debug_logger: DebugLogger) -> None:
        """Suggest checking debug logs."""
        log_path = debug_logger.get_log_path()
        if log_path:
            print_warning(f"See debug log: {log_path}")
        else:
            print_warning("Run with --debug flag for detailed logs")
