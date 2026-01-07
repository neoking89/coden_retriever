"""Query execution helpers for the coding agent.

Contains logic for running queries with fallback handling and error reporting.
Extracted from CodingAgent to follow Single Responsibility Principle.
"""

from typing import TYPE_CHECKING, Optional

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError

from .debug_logger import DebugLogger
from .permission_toolset import PermissionToolsetWrapper
from .react_loop import parse_messages_to_steps
from .response_renderer import AnswerRenderer, StreamRenderer
from .rich_console import console, format_exception_message, print_error, print_steps_rich, print_warning
from .stream_handler import StreamEventHandler
from .text_tool_fallback import handle_fallback_iteration
from ..config_loader import get_config
from ..mcp.tool_filter import display_filtered_tools

if TYPE_CHECKING:
    from .interactive_loop import CommandContext, InteractiveLoop

# Multiplier for request limit relative to max_steps.
# Allows for retries and intermediate requests within the step budget.
REQUEST_LIMIT_MULTIPLIER = 2


class QueryExecutor:
    """Handles query execution with streaming, fallback, and error handling."""

    def __init__(self, max_steps: int, model_str: str):
        self.max_steps = max_steps
        self.model_str = model_str

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
        self._apply_tool_filtering(context, prompt)

        current_history = loop.history
        all_messages = []
        fallback_iterations = 0

        while fallback_iterations < self.max_steps:
            result, stream_handler = await self._run_single_iteration(
                agent, prompt, current_history, debug_logger
            )

            if result is None:
                raise RuntimeError("Agent run failed without exception")

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
                prompt = fallback_result.continuation_prompt
                continue

            self._finalize_response(
                debug_logger, loop, all_messages, steps,
                total_tool_calls, answer_text
            )
            return

        print_warning("Reached max fallback iterations")
        if all_messages:
            loop.update_history(all_messages)

    def _apply_tool_filtering(
        self, context: Optional["CommandContext"], user_input: str
    ) -> None:
        """Apply dynamic tool filtering if enabled.

        Reads threshold from config cache for immediate updates via /config set.
        """
        if not context or not context.dynamic_tool_filtering:
            return

        # Use semantic_filter to set allowed tools for this query
        if context.semantic_filter:
            context.semantic_filter.set_filter_for_query(user_input)

        if context.tool_filter:
            config = get_config()
            threshold = config.agent.tool_filter_threshold if config else context.tool_filter_threshold
            filtered_results = context.tool_filter.filter(
                user_input, threshold=threshold
            )
            display_filtered_tools(filtered_results, console)

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


class ErrorHandler:
    """Handles error reporting for model and generic errors."""

    def __init__(self, model_str: str):
        self.model_str = model_str

    def handle_model_error(self, e: ModelHTTPError, debug_logger: DebugLogger) -> None:
        """Handle model HTTP errors with helpful suggestions."""
        debug_logger.log_error(e, context="Query execution")

        body = e.body if isinstance(e.body, dict) else {}
        error_type = body.get("type", "")
        error_message = body.get("message", str(e))

        is_tool_error = error_type == "invalid_request_error" and e.status_code == 400
        if is_tool_error:
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
