"""Coden-glue session orchestration around the library-pure CodingAgent.

Owns everything that depends on coden's config / daemon / MCP wiring:
- Interactive REPL lifecycle (daemon start/stop, MCP server context).
- Slash-command dispatch, @file expansion, study-mode triggers.
- Compaction post-processing of message history.
- Rich UI rendering plumbed through `EventCallbacks`.
- LLM tool-router setup and per-query filtering.

The pure `CodingAgent` lives in `coding_agent.py` and has zero coden
imports — extraction-ready. This module is what coden's CLI talks to.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage

from ..config_loader import get_config, load_config
from ..daemon import start_daemon_async, stop_daemon
from ..mcp.llm_tool_router import LLMToolRouter
from ..mcp.tool_filter import TOOL_QUERY_DESCRIPTIONS, ToolMetadata, display_filtered_tools
from .capabilities import is_tool_call_error
from .coden_models import AgentMode, SessionTrigger
from .coding_agent import CodingAgent
from .commands import FILTER_MODEL_SYNC_SENTINEL
from .compaction import maybe_compact_history
from .debug_logger import DebugLogger, create_debug_logger
from .file_reference import expand_file_references, find_file_references
from .filtering_toolset import create_filtered_toolset
from .interactive_loop import CommandContext, InteractiveLoop
from .mcp_server import create_mcp_server
from .model_factory import ModelFactory
from .models import AgentResponse
from .permission_toolset import PermissionCapability
from .prompt_builder import PromptBuilder
from .protocols import (
    EventCallbacks,
    PermissionChoice,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .query_executor import QueryExecutor
from .react_loop import parse_messages_to_steps
from .response_renderer import (
    AnswerRenderer,
    StderrToolReporter,
    StdoutStreamWriter,
    StreamRenderer,
)
from .rich_console import (
    console,
    format_exception_message,
    get_active_live,
    print_error,
    print_fatal_error,
    print_goodbye,
    print_session_baseline,
    print_steps_rich,
    print_token_usage,
    print_warning,
    print_welcome,
)
from .token_usage import derive_baseline_from_first_turn, get_active_context_usage
from .tool_permission_picker import ToolPermissionRequest, run_tool_permission_picker

# Session start triggers for study mode
_SESSION_START_TRIGGERS = frozenset(
    (SessionTrigger.EMPTY, SessionTrigger.START, SessionTrigger.BEGIN)
)
_SESSION_START_PREFIXES = ("begin the study session", "start the study session")


def get_mode_from_context(context: "CommandContext | None") -> AgentMode:
    """Determine agent mode from context."""
    if context and context.study_mode:
        return AgentMode.STUDY
    return AgentMode.CODING


def build_query_prompt(
    user_input: str | SessionTrigger,
    root_directory: str,
    mode: AgentMode,
    topic: str | None = None,
) -> str:
    """Build the query prompt based on agent mode."""
    if mode == AgentMode.STUDY:
        normalized = user_input.lower().strip()
        is_session_start = (
            normalized in _SESSION_START_TRIGGERS or
            any(normalized.startswith(p) for p in _SESSION_START_PREFIXES)
        )

        if is_session_start:
            return f"""[SESSION START] Topic: {topic or "General Architecture"}
Execute <teaching_flow> session start. NO tool calls - wait for user response."""

        return f"""[CONTINUE] User: "{user_input}"
Respond per <teaching_flow> and <constraints>. End with ONE question."""

    return f"Working directory: {root_directory}\n\n{user_input}"


def resolve_filter_model(
    coding_agent: CodingAgent,
    base_url: Optional[str],
    tool_filter_model: Optional[str],
):
    """Resolve the model instance for the tool filter router.

    Centralised here rather than on CodingAgent because the sync sentinel
    is a coden-side convention.
    """
    if not tool_filter_model or tool_filter_model == FILTER_MODEL_SYNC_SENTINEL:
        return coding_agent.model

    filter_factory = ModelFactory(
        tool_filter_model,
        base_url,
        api_key=get_config().model.generation.api_key,
    )
    return filter_factory.create_model()


def should_sync_router_on_model_switch(tool_filter_model: Optional[str]) -> bool:
    """Whether router model should follow the main model switch."""
    return not tool_filter_model or tool_filter_model == FILTER_MODEL_SYNC_SENTINEL


def build_event_callbacks(
    on_text: Optional[Callable[[TextEvent], None]] = None,
    on_thinking: Optional[Callable[[ThinkingEvent], None]] = None,
    on_tool_call: Optional[Callable[[ToolCallEvent], None]] = None,
    on_tool_result: Optional[Callable[[ToolResultEvent], None]] = None,
) -> EventCallbacks:
    """Wire QueryExecutor events to coden's rich UI.

    The four streaming callbacks are passed straight through to
    `EventCallbacks`; the caller owns the stream-renderer lifecycle so the
    Live region only stays open for the duration of one `executor.execute()`
    turn.
    """
    def _on_warning(msg: str) -> None:
        print_warning(msg)

    def _on_error(msg: str, error: Exception) -> None:
        print_error(f"{msg}: {format_exception_message(error)}")

    def _on_retry(attempt: int, _error: Exception) -> None:
        print_warning(f"Model produced an invalid tool call (retry {attempt})")

    return EventCallbacks(
        on_text=on_text,
        on_thinking=on_thinking,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_warning=_on_warning,
        on_error=_on_error,
        on_retry=_on_retry,
    )


def _show_debug_notification(
    debug_logger: DebugLogger,
    debug_enabled: bool,
    tool_permission_enabled: bool,
) -> None:
    """Show debug mode and tool permission status notifications."""
    if debug_enabled and debug_logger.get_log_path():
        console.print(
            f"[bold yellow]Debug mode enabled[/bold yellow] - "
            f"Logs: [cyan]{debug_logger.get_log_path()}[/cyan]"
        )
    permission_status = (
        "[bold green]enabled[/bold green]"
        if tool_permission_enabled
        else "[dim]disabled[/dim]"
    )
    console.print(f"Tool permission: {permission_status}")
    console.print()


def _display_token_usage(
    loop: InteractiveLoop,
    all_messages: list[ModelMessage],
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


def _suggest_debug_log(debug_logger: DebugLogger) -> None:
    log_path = debug_logger.get_log_path()
    if log_path:
        print_warning(f"See debug log: {log_path}")
    else:
        print_warning("Run with --debug flag for detailed logs")


def _handle_model_error(
    e: ModelHTTPError, debug_logger: DebugLogger, model_str: str
) -> None:
    """Handle model HTTP errors with helpful suggestions."""
    debug_logger.log_error(e, context="Query execution")
    body = e.body if isinstance(e.body, dict) else {}
    error_message = body.get("message", str(e))
    if is_tool_call_error(e):
        print_error(f"Model request failed (HTTP {e.status_code}): {error_message}")
        print_warning(
            f"The model '{model_str}' may not support tool calling properly. "
            "Maybe try a different model with better tool support."
        )
    else:
        print_error(f"Model error (HTTP {e.status_code}): {error_message}")
    _suggest_debug_log(debug_logger)


def _handle_generic_error(e: Exception, debug_logger: DebugLogger) -> None:
    """Handle generic errors with debug log suggestions."""
    debug_logger.log_error(e, context="Query execution")
    print_error(format_exception_message(e))
    _suggest_debug_log(debug_logger)


async def _apply_tool_filtering(
    context: CommandContext | None,
    user_input: str,
    debug_logger: DebugLogger,
) -> None:
    """Apply LLM-based tool routing if enabled.

    One router call per turn: the same `FilterResult` drives both the
    allow-list (via `ToolFilter.apply_allowlist`) and the user-facing
    display/log. Calling the router twice would double LLM latency and
    risk divergence between gated and displayed tools.
    """
    if not context or not context.dynamic_tool_filtering:
        return
    if not context.llm_filter or context.tool_router is None:
        return

    router = context.tool_router
    console.print("[dim]>> Routing tools...[/dim]")
    try:
        filter_result = await router.filter(user_input)
    except Exception:
        context.llm_filter.clear_filter()
        return

    names = {tool.metadata.name for tool in filter_result.all_tools}
    context.llm_filter.apply_allowlist(names)
    display_filtered_tools(filter_result, console)
    debug_logger.log_tool_routing(
        query=user_input,
        selected_tools=[t.metadata.name for t in filter_result.domain_tools],
        total_domain_tools=len(router.domain_tools),
    )


async def _run_executor(
    *,
    coding_agent: CodingAgent,
    pydantic_agent: "Agent[Any, str]",
    prompt: str,
    debug_logger: DebugLogger,
    message_history: list[ModelMessage],
    callbacks: EventCallbacks,
) -> AgentResponse:
    """Run one turn through `QueryExecutor`.

    The shared execution unit behind both the interactive renderer
    (`_execute_query`, Rich `StreamRenderer` callbacks) and the one-shot
    print path (`run_once`, plain-stdout callback). Only the callbacks and
    message history differ between the two; the executor wiring is identical.
    """
    executor = QueryExecutor(
        max_steps=coding_agent.max_steps,
        event_callbacks=callbacks,
        debug_logger=debug_logger,
    )
    return await executor.execute(
        pydantic_agent,
        prompt,
        message_history=message_history,
    )


async def _execute_query(
    *,
    coding_agent: CodingAgent,
    pydantic_agent: "Agent[Any, str]",
    prompt: str,
    debug_logger: DebugLogger,
    loop: InteractiveLoop,
    context: CommandContext | None,
) -> None:
    """Coden-side glue around the library-pure QueryExecutor.

    Wires the rich UI hooks (`event_callbacks`) and the live-settings
    provider, runs the executor, and renders the response. Compaction
    + token-usage live here so the executor stays library-pure.
    """
    await _apply_tool_filtering(context, prompt, debug_logger)

    wall_start = datetime.now()
    mono_start = time.monotonic()

    with StreamRenderer() as renderer:
        response = await _run_executor(
            coding_agent=coding_agent,
            pydantic_agent=pydantic_agent,
            prompt=prompt,
            debug_logger=debug_logger,
            message_history=loop.history,
            callbacks=build_event_callbacks(
                on_text=renderer.on_text,
                on_thinking=renderer.on_thinking,
                on_tool_call=renderer.on_tool_call,
                on_tool_result=renderer.on_tool_result,
            ),
        )

    outcome = await maybe_compact_history(
        response.messages, loop.tree, get_config().agent,
        debug_logger=debug_logger,
    )
    messages = response.messages
    steps = response.steps
    if outcome.happened:
        messages = outcome.messages
        steps = parse_messages_to_steps(messages)
        # Executor logged pre-compaction history; emit the post-compaction view
        # so the debug log shows what the next turn will actually see.
        debug_logger.log_message_history(messages)

    loop.update_history(messages)
    if steps:
        print_steps_rich(steps)
    if response.reached_max_steps:
        debug_logger.log_max_steps_reached(
            response.total_tool_calls, coding_agent.max_steps,
        )
        print_warning("Reached max steps limit, answer may be incomplete")
    if response.answer:
        AnswerRenderer().render(response.answer)
    else:
        console.print()
        print_warning("No response text generated")
        console.print()

    elapsed = time.monotonic() - mono_start
    _display_token_usage(
        loop, messages,
        elapsed_seconds=elapsed,
        wall_start=wall_start,
        wall_end=datetime.now(),
    )
    if outcome.bottom_line:
        console.print(outcome.bottom_line)
        console.print()


async def _run_query(
    *,
    coding_agent: CodingAgent,
    pydantic_agent: "Agent[Any, str]",
    user_input: str | SessionTrigger,
    root_directory: str,
    debug_logger: DebugLogger,
    loop: InteractiveLoop,
    context: CommandContext | None = None,
) -> None:
    """Execute a single query."""
    mode = get_mode_from_context(context)
    topic = context.study_topic if context else None
    prompt = build_query_prompt(user_input, root_directory, mode, topic)
    await _execute_query(
        coding_agent=coding_agent,
        pydantic_agent=pydantic_agent,
        prompt=prompt,
        debug_logger=debug_logger,
        loop=loop,
        context=context,
    )


def _build_session(
    *,
    coding_agent: CodingAgent,
    prompt_builder: PromptBuilder,
    server,
    root_directory: str,
    disabled_tools: list[str],
    debug_logger: DebugLogger,
    config,
    available_tools: list,
) -> tuple[CommandContext, Any]:
    """Build the agent's per-session state, shared by interactive and print modes.

    Wires the tool router / filtering toolset, the permission capability, the
    `CommandContext`, the system prompt, and rebuilds the pydantic agent. The
    `coding_agent` is mutated in place (toolsets / capabilities / system_prompt)
    exactly as the interactive path did inline. Returns `(context, pydantic_agent)`.

    `available_tools` is passed in so the caller lists tools once; this helper
    does no I/O. `on_model_switch` is deliberately NOT built here — it uses
    `nonlocal pydantic_agent` and must live in the caller's frame so the closure
    rebinds the caller's local, not this helper's.
    """
    max_retries = config.agent.max_retries
    ask_tool_permission = config.agent.ask_tool_permission

    tool_router = None
    filtering_toolset = None
    tool_filter_model_str = config.agent.tool_filter_model
    if config.agent.dynamic_tool_filtering:
        if not tool_filter_model_str:
            console.print(
                "[yellow]Tool filtering is ON but no filter model is set.\n"
                "  Use /filter-model <model> or /config set tool_filter_model model[/yellow]"
            )
        try:
            tool_metadata_list = [
                ToolMetadata(
                    name=tool.name,
                    description=tool.description or "",
                    query_description=TOOL_QUERY_DESCRIPTIONS.get(tool.name, ""),
                )
                for tool in available_tools
            ]
            router_model = resolve_filter_model(
                coding_agent, coding_agent.base_url, tool_filter_model_str,
            )
            tool_router = LLMToolRouter(tool_metadata_list, model=router_model)
            console.print("[dim]LLM tool routing enabled[/dim]")
        except Exception as e:
            escaped_msg = str(e).replace("[", "\\[")
            console.print(
                f"[yellow]Warning: Could not initialize tool router: {escaped_msg}[/yellow]"
            )

    llm_filter = None
    if tool_router is not None:
        # `filter_fn=None`: coden's `_apply_tool_filtering` calls the
        # router itself (single call) and pushes the allow-list via
        # `ToolFilter.apply_allowlist`. `set_filter_for_query` is unused.
        filtering_toolset, llm_filter = create_filtered_toolset(
            toolset=server,
            filter_fn=None,
        )
        base_toolset = filtering_toolset
    else:
        base_toolset = server

    async def _coden_picker(
        tool_name: str, args: dict[str, Any]
    ) -> Optional[PermissionChoice]:
        request = ToolPermissionRequest(tool_name=tool_name, tool_args=args)
        running_loop = asyncio.get_running_loop()
        # Rich Live (StreamRenderer) and prompt_toolkit both control the
        # terminal — running the picker while Live is active produces double
        # rendering and missing borders. Stop the Live first, restart after.
        live = get_active_live()
        live_was_stopped = False
        if live is not None:
            try:
                live.stop()
                live_was_stopped = True
            except Exception as e:
                # Live in a bad state; don't try to restart it later.
                logger.debug("get_active_live().stop() failed: %s", e)
                live = None
        try:
            return await running_loop.run_in_executor(
                None, run_tool_permission_picker, request
            )
        finally:
            if live is not None and live_was_stopped:
                try:
                    live.start()
                except Exception as e:
                    # Best-effort resume; if the display is gone, the next
                    # iteration's StreamRenderer will spin up a fresh one.
                    logger.debug("get_active_live().start() failed: %s", e)

    def _coden_is_permission_enabled() -> bool:
        return get_config().agent.ask_tool_permission

    def _coden_permission_notify(msg: str) -> None:
        console.print(f"[dim]{msg}[/dim]")

    permission_capability = PermissionCapability(
        is_enabled=_coden_is_permission_enabled,
        picker=_coden_picker,
        on_message=_coden_permission_notify,
    )

    context = CommandContext(
        model=coding_agent.model_str,
        base_url=coding_agent.base_url,
        max_steps=coding_agent.max_steps,
        max_retries=max_retries,
        debug=config.agent.debug,
        debug_logger=debug_logger,
        available_tools=available_tools,
        disabled_tools=set(disabled_tools),
        root_directory=root_directory,
        server=server,
        ask_tool_permission=ask_tool_permission,
        dynamic_tool_filtering=tool_router is not None,
        tool_filter_model=tool_filter_model_str,
        tool_router=tool_router,
    )

    context.filtering_toolset = filtering_toolset
    context.llm_filter = llm_filter

    coding_agent.toolsets = [base_toolset]
    coding_agent.capabilities = [permission_capability]

    system_prompt = prompt_builder.build(root_directory=root_directory)
    debug_logger.log_system_prompt(system_prompt)
    coding_agent.system_prompt = system_prompt

    coding_agent.rebuild_pydantic_agent()
    pydantic_agent = coding_agent.pydantic_agent

    return context, pydantic_agent


async def _run_interactive_session(
    *,
    coding_agent: CodingAgent,
    prompt_builder: PromptBuilder,
    server,
    root_directory: str,
    disabled_tools: list[str],
    debug_logger: DebugLogger,
    config,
    first_input: str | None = None,
) -> None:
    """Run the main interactive session loop."""
    available_tools = await server.list_tools()
    debug_logger.log_session_start(
        model=coding_agent.model_str,
        base_url=coding_agent.base_url,
        max_steps=coding_agent.max_steps,
    )

    context, pydantic_agent = _build_session(
        coding_agent=coding_agent,
        prompt_builder=prompt_builder,
        server=server,
        root_directory=root_directory,
        disabled_tools=disabled_tools,
        debug_logger=debug_logger,
        config=config,
        available_tools=available_tools,
    )

    # `on_model_switch` lives here (not in `_build_session`) so its
    # `nonlocal pydantic_agent` rebinds this frame's local on /model.
    def on_model_switch(new_model: str) -> None:
        nonlocal pydantic_agent
        coding_agent.model_str = new_model
        coding_agent.rebuild_pydantic_agent()
        pydantic_agent = coding_agent.pydantic_agent
        if (context.tool_router is not None
                and should_sync_router_on_model_switch(context.tool_filter_model)):
            router_model = resolve_filter_model(
                coding_agent, coding_agent.base_url, context.tool_filter_model,
            )
            context.tool_router.update_model(router_model)

    loop = InteractiveLoop(context, on_model_switch=on_model_switch)
    pending_input = first_input

    while True:
        try:
            if pending_input is not None:
                user_input: str = pending_input
                pending_input = None
            else:
                maybe_input = await loop.get_input()
                if maybe_input is None:
                    continue
                user_input = maybe_input

            cmd_result = await loop.process_command(user_input)
            if cmd_result.should_exit:
                break

            if cmd_result.should_continue:
                coding_agent.max_steps = context.max_steps
                if context.debug_logger is not debug_logger:
                    debug_logger = context.debug_logger
                needs_rebuild = (
                    cmd_result.study_mode_changed or
                    cmd_result.directory_changed or
                    cmd_result.config_changed
                )
                if needs_rebuild:
                    system_prompt = prompt_builder.build(
                        root_directory=context.root_directory,
                        study_mode=context.study_mode,
                        study_topic=context.study_topic,
                        refresh_tree=cmd_result.directory_changed,
                    )
                    debug_logger.log_system_prompt(system_prompt)
                    coding_agent.system_prompt = system_prompt
                    coding_agent.rebuild_pydantic_agent()
                    pydantic_agent = coding_agent.pydantic_agent
                    if context.tool_router is not None:
                        router_model = resolve_filter_model(
                            coding_agent, coding_agent.base_url,
                            context.tool_filter_model,
                        )
                        context.tool_router.update_model(router_model)
                    if context.study_mode:
                        try:
                            await _run_query(
                                coding_agent=coding_agent,
                                pydantic_agent=pydantic_agent,
                                user_input=SessionTrigger.BEGIN,
                                root_directory=context.root_directory,
                                debug_logger=debug_logger,
                                loop=loop,
                                context=context,
                            )
                        except (KeyboardInterrupt, asyncio.CancelledError):
                            console.print("\n[dim]Response interrupted.[/dim]")
                continue

            is_shell_prompt = cmd_result.shell_to_llm is not None
            if is_shell_prompt:
                user_input = cmd_result.shell_to_llm

            if not user_input:
                continue

            if not is_shell_prompt and find_file_references(user_input):
                user_input = expand_file_references(
                    user_input, context.root_directory,
                )

            try:
                await _run_query(
                    coding_agent=coding_agent,
                    pydantic_agent=pydantic_agent,
                    user_input=user_input,
                    root_directory=context.root_directory,
                    debug_logger=debug_logger,
                    loop=loop,
                    context=context,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                console.print("\n[dim]Response interrupted.[/dim]")

        except KeyboardInterrupt:
            console.print()
            break
        except ModelHTTPError as e:
            _handle_model_error(e, debug_logger, coding_agent.model_str)
        except Exception as e:
            _handle_generic_error(e, debug_logger)


@asynccontextmanager
async def _serve_mcp(disabled_tools: list[str] | None, config):
    """Run the MCP server in a background task; yield it once ready.

    Owns the server-task / shutdown-event / cleanup lifecycle so both
    `run_interactive` and `run_once` share one correct teardown:
    - any startup error is re-raised *before* the server is yielded;
    - the `finally` cancels the task even if the `async with` body raises,
      so a model error in `run_once` never leaks the server task.
    """
    server = create_mcp_server(
        disabled_tools=disabled_tools,
        timeout=config.agent.mcp_server_timeout,
        max_retries=config.agent.max_retries,
        tool_timeout=config.agent.tool_timeout,
    )

    server_ready = asyncio.Event()
    shutdown_event = asyncio.Event()
    server_error: list[BaseException | None] = [None]

    async def server_manager():
        try:
            async with server:
                server_ready.set()
                await shutdown_event.wait()
        except BaseException as e:
            server_error[0] = e
            server_ready.set()

    server_task = asyncio.create_task(server_manager())
    await server_ready.wait()

    if server_error[0] is not None:
        raise server_error[0]

    try:
        yield server
    finally:
        shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=2.0)
        except asyncio.TimeoutError:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


async def run_interactive(
    root_directory: str,
    model: str,
    base_url: Optional[str] = None,
    max_steps: int = 10,
    disabled_tools: list[str] | None = None,
    start_daemon: bool = True,
) -> None:
    """Entry point for the agent CLI.

    Args:
        root_directory: Absolute path to the project root.
        model: Model identifier string.
        base_url: Optional base URL for OpenAI-compatible API.
        max_steps: Maximum number of tool calls per query.
        disabled_tools: Optional list of tool names to disable.
        start_daemon: Whether to auto-start the daemon (gated by
            `daemon_enabled` upstream).
    """
    from .input_prompt import create_prompt_session, get_user_input_async

    config = load_config()

    coding_agent = CodingAgent(
        model=model,
        base_url=base_url,
        max_steps=max_steps,
        settings_provider=lambda: get_config().model.generation,
    )
    prompt_builder = PromptBuilder(
        include_tool_instructions=False,
        use_config_for_tool_instructions=True,
    )

    debug_logger = create_debug_logger(root_directory, debug=config.agent.debug)
    daemon_started = start_daemon_async() if start_daemon else False

    try:
        async with _serve_mcp(disabled_tools, config) as server:
            available_tools = await server.list_tools()
            print_welcome(
                root_directory,
                coding_agent.max_steps,
                tool_count=len(available_tools),
                model_name=coding_agent.model_str,
                base_url=coding_agent.base_url,
            )
            _show_debug_notification(
                debug_logger, config.agent.debug, config.agent.ask_tool_permission
            )

            prompt_session = create_prompt_session(lambda: root_directory)
            first_input = await get_user_input_async(prompt_session)

            try:
                await _run_interactive_session(
                    coding_agent=coding_agent,
                    prompt_builder=prompt_builder,
                    server=server,
                    root_directory=root_directory,
                    disabled_tools=disabled_tools or [],
                    debug_logger=debug_logger,
                    config=config,
                    first_input=first_input,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            except BaseException as e:
                debug_logger.log_error(e, context="Session terminated")
                print_fatal_error(e, show_traceback=True)
    finally:
        debug_logger.close()
        print_goodbye()
        if daemon_started:
            try:
                stop_daemon()
            except KeyboardInterrupt:
                pass


async def run_once(
    root_directory: str,
    model: str,
    base_url: Optional[str] = None,
    max_steps: int = 10,
    prompt: str = "",
    *,
    disabled_tools: list[str] | None = None,
    start_daemon: bool = True,
) -> int:
    """Run a single prompt non-interactively, stream the answer, and exit.

    The `coden -a -p` path. Reuses the interactive turn machinery
    (`_serve_mcp`, `_build_session`, `_run_executor`) but renders to plain
    stdout instead of a Rich `Live` region, auto-allows tools (no picker —
    there's no TTY in a pipe), and keeps the daemon running for the next call.

    Returns the process exit code: 0 on a produced answer, 1 otherwise.
    """
    # The cached singleton is the object the permission gate
    # (`_coden_is_permission_enabled`) reads via `get_config()`. Mutating it
    # here is what actually disables the picker; a fresh `load_config()` copy
    # would not be observed by the gate. Process-local, never persisted.
    config = get_config()
    config.agent.ask_tool_permission = False

    coding_agent = CodingAgent(
        model=model,
        base_url=base_url,
        max_steps=max_steps,
        settings_provider=lambda: get_config().model.generation,
    )
    prompt_builder = PromptBuilder(
        include_tool_instructions=False,
        use_config_for_tool_instructions=True,
    )
    debug_logger = create_debug_logger(root_directory, debug=config.agent.debug)

    # Gated by daemon_enabled upstream; never stopped here so repeated `-p`
    # calls reuse the warm daemon.
    if start_daemon:
        start_daemon_async()

    # Tool filtering is skipped: it prints to the Rich console (would pollute
    # stdout) and is an interactive-latency optimization only.
    writer = StdoutStreamWriter()
    tool_reporter = StderrToolReporter()
    try:
        try:
            async with _serve_mcp(disabled_tools, config) as server:
                available_tools = await server.list_tools()
                context, pydantic_agent = _build_session(
                    coding_agent=coding_agent,
                    prompt_builder=prompt_builder,
                    server=server,
                    root_directory=root_directory,
                    disabled_tools=disabled_tools or [],
                    debug_logger=debug_logger,
                    config=config,
                    available_tools=available_tools,
                )

                built = build_query_prompt(
                    prompt,
                    root_directory,
                    get_mode_from_context(context),
                    context.study_topic,
                )
                response = await _run_executor(
                    coding_agent=coding_agent,
                    pydantic_agent=pydantic_agent,
                    prompt=built,
                    debug_logger=debug_logger,
                    message_history=[],
                    callbacks=build_event_callbacks(
                        on_text=writer.on_text,
                        on_tool_call=tool_reporter.on_tool_call,
                        on_tool_result=tool_reporter.on_tool_result,
                    ),
                )
        except Exception as exc:
            # A model/network error (or an empty agent run) propagates out of
            # the executor. A one-shot CLI must not surface a raw traceback:
            # terminate any half-streamed line, report on stderr, exit non-zero.
            # KeyboardInterrupt is BaseException, so Ctrl-C still propagates.
            if writer.wrote_any:
                sys.stdout.write("\n")
                sys.stdout.flush()
            print(f"Agent run failed: {exc}", file=sys.stderr)
            return 1

        sys.stdout.write("\n")
        sys.stdout.flush()

        if response.answer:
            return 0
        if response.reached_max_steps:
            print("Reached max steps before producing an answer.", file=sys.stderr)
        else:
            print("No response text generated.", file=sys.stderr)
        return 1
    finally:
        debug_logger.close()


