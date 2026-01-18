"""Pydantic-AI coding agent with MCP tools and ReAct reasoning.

Uses coden-retriever MCP server for code search and analysis capabilities.
Displays reasoning steps (Thought -> Action -> Observation) for transparency.

Supported model formats:
- "ollama:model_name" - Ollama server (e.g. ollama:qwen2.5-coder:14b)
- "llamacpp:model_name" - llama-cpp-server (default: localhost:8080)
  Run: llama-cpp-server -m model.gguf --port 8080
  Docs: https://github.com/ggerganov/llama.cpp/tree/master/tools/server
- "openai:model_name" - Official OpenAI API (requires OPENAI_API_KEY env var)
- "model_name" with --base-url - Any OpenAI-compatible endpoint (LM Studio, vLLM, etc.)
"""

import asyncio
from typing import Any, AsyncIterator, Callable, Optional

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.settings import ModelSettings

from ..config_loader import GenerationSettings, get_config, load_config
from ..daemon import start_daemon_async, stop_daemon
from ..mcp.tool_filter import ToolFilter, ToolMetadata
from .debug_logger import DebugLogger, create_debug_logger
from .filtering_toolset import create_filtered_toolset
from .interactive_loop import CommandContext, InteractiveLoop
from .mcp_server import create_mcp_server
from .model_factory import ModelFactory
from .models import AgentMode, AgentResponse, ReActStep
from .permission_toolset import wrap_toolset_with_permission
from .prompt_builder import PromptBuilder
from .query_executor import ErrorHandler, QueryExecutor
from .react_loop import parse_messages_to_steps, run_with_react_display
from .rich_console import (
    console,
    print_fatal_error,
    print_goodbye,
    print_welcome,
)
from .text_tool_fallback import handle_fallback_iteration
from ..constants import DEFAULT_MAX_RETRIES


# AgentMode is imported from models.py for proper dependency injection support


def get_mode_from_context(context: "CommandContext | None") -> AgentMode:
    """Determine agent mode from context."""
    if context and context.study_mode:
        return AgentMode.STUDY
    return AgentMode.CODING


# Session start triggers for study mode
_SESSION_START_TRIGGERS = frozenset(("", "start", "begin"))
_SESSION_START_PREFIXES = ("begin the study session", "start the study session")


def build_query_prompt(
    user_input: str,
    root_directory: str,
    mode: AgentMode,
    topic: str | None = None,
) -> str:
    """Build the query prompt based on agent mode.

    Args:
        user_input: The user's query.
        root_directory: Working directory path.
        mode: Current agent mode (CODING or STUDY).
        topic: Optional focus topic for STUDY mode.

    Returns:
        Complete prompt string for the agent.
    """
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


class CodingAgent:
    """Interactive coding agent using pydantic-ai with MCP tools and ReAct reasoning."""

    def __init__(
        self,
        model: str = "ollama:",
        base_url: Optional[str] = None,
        max_steps: int = 10,
        tool_instructions: bool = True,
        generation: Optional[GenerationSettings] = None,
    ):
        """Initialize the coding agent.

        Args:
            model: Model identifier. Formats supported:
                - "ollama:model_name" - Ollama server (e.g. ollama:qwen2.5-coder:14b)
                - "llamacpp:model_name" - llama-cpp-server (default: localhost:8080)
                - "openai:model_name" - Official OpenAI API (requires OPENAI_API_KEY)
                - "model_name" - OpenAI-compatible endpoint (requires base_url)
            base_url: Base URL for OpenAI-compatible API (not needed for openai: prefix).
            max_steps: Maximum number of tool calls per query (default: 10).
            tool_instructions: Include detailed tool workflow instructions in prompt (default: True).
            generation: Generation settings (temperature, max_tokens, timeout, api_key).
        """
        self.model_str = model
        self.base_url = base_url
        self.max_steps = max_steps
        self.tool_instructions = tool_instructions
        self.generation = generation or GenerationSettings()

        self._model_factory = ModelFactory(model, base_url, self.generation)
        self._prompt_builder = PromptBuilder(
            include_tool_instructions=tool_instructions,
            use_config_for_tool_instructions=True,
        )

    def _build_model_settings(self) -> ModelSettings:
        """Build ModelSettings dict from current config.

        Reads from the config cache so that changes via /config set
        are applied immediately without requiring restart.
        """
        config = get_config()
        gen = config.model.generation
        settings: ModelSettings = {
            "temperature": gen.temperature,
            "timeout": gen.timeout,
        }
        if gen.max_tokens is not None:
            settings["max_tokens"] = gen.max_tokens
        return settings

    def _get_model(self):
        """Get cached model instance."""
        return self._model_factory.get_model()

    def _create_model(self):
        """Create a new model instance. For backwards compatibility with tests."""
        return self._model_factory._create_model()

    def _get_directory_tree(self, root_directory: str, refresh: bool = False) -> str:
        """Get cached directory tree."""
        return self._prompt_builder.get_directory_tree(root_directory, refresh=refresh)

    def _build_system_prompt(
        self,
        root_directory: str,
        study_mode: bool = False,
        study_topic: str | None = None,
        refresh_tree: bool = False,
    ) -> str:
        """Build system prompt with cached directory tree."""
        return self._prompt_builder.build(
            root_directory=root_directory,
            study_mode=study_mode,
            study_topic=study_topic,
            refresh_tree=refresh_tree,
        )

    async def _create_agent_context(
        self,
        root_directory: str,
        study_mode: bool = False,
        study_topic: str | None = None,
        disabled_tools: list[str] | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> tuple:
        """Create MCP server and agent with shared setup logic.

        Returns:
            Tuple of (server, agent, system_prompt) for use in run methods.
        """
        server = create_mcp_server(
            disabled_tools=disabled_tools,
            timeout=timeout,
            max_retries=max_retries,
        )
        system_prompt = self._build_system_prompt(
            root_directory,
            study_mode=study_mode,
            study_topic=study_topic,
        )
        agent = Agent(
            self._get_model(),
            system_prompt=system_prompt,
            toolsets=[server],
            retries=max_retries,
            model_settings=self._build_model_settings(),
        )
        return server, agent, system_prompt

    async def run(
        self,
        prompt: str,
        root_directory: str,
        message_history: list | None = None,
    ) -> AgentResponse:
        """Run a single query and return structured response.

        Args:
            prompt: User query to process.
            root_directory: Absolute path to the project root.
            message_history: Optional conversation history for multi-turn.

        Returns:
            AgentResponse with answer, reasoning steps, and message history.
        """
        server, agent, _ = await self._create_agent_context(root_directory)

        async with server:
            full_prompt = f"Working directory: {root_directory}\n\n{prompt}"
            return await run_with_react_display(
                agent=agent,
                prompt=full_prompt,
                message_history=message_history,
                max_steps=self.max_steps,
                server=server,
            )

    async def run_stream(
        self,
        prompt: str,
        root_directory: str,
        message_history: list | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        """Run a query with streaming text output.

        Includes fallback support for models that output tool calls as text
        instead of using proper function calling format.

        Args:
            prompt: User query to process.
            root_directory: Absolute path to the project root.
            message_history: Optional conversation history for multi-turn.
            on_text_chunk: Callback for each text chunk.

        Returns:
            AgentResponse with answer, reasoning steps, and message history.
        """
        server, agent, _ = await self._create_agent_context(root_directory)

        async with server:
            full_prompt = f"Working directory: {root_directory}\n\n{prompt}"
            current_history = message_history
            fallback_steps: list[ReActStep] = []
            fallback_tool_calls = 0
            all_messages = []
            steps = []

            for iteration in range(self.max_steps):
                streamed_text = ""

                async with agent.run_stream(
                    full_prompt if iteration == 0 else "",
                    message_history=current_history,
                    usage_limits=UsageLimits(request_limit=self.max_steps * 2),
                ) as result:
                    async for text in result.stream_text():
                        streamed_text += text
                        if on_text_chunk:
                            on_text_chunk(text)
                    final_output = await result.get_output()

                all_messages = result.all_messages()
                steps = parse_messages_to_steps(all_messages)
                total_tool_calls = sum(1 for step in steps if step.action is not None)
                answer_text = str(final_output) if final_output else streamed_text

                fallback_result = await handle_fallback_iteration(
                    server=server,
                    answer_text=answer_text,
                    total_tool_calls=total_tool_calls,
                    all_messages=all_messages,
                    step_number_start=len(fallback_steps),
                )
                if fallback_result.should_continue:
                    fallback_steps.extend(fallback_result.steps)
                    fallback_tool_calls += fallback_result.tool_call_count
                    current_history = fallback_result.updated_history
                    full_prompt = fallback_result.continuation_prompt
                    continue

                return AgentResponse(
                    answer=str(final_output),
                    steps=fallback_steps + steps,
                    total_tool_calls=fallback_tool_calls + total_tool_calls,
                    reached_max_steps=(fallback_tool_calls + total_tool_calls) >= self.max_steps,
                    messages=all_messages,
                )

            return AgentResponse(
                answer="[Max iterations reached]",
                steps=fallback_steps + steps,
                total_tool_calls=fallback_tool_calls,
                reached_max_steps=True,
                messages=all_messages,
            )

    async def stream_text(
        self,
        prompt: str,
        root_directory: str,
        message_history: list | None = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks from agent response.

        Args:
            prompt: User query to process.
            root_directory: Absolute path to the project root.
            message_history: Optional conversation history for multi-turn.

        Yields:
            Text chunks as they arrive from the model.
        """
        server, agent, _ = await self._create_agent_context(root_directory)

        async with server:
            full_prompt = f"Working directory: {root_directory}\n\n{prompt}"

            async with agent.run_stream(
                full_prompt,
                message_history=message_history,
                usage_limits=UsageLimits(request_limit=self.max_steps * 2),
            ) as result:
                async for text in result.stream_text():
                    yield text

    async def run_interactive(
        self,
        root_directory: str,
        disabled_tools: list[str] | None = None,
    ) -> None:
        """Run interactive REPL loop with ReAct reasoning display.

        Args:
            root_directory: Absolute path to the project root.
            disabled_tools: Optional list of tool names to disable.
        """
        from .input_prompt import create_prompt_session, get_user_input_async

        config = self._load_config()
        debug_logger = create_debug_logger(root_directory, debug=config.agent.debug)

        # Start daemon in background for fast code search
        daemon_started = start_daemon_async()

        server = create_mcp_server(
            disabled_tools=disabled_tools,
            timeout=config.agent.mcp_server_timeout,
            max_retries=config.agent.max_retries,
        )

        server_ready = asyncio.Event()
        shutdown_event = asyncio.Event()
        server_error: list[BaseException | None] = [None]

        async def server_manager():
            """Manage server lifecycle in its own task."""
            try:
                async with server:
                    server_ready.set()
                    await shutdown_event.wait()  # Keep alive until shutdown
            except BaseException as e:
                server_error[0] = e
                server_ready.set()

        server_task = asyncio.create_task(server_manager())
        await server_ready.wait()

        if server_error[0] is not None:
            raise server_error[0]

        available_tools = await server.list_tools()
        print_welcome(
            root_directory,
            self.max_steps,
            tool_count=len(available_tools),
            model_name=self.model_str,
            base_url=self.base_url,
        )
        self._show_debug_notification(
            debug_logger, config.agent.debug, config.agent.ask_tool_permission
        )

        prompt_session = create_prompt_session(lambda: root_directory)
        first_input = await get_user_input_async(prompt_session)

        try:
            await self._run_interactive_session(
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
            # Signal server to shutdown and wait for clean exit
            shutdown_event.set()
            try:
                await asyncio.wait_for(server_task, timeout=2.0)
            except asyncio.TimeoutError:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            debug_logger.close()
            print_goodbye()
            # Stop daemon when exiting agent mode
            if daemon_started:
                stop_daemon()

    def _load_config(self):
        """Load configuration settings."""
        from ..config_loader import get_config
        return get_config()

    async def _run_interactive_session(
        self,
        server,
        root_directory: str,
        disabled_tools: list[str],
        debug_logger,
        config,
        first_input: str | None = None,
    ) -> None:
        """Run the main interactive session loop."""
        available_tools = await server.list_tools()
        max_retries = config.agent.max_retries
        ask_tool_permission = config.agent.ask_tool_permission

        debug_logger.log_session_start(
            model=self.model_str,
            base_url=self.base_url,
            max_steps=self.max_steps,
        )

        # Create tool filter if dynamic filtering is enabled
        tool_filter = None
        filtering_toolset = None
        if config.agent.dynamic_tool_filtering:
            try:
                tool_metadata_list = [
                    ToolMetadata(name=tool.name, description=tool.description or "")
                    for tool in available_tools
                ]
                tool_filter = ToolFilter(tool_metadata_list)
                console.print("[dim]Dynamic tool filtering enabled[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not initialize tool filter: {e}[/yellow]")

        # Wrap toolset with filtering if tool_filter is available
        semantic_filter = None
        if tool_filter is not None:
            filtering_toolset, semantic_filter = create_filtered_toolset(
                toolset=server,
                tool_filter=tool_filter,
                threshold=config.agent.tool_filter_threshold,
            )
            base_toolset = filtering_toolset
        else:
            base_toolset = server

        # Create permission-wrapped toolset (permission checking controlled by config)
        toolset = wrap_toolset_with_permission(toolset=base_toolset)

        context = CommandContext(
            model=self.model_str,
            base_url=self.base_url,
            max_steps=self.max_steps,
            max_retries=max_retries,
            debug=config.agent.debug,
            debug_logger=debug_logger,
            available_tools=available_tools,
            disabled_tools=set(disabled_tools),
            root_directory=root_directory,
            server=server,
            toolset=toolset,
            ask_tool_permission=ask_tool_permission,
            dynamic_tool_filtering=tool_filter is not None,
            tool_filter=tool_filter,
            tool_filter_threshold=config.agent.tool_filter_threshold,
        )

        # Store filtering toolset and semantic filter in context for per-query updates
        context.filtering_toolset = filtering_toolset
        context.semantic_filter = semantic_filter

        # Build initial system prompt and agent (uses instance caching)
        system_prompt = self._build_system_prompt(root_directory)
        debug_logger.log_system_prompt(system_prompt)

        agent = Agent(
            self._get_model(),
            system_prompt=system_prompt,
            toolsets=[toolset],
            retries=max_retries,
            model_settings=self._build_model_settings(),
        )

        def on_model_switch(new_model: str) -> None:
            nonlocal agent
            self.model_str = new_model
            self._model_factory.model_str = new_model
            self._model_factory.clear_cache()
            agent = Agent(
                self._get_model(),
                system_prompt=system_prompt,
                toolsets=[toolset],
                retries=context.max_retries,
                model_settings=self._build_model_settings(),
            )

        loop = InteractiveLoop(context, on_model_switch=on_model_switch)
        pending_input = first_input

        # Main REPL loop
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
                    self._sync_settings_from_context(context)
                    # Sync debug_logger if it was changed by /debug command
                    if context.debug_logger is not debug_logger:
                        debug_logger = context.debug_logger
                    # Rebuild agent when mode, directory, or config changes
                    needs_rebuild = (
                        cmd_result.study_mode_changed or
                        cmd_result.directory_changed or
                        cmd_result.config_changed
                    )
                    if needs_rebuild:
                        system_prompt = self._build_system_prompt(
                            context.root_directory,
                            study_mode=context.study_mode,
                            study_topic=context.study_topic,
                            refresh_tree=cmd_result.directory_changed,
                        )
                        debug_logger.log_system_prompt(system_prompt)
                        agent = Agent(
                            self._get_model(),
                            system_prompt=system_prompt,
                            toolsets=[toolset],
                            retries=context.max_retries,
                            model_settings=self._build_model_settings(),
                        )
                        # Auto-start study session when entering study mode
                        if context.study_mode:
                            await self._run_query(
                                agent=agent,
                                user_input="begin",  # Triggers SESSION START assessment
                                root_directory=context.root_directory,
                                debug_logger=debug_logger,
                                loop=loop,
                                context=context,
                            )
                    continue

                if not user_input:
                    continue

                await self._run_query(
                    agent=agent,
                    user_input=user_input,
                    root_directory=context.root_directory,
                    debug_logger=debug_logger,
                    loop=loop,
                    context=context,
                )

            except KeyboardInterrupt:
                console.print()
                break
            except ModelHTTPError as e:
                self._handle_model_error(e, debug_logger)
            except Exception as e:
                self._handle_generic_error(e, debug_logger)

    def _show_debug_notification(
        self, debug_logger, debug_enabled: bool, tool_permission_enabled: bool
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

    def _sync_settings_from_context(self, context: CommandContext) -> None:
        """Sync runtime settings from command context."""
        self.max_steps = context.max_steps

    async def _run_query(
        self,
        agent: "Agent[Any, str]",
        user_input: str,
        root_directory: str,
        debug_logger: "DebugLogger",
        loop: InteractiveLoop,
        context: CommandContext | None = None,
    ) -> None:
        """Execute a single query with streaming and logging."""
        mode = get_mode_from_context(context)
        topic = context.study_topic if context else None
        prompt = build_query_prompt(user_input, root_directory, mode, topic)

        executor = QueryExecutor(self.max_steps, self.model_str)
        await executor.execute(agent, prompt, debug_logger, loop, context)

    def _handle_model_error(self, e: ModelHTTPError, debug_logger: DebugLogger) -> None:
        """Handle model HTTP errors."""
        ErrorHandler(self.model_str).handle_model_error(e, debug_logger)

    def _handle_generic_error(self, e: Exception, debug_logger: DebugLogger) -> None:
        """Handle generic errors."""
        ErrorHandler(self.model_str).handle_generic_error(e, debug_logger)


async def run_interactive(
    root_directory: str,
    model: str,
    base_url: Optional[str] = None,
    max_steps: int = 10,
    disabled_tools: list[str] | None = None,
    generation: Optional[GenerationSettings] = None,
) -> None:
    """Entry point for CLI.

    Args:
        root_directory: Absolute path to the project root.
        model: Model identifier string.
        base_url: Optional base URL for OpenAI-compatible API.
        max_steps: Maximum number of tool calls per query.
        disabled_tools: Optional list of tool names to disable.
        generation: Generation settings (temperature, max_tokens, timeout, api_key).
                   If None, uses config defaults.
    """
    config = load_config()

    # Merge CLI overrides with config defaults
    effective_generation = generation or config.model.generation

    agent = CodingAgent(
        model=model,
        base_url=base_url,
        max_steps=max_steps,
        generation=effective_generation,
    )
    await agent.run_interactive(root_directory, disabled_tools=disabled_tools)
