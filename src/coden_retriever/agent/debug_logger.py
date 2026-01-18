"""Debug logging for the coding agent.

Provides comprehensive logging when debug mode is enabled:
- Complete system prompts
- All tool calls with timestamps
- Model responses and thinking traces
- Streamed to ~/.coden-retriever/{project_key}/logs/ directory

Uses Python's RotatingFileHandler for robust log rotation.
"""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from ..config import get_project_cache_dir

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    TextPart,
    UserPromptPart,
    SystemPromptPart,
    ThinkingPart,
)


MAX_DEBUG_LOG_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DEBUG_LOG_FILES = 10


class _PassthroughFormatter(logging.Formatter):
    """Formatter that outputs the message as-is without any formatting."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the message without any modification."""
        return record.getMessage()


class DebugLogger:
    """Debug logger that streams detailed agent execution logs to a file.

    When enabled, logs:
    - Session start with configuration
    - Complete system prompts
    - User prompts with timestamps
    - All tool calls with arguments and timestamps
    - Tool results/errors
    - Model responses (text and thinking traces)
    - Session summary on close

    Uses Python's RotatingFileHandler for automatic size-based rotation
    and cleanup of old log files.
    """

    LOGS_DIR = "logs"

    def __init__(self, root_directory: str, enabled: bool = False):
        """Initialize the debug logger.

        Args:
            root_directory: Project root directory (used to determine cache location)
            enabled: Whether debug logging is enabled
        """
        self.enabled = enabled
        self.root_directory = Path(root_directory)
        self._logger: Optional[logging.Logger] = None
        self._logger_name: Optional[str] = None
        self._handler: Optional[RotatingFileHandler] = None
        self.log_path: Optional[Path] = None
        self.session_start: Optional[datetime] = None
        self.tool_call_count = 0
        self.query_count = 0

        if self.enabled:
            self._init_log_file()

    def _get_logs_dir(self) -> Path:
        """Get the logs directory path in centralized cache."""
        return get_project_cache_dir(self.root_directory) / self.LOGS_DIR

    def _init_log_file(self) -> None:
        """Initialize the log file with timestamp-based name using RotatingFileHandler."""
        logs_dir = self._get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now()
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.log_path = logs_dir / f"debug_{timestamp}.log"

        # Create a unique logger for this session to avoid conflicts
        self._logger_name = f"debug_logger_{timestamp}_{id(self)}"
        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(logging.DEBUG)
        # Prevent propagation to root logger
        self._logger.propagate = False

        # Set up rotating file handler
        # Note: backupCount handles cleanup of old rotated files (e.g., .log.1, .log.2)
        # but not the timestamp-based files from previous sessions
        self._handler = RotatingFileHandler(
            filename=str(self.log_path),
            maxBytes=MAX_DEBUG_LOG_SIZE,
            backupCount=MAX_DEBUG_LOG_FILES - 1,
            encoding="utf-8",
        )
        self._handler.setLevel(logging.DEBUG)
        self._handler.setFormatter(_PassthroughFormatter())

        self._logger.addHandler(self._handler)

        # Clean up old session log files (timestamp-based files from previous sessions)
        self._cleanup_old_logs()

    def _rotate_log(self) -> None:
        """Force a log rotation.

        Note: With RotatingFileHandler, rotation is automatic based on size.
        This method is provided for manual rotation if needed (e.g., testing).
        Session start is preserved to maintain accurate duration tracking.
        """
        if self._handler:
            self._handler.doRollover()

    def _cleanup_old_logs(self) -> None:
        """Remove oldest session log files if exceeding MAX_DEBUG_LOG_FILES.

        Note: This cleans up timestamp-based log files from previous sessions.
        The RotatingFileHandler's backupCount handles rotated files within a session.
        """
        logs_dir = self._get_logs_dir()
        if not logs_dir.exists():
            return

        # Get list of log files with their mtime, handling potential race conditions
        log_files_with_mtime: list[tuple[Path, float]] = []
        for p in logs_dir.glob("debug_*.log"):
            try:
                # Skip the current log file and its rotated versions (e.g., .log.1, .log.2)
                if self.log_path and (
                    p.resolve() == self.log_path.resolve() or
                    str(p).startswith(str(self.log_path) + ".")
                ):
                    continue
                mtime = p.stat().st_mtime
                log_files_with_mtime.append((p, mtime))
            except OSError:
                # File was deleted between glob and stat, skip it
                continue

        # Sort by mtime (oldest first)
        log_files_with_mtime.sort(key=lambda x: x[1])

        # Account for current log file in count (if it exists)
        current_log_count = 1 if self.log_path and self.log_path.exists() else 0

        # Delete oldest files to stay under limit
        while len(log_files_with_mtime) + current_log_count > MAX_DEBUG_LOG_FILES:
            if not log_files_with_mtime:
                break
            oldest, _ = log_files_with_mtime.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass  # Ignore errors deleting old logs (file might already be gone)

    def get_log_path(self) -> Optional[Path]:
        """Get the current log file path."""
        return self.log_path

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _write(self, content: str) -> None:
        """Write content to log file."""
        if self._logger:
            # Log each line separately to ensure proper handling
            self._logger.debug(content.rstrip('\n'))

    def _write_section(self, title: str, content: str = "") -> None:
        """Write a titled section to the log."""
        self._write(f"\n{'='*80}")
        self._write(f"[{self._timestamp()}] {title}")
        self._write(f"{'='*80}")
        if content:
            self._write(content)

    def _write_subsection(self, title: str, content: str = "") -> None:
        """Write a subsection to the log."""
        self._write(f"\n{'-'*40}")
        self._write(f"[{self._timestamp()}] {title}")
        self._write(f"{'-'*40}")
        if content:
            self._write(content)

    def log_session_start(
        self,
        model: str,
        base_url: Optional[str],
        max_steps: int,
    ) -> None:
        """Log session start with configuration."""
        if not self.enabled:
            return

        self._write_section("DEBUG SESSION START")
        self._write(f"Timestamp: {self._timestamp()}")
        self._write(f"Root Directory: {self.root_directory}")
        self._write(f"Model: {model}")
        self._write(f"Base URL: {base_url or '(default)'}")
        self._write(f"Max Steps: {max_steps}")
        self._write(f"Log File: {self.log_path}")

    def log_system_prompt(self, system_prompt: str) -> None:
        """Log the complete system prompt."""
        if not self.enabled:
            return

        self._write_section("SYSTEM PROMPT")
        self._write(system_prompt)

    def log_user_prompt(self, prompt: str) -> None:
        """Log a user prompt/query."""
        if not self.enabled:
            return

        self.query_count += 1
        self._write_section(f"USER QUERY #{self.query_count}")
        self._write(prompt)

    def log_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Log a tool call with arguments."""
        if not self.enabled:
            return

        self.tool_call_count += 1
        self._write_subsection(f"TOOL CALL #{self.tool_call_count}: {tool_name}")
        if tool_call_id:
            self._write(f"Call ID: {tool_call_id}")
        self._write("Arguments:")
        try:
            formatted_args = json.dumps(tool_args, indent=2, default=str)
            self._write(formatted_args)
        except Exception:
            self._write(str(tool_args))

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Log a tool result."""
        if not self.enabled:
            return

        status = "SUCCESS" if success else "ERROR"
        self._write_subsection(f"TOOL RESULT ({status}): {tool_name}")
        if tool_call_id:
            self._write(f"Call ID: {tool_call_id}")

        result_str = str(result)
        # Log full result for debugging
        self._write(f"Result:\n{result_str}")

    def log_model_response(self, response_text: str, is_final: bool = False) -> None:
        """Log model response text."""
        if not self.enabled:
            return

        title = "FINAL RESPONSE" if is_final else "MODEL RESPONSE"
        self._write_subsection(title)
        self._write(response_text)

    def log_thinking_trace(self, thinking: str) -> None:
        """Log model thinking/reasoning trace."""
        if not self.enabled:
            return

        self._write_subsection("THINKING TRACE")
        self._write(thinking)

    def log_raw_event(self, event: Any, context: str = "") -> None:
        """Log a raw streaming event for debugging unknown event types.

        Use this when encountering event types that need investigation.
        """
        if not self.enabled:
            return

        self._write_subsection(f"RAW EVENT: {type(event).__name__}")
        if context:
            self._write(f"Context: {context}")
        # Log event attributes
        event_dict = {}
        for attr in ['event_kind', 'delta', 'part', 'result']:
            if hasattr(event, attr):
                val = getattr(event, attr)
                event_dict[attr] = f"{type(val).__name__}: {str(val)[:500]}"
        if event_dict:
            self._write(f"Attributes:\n{json.dumps(event_dict, indent=2)}")
        else:
            self._write(f"Full event: {str(event)[:1000]}")

    def log_message_history(self, messages: list[ModelMessage]) -> None:
        """Log complete message history from pydantic-ai.

        This captures ALL messages including:
        - System prompts
        - User prompts
        - Tool calls and results
        - Model responses
        - Any thinking traces
        """
        if not self.enabled:
            return

        self._write_section("COMPLETE MESSAGE HISTORY")

        for i, msg in enumerate(messages):
            self._write(f"\n--- Message {i + 1} ({type(msg).__name__}) ---")

            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        self._write(f"[SystemPrompt]\n{part.content}")
                    elif isinstance(part, UserPromptPart):
                        self._write(f"[UserPrompt]\n{part.content}")
                    elif isinstance(part, ToolReturnPart):
                        self._write(f"[ToolReturn] {part.tool_name} (id={part.tool_call_id})")
                        content_str = str(part.content)
                        self._write(content_str)
                    else:
                        self._write(f"[{type(part).__name__}]\n{part}")

            elif isinstance(msg, ModelResponse):
                for resp_part in msg.parts:
                    if isinstance(resp_part, TextPart):
                        self._write(f"[TextPart]\n{resp_part.content}")
                    elif isinstance(resp_part, ThinkingPart):
                        # Explicitly handle ThinkingPart for model reasoning traces
                        thinking_content = getattr(resp_part, 'content', str(resp_part))
                        self._write(f"[ThinkingPart]\n{thinking_content}")
                    elif isinstance(resp_part, ToolCallPart):
                        args_str = json.dumps(resp_part.args, indent=2, default=str) if isinstance(resp_part.args, dict) else str(resp_part.args)
                        self._write(f"[ToolCall] {resp_part.tool_name} (id={resp_part.tool_call_id})")
                        self._write(f"Args: {args_str}")
                    else:
                        # Capture any other parts
                        self._write(f"[{type(resp_part).__name__}]\n{resp_part}")
            else:
                self._write(str(msg))

    def log_error(self, error: BaseException, context: str = "") -> None:
        """Log an error with context."""
        if not self.enabled:
            return

        self._write_subsection(f"ERROR: {type(error).__name__}")
        if context:
            self._write(f"Context: {context}")
        self._write(f"Message: {error}")

        # Include traceback for debugging
        import traceback
        self._write(f"Traceback:\n{traceback.format_exc()}")

    def log_max_steps_reached(self, total_steps: int, max_steps: int) -> None:
        """Log when max steps limit is reached."""
        if not self.enabled:
            return

        self._write_subsection("MAX STEPS REACHED")
        self._write(f"Total tool calls: {total_steps}")
        self._write(f"Max allowed: {max_steps}")
        self._write("Response may be incomplete.")

    def close(self) -> None:
        """Close the debug session and write summary.

        Safe to call multiple times - subsequent calls are no-ops.
        """
        if not self.enabled or not self._logger:
            return

        # Write session summary
        self._write_section("DEBUG SESSION END")
        if self.session_start:
            duration = datetime.now() - self.session_start
            self._write(f"Duration: {duration}")
        self._write(f"Total Queries: {self.query_count}")
        self._write(f"Total Tool Calls: {self.tool_call_count}")
        self._write(f"Log saved to: {self.log_path}")

        # Close and remove handler
        if self._handler:
            try:
                self._handler.close()
            except Exception as e:
                logging.debug(f"Error closing handler during cleanup: {e}")
            if self._logger:
                self._logger.removeHandler(self._handler)
            self._handler = None

        # Remove logger from registry to prevent memory leak
        if self._logger_name and self._logger_name in logging.Logger.manager.loggerDict:
            try:
                del logging.Logger.manager.loggerDict[self._logger_name]
            except KeyError:
                logging.debug(f"Logger {self._logger_name} already removed from registry")
        self._logger = None
        self._logger_name = None

    def __enter__(self) -> "DebugLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.log_error(exc_val, "Session terminated due to exception")
        self.close()


def create_debug_logger(root_directory: str, debug: bool = False) -> DebugLogger:
    """Factory function to create a debug logger.

    Args:
        root_directory: Project root directory
        debug: Whether debug mode is enabled

    Returns:
        DebugLogger instance (may be disabled if debug=False)
    """
    return DebugLogger(root_directory, enabled=debug)
