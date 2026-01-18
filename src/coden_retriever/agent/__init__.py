"""Pydantic-AI coding agent module with ReAct reasoning.

Requires the 'agent' extra:
    pip install 'coden-retriever[agent]'
"""

from ..utils.optional_deps import MissingDependencyError

# Try to import agent components - raise helpful error if pydantic-ai not installed
try:
    from .coding_agent import CodingAgent, run_interactive
    from .models import (
        Action,
        AgentResponse,
        Observation,
        ReActStep,
        Thought,
    )
    from .react_loop import print_steps, run_with_react_display

    from .rich_console import (
        console,
        get_user_input,
        print_agent_response,
        print_error,
        print_fatal_error,
        print_goodbye,
        print_steps_rich,
        print_warning,
        print_welcome,
    )
except ImportError as e:
    # Check if pydantic-ai is the missing dependency
    if "pydantic_ai" in str(e) or "pydantic-ai" in str(e):
        raise MissingDependencyError("agent") from e
    raise

__all__ = [
    # Main agent
    "CodingAgent",
    "run_interactive",
    # Models
    "AgentResponse",
    "ReActStep",
    "Thought",
    "Action",
    "Observation",
    # Utilities
    "run_with_react_display",
    "print_steps",
    # Rich console
    "console",
    "get_user_input",
    "print_agent_response",
    "print_error",
    "print_fatal_error",
    "print_goodbye",
    "print_steps_rich",
    "print_warning",
    "print_welcome",
]
