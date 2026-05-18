"""Pydantic-AI coding agent module with ReAct reasoning.

Heavy submodules (coden_session, react_loop, rich_console) are exposed
lazily via PEP 562 `__getattr__` so that importing this package does NOT
eagerly load coden-glue modules. Eager-loading `coden_session` re-enters
`coden_retriever.config_loader` mid-init (config_loader pulls constants
from `.agent._constants`), causing an `ImportError: cannot import name
'GenerationSettings' from partially initialized module`.
"""

import importlib
from typing import Any

from .models import (
    Action,
    AgentResponse,
    Observation,
    ReActStep,
    Thought,
)

_LAZY_ATTRS: dict[str, str] = {
    "CodingAgent": ".coding_agent",
    "run_interactive": ".coden_session",
    "print_steps": ".rich_console",
    "console": ".rich_console",
    "get_user_input": ".rich_console",
    "print_agent_response": ".rich_console",
    "print_error": ".rich_console",
    "print_fatal_error": ".rich_console",
    "print_goodbye": ".rich_console",
    "print_steps_rich": ".rich_console",
    "print_warning": ".rich_console",
    "print_welcome": ".rich_console",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "CodingAgent",
    "run_interactive",
    "AgentResponse",
    "ReActStep",
    "Thought",
    "Action",
    "Observation",
    "print_steps",
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
