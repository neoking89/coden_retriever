"""Coden-specific agent models.

These types depend on coden-only concepts (study mode, project-rooted deps,
session-start trigger words) and intentionally live outside `models.py` so
the library-target file has no coden-domain leakage.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AgentMode(Enum):
    """Agent operating modes for dependency injection."""

    CODING = "coding"
    STUDY = "study"


class SessionTrigger(str, Enum):
    """Special input values that trigger study session start."""

    EMPTY = ""
    START = "start"
    BEGIN = "begin"


@dataclass
class AgentDeps:
    """Dependencies injected into agent tools and instructions via RunContext.

    Used with pydantic-ai's `deps_type` parameter to give tools and
    instruction generators context about the current project root, mode,
    and study topic.
    """

    root_directory: str
    mode: AgentMode = AgentMode.CODING
    include_tool_instructions: bool = False
    debug: bool = False
    study_topic: Optional[str] = None
