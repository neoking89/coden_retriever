"""Pydantic models for ReAct agent framework.

Defines structured output types for reasoning steps, actions, and observations.
Also includes dependency injection types for pydantic-ai agent integration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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

    This dataclass is used with pydantic-ai's deps_type parameter to provide
    context-aware information to tools and instruction generators.

    Attributes:
        root_directory: Absolute path to the project root directory.
        mode: Current agent mode (CODING for code analysis, STUDY for tutoring).
        include_tool_instructions: Whether to include detailed tool workflow guidance.
        debug: Whether debug logging is enabled.
        study_topic: Optional focus topic for STUDY mode sessions.
    """
    root_directory: str
    mode: AgentMode = AgentMode.CODING
    include_tool_instructions: bool = False
    debug: bool = False
    study_topic: Optional[str] = None


class Thought(BaseModel):
    """Agent's reasoning step."""

    reasoning: str = Field(description="Current analysis of the situation")
    next_action: str = Field(description="What tool to call and why")


class Action(BaseModel):
    """Tool call decision."""

    tool_name: str = Field(description="Name of the tool to call")
    tool_input: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class Observation(BaseModel):
    """Result from tool execution."""

    tool_name: str = Field(description="Name of the tool that was called")
    result: Any = Field(default=None, description="Result from the tool")
    success: bool = Field(description="Whether the tool call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ReActStep(BaseModel):
    """Single ReAct iteration (Thought -> Action -> Observation)."""

    step_number: int = Field(description="Step number in the reasoning chain")
    thought: Optional[Thought] = Field(
        default=None, description="Agent's reasoning for this step"
    )
    action: Optional[Action] = Field(
        default=None, description="Tool call made in this step"
    )
    observation: Optional[Observation] = Field(
        default=None, description="Result from tool execution"
    )


class AgentResponse(BaseModel):
    """Final structured response from the agent."""

    answer: str = Field(description="Final answer to the user's query")
    steps: list[ReActStep] = Field(
        default_factory=list, description="ReAct steps taken to reach the answer"
    )
    total_tool_calls: int = Field(
        default=0, description="Total number of tool calls made"
    )
    reached_max_steps: bool = Field(
        default=False, description="Whether max steps limit was reached"
    )
    messages: list[Any] = Field(
        default_factory=list, description="Message history for multi-turn conversations"
    )


class FallbackIterationResult(BaseModel):
    """Result of checking and handling text-based tool call fallback."""

    should_continue: bool = Field(
        default=False, description="Whether to continue the fallback loop"
    )
    updated_history: list[Any] = Field(
        default_factory=list, description="Updated message history for next iteration"
    )
    continuation_prompt: str = Field(
        default="", description="Prompt for next iteration"
    )
    steps: list[ReActStep] = Field(
        default_factory=list, description="Steps from fallback execution"
    )
    tool_call_count: int = Field(
        default=0, description="Number of tool calls executed in fallback"
    )
