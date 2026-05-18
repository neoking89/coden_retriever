"""Pydantic models for the ReAct agent.

Structured Thought / Action / Observation / ReActStep types plus their
aggregated `AgentResponse`. Returned by `CodingAgent.run`; consumed by
callers that want to inspect each step of the reasoning chain.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessage


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    messages: list[ModelMessage] = Field(
        default_factory=list, description="Message history for multi-turn conversations"
    )
