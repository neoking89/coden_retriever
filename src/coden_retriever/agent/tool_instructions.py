"""Tool workflow instructions for the coding agent.

The actual instruction text lives in config (`agent.tool_instructions_template`
and `agent.study_tool_instructions_template`). Defaults sit in
`coden_retriever.constants` so users can override the strings — including
via the `file:` prefix — through `~/.coden-retriever/settings.json`.
"""
from ..config_loader import get_config, resolve_or_default
from ..constants import (
    DEFAULT_STUDY_TOOL_INSTRUCTIONS_TEMPLATE,
    DEFAULT_TOOL_INSTRUCTIONS_TEMPLATE,
)


def get_tool_instructions(study_mode: bool = False) -> str:
    """Return the tool workflow instructions for inclusion in system prompt.

    Args:
        study_mode: If True, appends the study-mode instructions with
                    pedagogical guidance for tutoring sessions.

    Returns:
        Complete tool instructions string (base + study additions if enabled).
    """
    agent_cfg = get_config().agent
    instructions = resolve_or_default(
        agent_cfg.tool_instructions_template,
        DEFAULT_TOOL_INSTRUCTIONS_TEMPLATE,
        "tool_instructions_template",
    )
    if study_mode:
        instructions += "\n" + resolve_or_default(
            agent_cfg.study_tool_instructions_template,
            DEFAULT_STUDY_TOOL_INSTRUCTIONS_TEMPLATE,
            "study_tool_instructions_template",
        )
    return instructions
