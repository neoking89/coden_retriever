"""Prompt builder for constructing system prompts.

Provides templates and helper functions for generating system prompts.
Supports both CODING and STUDY mode prompts with directory tree caching.

The PromptBuilder class is kept for backwards compatibility and caching,
while the templates and generate_*_instructions() functions are exported
for use with pydantic-ai's @agent.instructions decorator pattern.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..config_loader import get_config
from ..formatters import generate_shallow_tree
from .tool_instructions import get_tool_instructions

from .models import AgentMode

if TYPE_CHECKING:
    from .models import AgentDeps


STUDY_MODE_PROMPT = """
<role>
You are an Interactive Coding Tutor. Teach through discovery, not lectures.
Guide users to understand code by asking focused questions with code citations.
</role>

<environment>
Working Directory: {root_directory}
Topic: {study_topic}
Use absolute paths for all tool calls.
</environment>

<experience_levels>
| Level | Signals | Approach |
|-------|---------|----------|
| EXPLORER | "first time", "new to this" | Define terms, analogies, slow pace, architecture first |
| LEARNER | "looked around", "still learning" | Connect concepts, medium pace, reinforce connections |
| PRACTITIONER | "know the basics", "want depth" | Jump to specifics, trade-offs, "why" questions |
| EXPERT | "I maintain this", "contributor" | Precise lookups, skip basics, edge cases |
</experience_levels>

<teaching_flow>
**Session start:** Greet warmly (one sentence), ask their experience level and goal. NO tool calls.

**Responding to answers:**
- Correct: Brief acknowledgment, go deeper, slightly harder follow-up
- Partial: Acknowledge correct part, guide to missing piece
- Stuck: No judgment, hint or simplify, easier follow-up

**Question types** (rotate based on level):
- RECALL: "What does X do?" (EXPLORER/LEARNER)
- COMPREHENSION: "How do X and Y connect?" (LEARNER/PRACTITIONER)
- APPLICATION: "Which component would you modify for [goal]?" (PRACTITIONER)
- ANALYSIS: "Why this design?" (PRACTITIONER/EXPERT)
- PREDICTION: "What if X changed?" (EXPERT)

**Session management:**
- Track: level, goal, topics covered, answer patterns
- Summarize progress when natural (not forced intervals)
- On exit: brief recap, key files, next steps
</teaching_flow>

<constraints>
1. **No hallucination**: If a tool fails, try different terms - never repeat failures.
2. **Cite sources**: Only cite `file:line` if verified by a tool call this turn.
3. **Memory**: Track what you've explored - don't re-fetch.
4. **Concise**: ~100-150 words per response. End with ONE question.
5. **Be Thorough**: Keep making tool calls until every claim has evidence and the user's request is fully satisfied.
</constraints>

<directory_structure>
{directory_tree}
</directory_structure>
"""

SYSTEM_PROMPT_TEMPLATE = """
<role>
You are an Expert Code Analysis Agent. You are precise, methodical, and never hallucinate.
Your purpose is to help users understand, explore, and debug codebases using the available tools.
When you don't know something, you use tools to find out - you never invent code or file contents.
</role>

<environment>
Current Working Directory: {root_directory}
All tool calls MUST use absolute paths constructed from this directory.
Example: If a file is at "src/main.py", the absolute path is "{root_directory}/src/main.py"
</environment>

<constraints>
1. **Parallel Execution**: If you need to read 3 files or search 2 terms, call all tools in a SINGLE turn. Do not do them sequentially unless the second depends on the result of the first.
2. **Absolute Paths**: Always combine the Current Working Directory with relative paths to form absolute paths for tool calls. Never use relative paths directly.
3. **No Hallucination**: If a tool returns "not found" or an error, stop and search with different terms. Do not invent code, file contents, or paths.
4. **Token Budget**: Prefer reading specific line ranges over full files. Request only the lines you need.
5. **Cite Sources**: Always include file paths and line numbers when referencing code (format: `path/to/file.py:42`).
6. **No Preamble**: Skip apologies, self-references, and filler. Start directly with findings or actions.
7. **Secrets**: Never output `[REDACTED]` tokens, API keys, passwords, or credentials. If found, note their presence but do not display values.
8. **Never Assume**: Always verify your findings with tool calls. NEVER assume or guess code structure or content.
9. **Be Thorough**: Keep making tool calls until every claim has evidence AND the user's request is FULLY satisfied.
</constraints>

<reasoning_process>
Follow the ReAct loop for every query:
1. THOUGHT: Analyze what the user wants. Identify their intent category (exploration, lookup, reading, debugging, git analysis).
2. PLAN: Select the appropriate tool chain based on intent. List tools you will call.
3. ACTION: Execute tool calls. Batch independent calls together in parallel.
4. OBSERVATION: Process results. If incomplete, return to THOUGHT.
5. ANSWER: Synthesize findings into a clear, structured response with code citations.
</reasoning_process>

<directory_structure>
{directory_tree}
</directory_structure>
"""


class PromptBuilder:
    """Builder for constructing system prompts with caching support.

    Caches directory trees to avoid regenerating them on every prompt build.

    When use_config_for_tool_instructions is True, tool_instructions setting
    is read from config cache for immediate updates via /config set.
    When False, uses the include_tool_instructions constructor parameter.
    """

    def __init__(
        self,
        include_tool_instructions: bool = True,
        use_config_for_tool_instructions: bool = False,
    ):
        self.include_tool_instructions = include_tool_instructions
        self.use_config_for_tool_instructions = use_config_for_tool_instructions
        self._cached_tree: str | None = None
        self._cached_tree_path: str | None = None

    def get_directory_tree(self, root_directory: str, refresh: bool = False) -> str:
        """Get cached directory tree, regenerating only if path changed or refresh requested."""
        abs_root = str(Path(root_directory).resolve())
        if not refresh and self._cached_tree is not None and self._cached_tree_path == abs_root:
            return self._cached_tree

        self._cached_tree = generate_shallow_tree(Path(abs_root))
        self._cached_tree_path = abs_root
        return self._cached_tree

    def build(
        self,
        root_directory: str,
        study_mode: bool = False,
        study_topic: Optional[str] = None,
        refresh_tree: bool = False,
    ) -> str:
        """Build the complete system prompt.

        Args:
            root_directory: Path to the project root (will be resolved to absolute).
            study_mode: If True, use the study/tutor prompt instead of normal prompt.
            study_topic: Optional topic to focus the study session on.
            refresh_tree: If True, regenerate the directory tree even if cached.

        Returns:
            Complete system prompt with directory structure and tool instructions.
        """
        abs_root = str(Path(root_directory).resolve())
        directory_tree = self.get_directory_tree(root_directory, refresh=refresh_tree)

        if study_mode:
            topic_text = study_topic if study_topic else "General codebase exploration"
            system_prompt = STUDY_MODE_PROMPT.format(
                root_directory=abs_root,
                study_topic=topic_text,
                directory_tree=directory_tree,
            )
        else:
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                root_directory=abs_root,
                directory_tree=directory_tree,
            )

        # Read from config for immediate updates via /config set
        if self.use_config_for_tool_instructions:
            config = get_config()
            include_tools = config.agent.tool_instructions if config else self.include_tool_instructions
        else:
            include_tools = self.include_tool_instructions

        if include_tools:
            system_prompt += "\n" + get_tool_instructions(study_mode=study_mode)

        return system_prompt


def build_system_prompt(
    root_directory: str,
    include_tool_instructions: bool = True,
    study_mode: bool = False,
    study_topic: str | None = None,
) -> str:
    """Build system prompt with directory tree context.

    This is a convenience function that creates a one-off PromptBuilder.
    For repeated calls, use PromptBuilder directly to benefit from caching.

    Args:
        root_directory: Path to the project root (will be resolved to absolute).
        include_tool_instructions: If True, append detailed tool workflow instructions.
        study_mode: If True, use the study/tutor prompt instead of normal prompt.
        study_topic: Optional topic to focus the study session on.

    Returns:
        Complete system prompt with directory structure and absolute path.
    """
    builder = PromptBuilder(include_tool_instructions=include_tool_instructions)
    return builder.build(
        root_directory=root_directory,
        study_mode=study_mode,
        study_topic=study_topic,
    )


# Thread-safe cache for directory trees
_tree_cache: dict[str, str] = {}
_tree_cache_lock = threading.Lock()


def generate_directory_tree(root_directory: str, refresh: bool = False) -> str:
    """Generate a directory tree for the given root directory.

    Uses a thread-safe cache to avoid regenerating for the same directory.

    Args:
        root_directory: Path to the project root.
        refresh: If True, regenerate even if cached.

    Returns:
        Directory tree string.
    """
    abs_root = str(Path(root_directory).resolve())

    with _tree_cache_lock:
        if not refresh and abs_root in _tree_cache:
            return _tree_cache[abs_root]

        tree = generate_shallow_tree(Path(abs_root))
        _tree_cache[abs_root] = tree
        return tree


def generate_system_instructions(deps: "AgentDeps") -> str:
    """Generate mode-specific system prompt from AgentDeps.

    This function is designed to be used with pydantic-ai's @agent.instructions
    decorator or called directly to generate the system prompt.

    Args:
        deps: AgentDeps with mode and configuration.

    Returns:
        Complete system prompt string.
    """
    tree = generate_directory_tree(deps.root_directory)

    if deps.mode == AgentMode.STUDY:
        return STUDY_MODE_PROMPT.format(
            root_directory=deps.root_directory,
            study_topic=deps.study_topic or "General codebase exploration",
            directory_tree=tree,
        )
    else:
        return SYSTEM_PROMPT_TEMPLATE.format(
            root_directory=deps.root_directory,
            directory_tree=tree,
        )


def generate_tool_instructions_from_deps(deps: "AgentDeps") -> str:
    """Generate tool workflow instructions from AgentDeps.

    This function is designed to be used with pydantic-ai's @agent.instructions
    decorator to conditionally include tool instructions.

    Args:
        deps: AgentDeps with configuration.

    Returns:
        Tool instructions string, or empty string if disabled.
    """
    if not deps.include_tool_instructions:
        return ""

    study_mode = deps.mode == AgentMode.STUDY
    return get_tool_instructions(study_mode=study_mode)
