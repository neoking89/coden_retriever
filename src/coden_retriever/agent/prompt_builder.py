"""Prompt builder for constructing system prompts.

Provides templates and helper functions for generating system prompts.
Supports both CODING and STUDY mode prompts with directory tree caching.
"""

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..config_loader import AppConfig, get_config, resolve_template
from ..constants import (
    DEFAULT_STUDY_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
)
from ..formatters import generate_shallow_tree
from .tool_instructions import get_tool_instructions

from .coden_models import AgentMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .coden_models import AgentDeps


def _build_coding_prompt(config: AppConfig, abs_root: str, directory_tree: str) -> str:
    """Build the CODING mode base prompt from config template.

    Falls back to DEFAULT_SYSTEM_PROMPT_TEMPLATE if the template fails to load
    (e.g., file: path not found, permission error, missing placeholders).
    """
    try:
        template = resolve_template(config.agent.system_prompt_template)
        return template.format(root_directory=abs_root, directory_tree=directory_tree)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("system_prompt_template failed (%s), using default", exc)

    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(root_directory=abs_root, directory_tree=directory_tree)


def _build_study_prompt(
    config: AppConfig,
    abs_root: str,
    study_topic: Optional[str],
    directory_tree: str,
) -> str:
    """Build the STUDY mode base prompt from config template.

    Falls back to DEFAULT_STUDY_PROMPT_TEMPLATE if the template fails to load
    (e.g., file: path not found, permission error, missing placeholders).
    """
    topic_text = study_topic if study_topic else "General codebase exploration"
    try:
        template = resolve_template(config.agent.study_prompt_template)
        return template.format(
            root_directory=abs_root, study_topic=topic_text, directory_tree=directory_tree,
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("study_prompt_template failed (%s), using default", exc)

    return DEFAULT_STUDY_PROMPT_TEMPLATE.format(
        root_directory=abs_root, study_topic=topic_text, directory_tree=directory_tree,
    )


class PromptBuilder:
    """Builder for constructing system prompts with caching support.

    Caches directory trees to avoid regenerating them on every prompt build.

    When use_config_for_tool_instructions is True, tool_instructions setting
    is read from config cache for immediate updates via /config set.
    When False, uses the include_tool_instructions constructor parameter.
    """

    def __init__(
        self,
        include_tool_instructions: bool = False,
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
        config = get_config()

        if study_mode:
            system_prompt = _build_study_prompt(config, abs_root, study_topic, directory_tree)
        else:
            system_prompt = _build_coding_prompt(config, abs_root, directory_tree)

        if self._should_include_tools(config):
            system_prompt += "\n" + get_tool_instructions(study_mode=study_mode)

        return system_prompt

    def _should_include_tools(self, config: AppConfig) -> bool:
        """Determine whether tool instructions should be included."""
        if self.use_config_for_tool_instructions:
            return config.agent.tool_instructions
        return self.include_tool_instructions


def build_system_prompt(
    root_directory: str,
    include_tool_instructions: bool = False,
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
    config = get_config()

    if deps.mode == AgentMode.STUDY:
        return _build_study_prompt(config, deps.root_directory, deps.study_topic, tree)

    return _build_coding_prompt(config, deps.root_directory, tree)


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
