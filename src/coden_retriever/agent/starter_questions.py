"""Starter questions for new developers exploring a codebase.

Provides pre-defined questions that help developers quickly understand
a new project. Users can select these with Tab or type their own questions.
Questions are stored in ~/.coden-retriever/settings.json under
agent.starter_questions and can be customized by editing that file.
"""

from ..config_loader import get_config


def get_starter_questions() -> list[str]:
    """Get the list of starter questions from config.

    Returns:
        List of question strings. Defaults are provided by
        DEFAULT_STARTER_QUESTIONS in constants.py; users can
        override via settings.json.
    """
    return get_config().agent.starter_questions
