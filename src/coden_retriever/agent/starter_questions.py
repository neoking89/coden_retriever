"""Starter questions for new developers exploring a codebase.

Provides pre-defined questions that help developers quickly understand
a new project. Users can select these with Tab or type their own questions.

Questions are based on common onboarding challenges:
- https://mannes.tech/10-questions-codebase/
- https://www.cortex.io/post/developer-onboarding-guide
- https://trstringer.com/20-questions-for-new-software-team/
"""

from dataclasses import dataclass


@dataclass
class StarterQuestion:
    """A starter question with short label and full question text."""

    label: str  # Short label shown in completion menu
    question: str  # Full question text
    category: str  # Category for grouping


# Common questions new developers ask, organized by category
STARTER_QUESTIONS = [
    # Project Overview
    StarterQuestion(
        label="Getting started",
        question="How do I get started with this project? Walk me through setup, installation, and first steps.",
        category="overview",
    ),
    StarterQuestion(
        label="What does this project do?",
        question="What does this project do? Give me a high-level overview of its purpose and main features.",
        category="overview",
    ),
    StarterQuestion(
        label="Project structure",
        question="Explain the project structure. What are the main directories and what do they contain?",
        category="overview",
    ),
    StarterQuestion(
        label="Entry points",
        question="Where are the main entry points of this application? How does it start?",
        category="overview",
    ),
    # Architecture
    StarterQuestion(
        label="Architecture patterns",
        question="What architectural patterns does this codebase follow? (MVC, Clean Architecture, etc.)",
        category="architecture",
    ),
    StarterQuestion(
        label="Key components",
        question="What are the key components/modules and how do they interact?",
        category="architecture",
    ),
    StarterQuestion(
        label="Data flow",
        question="How does data flow through the application? Trace a typical request/operation.",
        category="architecture",
    ),
    # Technologies
    StarterQuestion(
        label="Tech stack",
        question="What technologies and frameworks are used? Check package.json, requirements.txt, or similar.",
        category="tech",
    ),
    StarterQuestion(
        label="Dependencies",
        question="What are the main dependencies and what are they used for?",
        category="tech",
    ),
    # Code Navigation
    StarterQuestion(
        label="Find function",
        question="How do I find where a specific function or class is defined?",
        category="navigation",
    ),
    StarterQuestion(
        label="API endpoints",
        question="Where are the API endpoints defined? List them with their routes.",
        category="navigation",
    ),
    StarterQuestion(
        label="Database models",
        question="Where are the database models/schemas defined? What entities exist?",
        category="navigation",
    ),
    # Development
    StarterQuestion(
        label="How to build",
        question="How do I build and run this project locally?",
        category="development",
    ),
    StarterQuestion(
        label="Run tests",
        question="How do I run the tests? What testing framework is used?",
        category="development",
    ),
    StarterQuestion(
        label="Configuration",
        question="How is the application configured? Where are config files and environment variables?",
        category="development",
    ),
    StarterQuestion(
        label="Refactoring targets",
        question="What functions have high coupling and should be refactored? Find code with many callers AND dependencies.",
        category="refactoring",
    ),
    StarterQuestion(
        label="Architectural bottlenecks",
        question="What are the architectural bottlenecks in this codebase? Find functions that many call paths flow through.",
        category="refactoring",
    ),
    StarterQuestion(
        label="Impact of change",
        question="If I change <insert function here>, what code will be affected? Show me the blast radius.",
        category="refactoring",
    ),
    StarterQuestion(
        label="Code complexity",
        question="Which parts of the codebase are most complex? Where should I focus code review efforts?",
        category="refactoring",
    ),
]


def get_starter_questions() -> list[StarterQuestion]:
    """Get the list of starter questions.

    Returns:
        List of StarterQuestion objects.
    """
    return STARTER_QUESTIONS


def get_question_by_label(label: str) -> str | None:
    """Get the full question text for a label.

    Args:
        label: The short label to search for.

    Returns:
        The full question text, or None if not found.
    """
    for q in STARTER_QUESTIONS:
        if q.label.lower() == label.lower():
            return q.question
    return None


def match_questions(partial: str) -> list[StarterQuestion]:
    """Find questions matching a partial string.

    Args:
        partial: Partial text to match against labels.

    Returns:
        List of matching StarterQuestion objects.
    """
    partial_lower = partial.lower()
    return [q for q in STARTER_QUESTIONS if partial_lower in q.label.lower()]
