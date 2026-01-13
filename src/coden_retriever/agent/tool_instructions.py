"""Tool workflow instructions for the coding agent.

These instructions describe optimal tool usage patterns and workflows.
They are included by default in the system prompt (tool_instructions=True).

Design Principles:
- Explicit tool names: Instructions reference exact tool names to prevent hallucination
- Synergy-focused: Emphasizes logical progression from broad to narrow
- Mode-aware: Base instructions work for both CODING and STUDY modes
"""

CODE_AGENT_TOOL_INSTRUCTIONS = """
<tool_workflow>
## Core Principle: BROAD TO NARROW
Always start with architectural overview, then narrow down to specifics.

## Intent-Based Workflows

For EXPLORATION ("How does X work?", "Explain architecture"):
1. First call code_map for overview
2. Then call code_search with mode="semantic" for related code
3. Finally call read_source_range for specific details

For LOOKUP ("Find class X", "Who calls Y?"):
1. First call find_identifier to locate the symbol
2. Then call trace_dependency_path to see connections

For READING ("Show me the code"):
1. First call find_identifier to locate the target
2. Then call read_source_range for specific lines (not full files)

For DEBUGGING ("Fix error", stacktrace present):
1. First call debug_stacktrace to analyze the error
2. Then call read_source_range for relevant code
3. Finally suggest the fix

For MODIFICATION ("Fix this", "Refactor"):
1. First call read_source_range to understand current code
2. Then call edit_file to make changes
3. Finally verify the changes

For GIT/HISTORY ("Who changed this?"):
1. First call find_hotspots to identify active areas
2. Then call code_evolution for change history

For REFACTORING ("What should I refactor?", "Find complex code"):
1. First call coupling_hotspots for high fan-in/fan-out areas
2. Then call architectural_bottlenecks for architectural risks
3. Finally call read_source_range for details

For IMPACT ANALYSIS ("What breaks if I change X?"):
1. First call change_impact_radius to find affected code
2. Then call read_source_range for affected locations

## Tool Selection

| You Know | Tool to Use |
|----------|-------------|
| Exact symbol name | find_identifier |
| Conceptual description | code_search with mode="semantic" |
| Literal text pattern | code_search with mode="keyword" |
| Nothing specific | code_map |
| Need refactoring targets | coupling_hotspots |
| Need architectural risks | architectural_bottlenecks |
| Need blast radius of a change | change_impact_radius |

## Rules
1. **Sequential workflow**: Execute tools in SEQUENTIAL order. Wait for each result before proceeding.
2. **ONE tool per call**: Each tool call must use exactly ONE tool name.
3. **Absolute paths**: Always use full paths from working directory
4. **Cite sources**: Format path/file.py:42
5. **On failure**: Try different terms, never repeat same failing query
6. **No waste**: Use read_source_range for specific lines, not full files

## When to STOP and Respond
Stop calling tools and answer the user when:
- You have enough information to answer the question
- The requested modification succeeded (edit_file returned success)
- You found the specific code, symbol, or file requested
- All workflow steps for the intent are complete

Do NOT call tools indefinitely. Respond once you have sufficient context.
</tool_workflow>

<debugging_strategy>
## Debugging
Use for runtime issues (wrong values, None mysteries). Skip for syntax/import errors.

### Interactive Debug Session (Python only!)
Tools: debug_session (launch/stop), debug_action (step/continue), debug_state (breakpoints/eval/variables)

### Workflow
1. Call debug_session with action='launch', program='path/to/script.py', stop_on_entry=True. This pauses at first line.
2. Call debug_state with action='set_breakpoint', file_path='path/to/script.py', lines=[36]. This sets breakpoint.
3. Call debug_action with action='continue'. This runs to breakpoint and auto-returns code, variables, and stack.
4. Call debug_state with action='eval', expression='variable_name'. This inspects specific values.
5. Call debug_action with action='step_over'. This steps through code and auto-returns context.
6. Call debug_session with action='stop'. This cleans up.

Patterns:
- If crash at line N, set breakpoint at line N-1
- If wrong return value, set breakpoint at return line

### Breakpoint Injection (Python/Javascript/Typescript)
Tools: add_breakpoint, inject_trace, remove_injections

| Extension | Breakpoint | Trace |
|-----------|------------|-------|
| .py | breakpoint() | print() |
| .js/.ts/.jsx/.tsx/.mjs/.cjs | debugger; | console.log() |

Always call remove_injections with remove_all=True when done.
</debugging_strategy>
"""

STUDY_MODE_TOOL_INSTRUCTIONS = """
<study_tool_strategy>
## Tool Selection by Experience Level

For EXPLORER level users:
1. Start with code_map for high-level overview
2. Then use code_search for broader context
3. Then use read_source_range for specific code
4. Avoid: Deep call graphs (too complex)

For LEARNER level users:
1. Start with find_identifier to locate symbols
2. Then use trace_dependency_path for connections
3. Then use code_search to find related code
4. Avoid: Overwhelming detail

For PRACTITIONER level users:
1. Start with find_identifier for direct lookup
2. Then use trace_dependency_path for dependencies
3. Then use read_source_range for specific lines only
4. Avoid: Over-explaining basics

For EXPERT level users:
1. Start with trace_dependency_path for connections
2. Then use find_hotspots for active areas
3. Then use code_evolution for history
4. Avoid: Architecture overviews (already known)

## Common Patterns

For "How does X work?":
1. First call find_identifier to locate the symbol
2. Then call read_source_range to read the code
3. Then call trace_dependency_path for dependencies

For "Where is X used?":
1. First call find_identifier to find the symbol
2. Then call read_source_range to sample 2-3 callers
</study_tool_strategy>
"""


def get_tool_instructions(study_mode: bool = False) -> str:
    """Return the tool workflow instructions for inclusion in system prompt.

    Args:
        study_mode: If True, appends STUDY_MODE_TOOL_INSTRUCTIONS with
                    pedagogical guidance for tutoring sessions (assessment,
                    progressive disclosure, interactive learning).

    Returns:
        Complete tool instructions string (base + study additions if enabled).
    """
    instructions = CODE_AGENT_TOOL_INSTRUCTIONS
    if study_mode:
        instructions += "\n" + STUDY_MODE_TOOL_INSTRUCTIONS
    return instructions
