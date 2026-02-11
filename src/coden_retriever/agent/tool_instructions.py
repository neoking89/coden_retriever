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
1. code_map → code_search (mode="semantic") → read_source_range

For LOOKUP ("Find class X", "Who calls Y?"):
1. find_identifier → trace_dependency_path

For READING ("Show me the code"):
1. find_identifier → read_source_range (or read_source_ranges for multiple locations)

For DEBUGGING ("Fix error", stacktrace present):
1. debug_stacktrace → read_source_range → suggest fix

For FILE MODIFICATION ("Fix this", "Create file"):
1. read_source_range → edit_file (or write_file for new files)
2. On mistake: undo_file_change
3. To remove: delete_file

For GIT/HISTORY ("Who changed this?", "Why was this written?"):
1. find_hotspots (churn analysis) → code_evolution (function history)
2. git_history_context for line-level blame + commit messages

For REFACTORING ("What should I refactor?"):
1. coupling_hotspots → architectural_bottlenecks → read_source_range

For IMPACT ANALYSIS ("What breaks if I change X?"):
1. change_impact_radius → read_source_range

For CODE QUALITY ("Find issues", "Clean up code"):
1. detect_clones (mode: combined|semantic|syntactic)
2. detect_dead_code (unused functions)
3. detect_echo_comments (redundant comments)
4. propagation_cost (architecture health)
5. detect_tramp_data (parameters repeated across functions)
6. detect_sensitive_values (hardcoded secrets, API keys, credentials)

For THIRD-PARTY CODE ("How does library X work?"):
1. check_python_virtual_env → get_python_package_path → code_search

For CODE FLAGGING ("Mark issues in source"):
1. flag_code_tool (insert [CODEN] markers) → flag_clear_tool (remove markers)

## Tool Selection

| You Need | Tool |
|----------|------|
| Exact symbol name | find_identifier |
| Conceptual search | code_search mode="semantic" |
| Literal text match | code_search mode="keyword" |
| Overview | code_map |
| Call paths/dependencies | trace_dependency_path |
| Refactoring targets | coupling_hotspots |
| Architectural risks | architectural_bottlenecks |
| Blast radius | change_impact_radius |
| Architecture health | propagation_cost |
| Duplicate code | detect_clones (combined/semantic/syntactic) |
| Unused code | detect_dead_code |
| Tramp data / coupling | detect_tramp_data |
| Hardcoded secrets/sensitive values | detect_sensitive_values |
| Useless comments | detect_echo_comments |
| Parse error stacktrace | debug_stacktrace |
| Line-level blame | git_history_context |
| Churn analysis | find_hotspots |
| Function history | code_evolution |
| Virtual env check | check_python_virtual_env |
| Library source path | get_python_package_path |
| Create new file | write_file |
| Edit existing file | edit_file |
| Remove file | delete_file |
| Undo file change | undo_file_change |
| Flag code issues | flag_code_tool |
| Clear [CODEN] markers | flag_clear_tool |
| Read specific lines | read_source_range |
| Read multiple ranges | read_source_ranges |

## Rules
1. **Sequential workflow**: Wait for each result before proceeding
2. **Absolute paths**: Always use full paths
3. **Cite sources**: Format path/file.py:42
4. **On failure**: Try different terms, never repeat same failing query
5. **Efficiency**: Use read_source_range for specific lines, read_source_ranges for multiple locations

## When to STOP
Stop calling tools when:
- You have enough information to answer
- Modification succeeded (edit_file/write_file returned success)
- Found the requested code/symbol/file
</tool_workflow>

<debugging_strategy>
## Debugging
Use for runtime issues (wrong values, None mysteries). Skip for syntax/import errors.

### Interactive Debug Session (Python only)
Tools: debug_session, debug_action, debug_state

Workflow:
1. debug_session action='launch', program='script.py', stop_on_entry=True
2. debug_state action='set_breakpoint', file_path='script.py', lines=[36]
3. debug_action action='continue' → auto-returns code, variables, stack
4. debug_state action='eval', expression='variable_name'
5. debug_action action='step_over' → auto-returns context
6. debug_session action='stop'

For IDE attachment: debug_server to start a server, then attach from IDE.

### Breakpoint Injection (Python/JS/TS)
Tools: add_breakpoint, inject_trace, list_injections, remove_injections

| Extension | Breakpoint | Trace |
|-----------|------------|-------|
| .py | breakpoint() | print() |
| .js/.jsx/.mjs/.cjs | debugger; | console.log() |
| .ts/.tsx | debugger; | console.log() |

Use list_injections to see active injections. Always remove_injections when done.

### Stacktrace Analysis
Tool: debug_stacktrace - Parse any language stacktrace and map to local code.
Workflow: Paste stacktrace -> get user frames with source context -> read_source_range for details
</debugging_strategy>
"""

STUDY_MODE_TOOL_INSTRUCTIONS = """
<study_tool_strategy>
## Tool Selection by Experience Level

For EXPLORER level:
1. code_map → code_search → read_source_range
2. Avoid: Deep call graphs, analysis tools

For LEARNER level:
1. find_identifier → trace_dependency_path → code_search
2. Avoid: Overwhelming detail

For PRACTITIONER level:
1. find_identifier → trace_dependency_path → read_source_range
2. Code quality: detect_clones (mode=semantic), detect_dead_code

For EXPERT level:
1. trace_dependency_path → find_hotspots → code_evolution
2. Architecture: coupling_hotspots, architectural_bottlenecks, propagation_cost
3. Avoid: Architecture overviews

## Common Patterns

For "How does X work?":
find_identifier → read_source_range → trace_dependency_path

For "Where is X used?":
find_identifier → read_source_ranges (sample 2-3 callers)

For "What should I refactor?":
coupling_hotspots → detect_clones (mode=combined) → read_source_range
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
