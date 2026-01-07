"""Tool workflow instructions for the coding agent.

These instructions describe optimal tool usage patterns and workflows.
They are included by default in the system prompt (tool_instructions=True).

Design Principles:
- Explicit tool names: Instructions reference exact tool names to prevent hallucination
- Synergy-focused: Emphasizes logical progression from broad → narrow
- Mode-aware: Base instructions work for both CODING and STUDY modes
"""

CODE_AGENT_TOOL_INSTRUCTIONS = """
<tool_workflow>
## Core Principle: BROAD → NARROW
Always start with architectural overview, then narrow down to specifics.

## Intent → Strategy

| Intent | Signals | Tool Flow |
|--------|---------|-----------|
| Exploration | "How does X work?", "Explain architecture" | `code_map` → `code_search(mode="semantic")` → `read_source_range` |
| Lookup | "Find class X", "Who calls Y?" | `find_identifier` → `trace_dependency_path` |
| Reading | "Show me the code" | `find_identifier` → `read_source_range` (not full files) |
| Debugging | "Fix error", stacktrace | `debug_stacktrace` → `read_source_range` → Suggest fix |
| Modification | "Fix this", "Refactor" | `read_source_range` → `edit_file` → Verify |
| Git/History | "Who changed this?" | `find_hotspots` → `code_evolution` |
| Refactoring | "What should I refactor?", "Find complex code" | `coupling_hotspots` → `architectural_bottlenecks` → `read_source_range` |
| Impact Analysis | "What breaks if I change X?" | `change_impact_radius` → `read_source_range` |

## Tool Selection

| You Know | Tool |
|----------|------|
| Exact symbol name (e.g. `AuthManager`) | `find_identifier(identifier="AuthManager")` |
| Conceptual description (e.g. "auth logic") | `code_search(query="auth logic", mode="semantic")` |
| Literal text (e.g. "TODO") | `code_search(query="TODO", mode="keyword")` |
| Nothing specific | `code_map` first |
| Need refactoring targets | `coupling_hotspots` (high fan-in x fan-out) |
| Need architectural risks | `architectural_bottlenecks` (high betweenness) |
| Need blast radius of a change | `change_impact_radius(symbol_name="...")` |

## Rules
1. **Batch calls**: Multiple independent tool calls in ONE turn
2. **Absolute paths**: Always use full paths from working directory
3. **Cite sources**: Format `path/file.py:42`
4. **On failure**: Try different terms, never repeat same failing query
5. **No waste**: Use `read_source_range` for specific lines, not full files
</tool_workflow>

<debugging_strategy>
## Debugging
Use for runtime issues (wrong values, None mysteries). Skip for syntax/import errors.

### Interactive Debug Session (Python only!)
Tools: `debug_session` (launch/stop), `debug_action` (step/continue), `debug_state` (breakpoints/eval/variables)

### Workflow
1. `debug_session(action='launch', program='path/to/script.py', stop_on_entry=True)` → pauses at first line
2. `debug_state(action='set_breakpoint', file_path='path/to/script.py', lines=[36])` → set breakpoint
3. `debug_action(action='continue')` → run to breakpoint, auto-returns code + variables + stack
4. `debug_state(action='eval', expression='variable_name')` → inspect specific values
5. `debug_action(action='step_over')` → step through code, auto-returns context
6. `debug_session(action='stop')` → cleanup
Patterns: crash at N → breakpoint at N-1; wrong return → breakpoint at return line

### Breakpoint Injection (Python/Javascript/Typescript)
Tools: `add_breakpoint`, `inject_trace`, `remove_injections`

| Extension | Breakpoint | Trace |
|-----------|------------|-------|
| .py | `breakpoint()` | `print()` |
| .js/.ts/.jsx/.tsx/.mjs/.cjs | `debugger;` | `console.log()` |

Always call `remove_injections(remove_all=True)` when done.
</debugging_strategy>
"""

STUDY_MODE_TOOL_INSTRUCTIONS = """
<study_tool_strategy>
## Tool Selection by Experience Level

| Level | Start With | Then Use | Avoid |
|-------|------------|----------|-------|
| EXPLORER | `code_map` | `code_search(mode="semantic")`, `read_source_range` | Deep call graphs |
| LEARNER | `find_identifier`, `trace_dependency_path` | `code_search` for connections | Overwhelming detail |
| PRACTITIONER | `find_identifier`, `trace_dependency_path` | `read_source_range` (specific lines) | Over-explaining basics |
| EXPERT | `trace_dependency_path`, `find_hotspots` | `code_evolution`, precise line ranges | Architecture overviews |

## Common Patterns
- "How does X work?": `find_identifier` → `read_source_range` → `trace_dependency_path`
- "Where is X used?": `find_identifier` → sample 2-3 callers with `read_source_range`
- Batch related lookups in one turn for efficiency
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
