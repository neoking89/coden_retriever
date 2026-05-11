"""Interactive Tool Wizard for manual tool execution.

Provides a TUI-based wizard that allows users to:
- Browse available MCP tools via a menu
- Configure tool parameters via a guided form
- Execute tools directly (bypassing LLM inference)
- Inject results into the agent's conversation history
"""

import json
from dataclasses import dataclass
from typing import Any, Sequence

from mcp.types import Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.mcp import MCPServerStdio
from rich import box
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text

from .rich_console import console, format_exception_message
from ..mcp.constants import get_tool_categories


@dataclass
class ToolParameter:
    """Represents a tool parameter extracted from JSON Schema."""

    name: str
    param_type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    default: Any | None
    required: bool
    choices: list[str] | None = None  # For enum types
    minimum: int | None = None
    maximum: int | None = None


@dataclass
class ManualToolResult:
    """Result of a manually executed tool."""

    tool_name: str
    args: dict[str, Any]
    output: Any
    success: bool
    error: str | None = None


@dataclass
class WizardContext:
    """Context passed through the wizard."""

    root_directory: str
    available_tools: Sequence[Tool]
    server: MCPServerStdio


AUTO_FILL_PARAMS = {
    "root_directory": lambda ctx: ctx.root_directory,
}

def introspect_tool(tool: Tool) -> list[ToolParameter]:
    """Extract parameters from MCP Tool's inputSchema.

    Uses JSON Schema structure to build parameter metadata.
    No regex parsing - pure schema inspection.

    Args:
        tool: The MCP tool to introspect.

    Returns:
        List of ToolParameter objects.
    """
    schema = tool.inputSchema
    properties = schema.get("properties", {})
    required_params = set(schema.get("required", []))

    params = []
    for name, prop in properties.items():
        # Determine parameter type
        param_type = prop.get("type", "string")

        # Handle enum (choices)
        choices = prop.get("enum")

        # Handle constraints
        minimum = prop.get("minimum") or prop.get("ge")
        maximum = prop.get("maximum") or prop.get("le")

        param = ToolParameter(
            name=name,
            param_type=param_type,
            description=prop.get("description", ""),
            default=prop.get("default"),
            required=name in required_params,
            choices=choices,
            minimum=minimum,
            maximum=maximum,
        )
        params.append(param)

    return params


def print_tool_menu(tools: Sequence[Tool]) -> list[Tool]:
    """Display the tool selection menu grouped by category.

    Args:
        tools: List of MCP tools to display.

    Returns:
        Ordered list of tools matching the displayed numbering.
    """
    console.print()

    # Build tool lookup
    tool_map = {tool.name: tool for tool in tools}

    # Track displayed tools and build numbered index
    displayed_tools: list[Tool] = []
    tool_number = 1

    table = Table(
        title="Run Tool - Select a Tool",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold green",
    )
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    # Display tools by category
    for category_name, category_tools in get_tool_categories():
        # Get tools in this category that are available
        available_in_category = [
            tool_map[name] for name in category_tools if name in tool_map
        ]
        if not available_in_category:
            continue

        # Add category header row
        table.add_row("", f"[bold dim]--- {category_name} ---[/bold dim]", "")

        for tool in available_in_category:
            description = tool.description or ""
            first_line = description.split("\n")[0].strip()
            if len(first_line) > 55:
                first_line = first_line[:52] + "..."
            table.add_row(str(tool_number), tool.name, first_line)
            displayed_tools.append(tool)
            tool_number += 1

    # Display uncategorized tools
    displayed_names = {t.name for t in displayed_tools}
    uncategorized = [t for t in tools if t.name not in displayed_names]
    if uncategorized:
        table.add_row("", "[bold dim]--- Other ---[/bold dim]", "")
        for tool in uncategorized:
            description = tool.description or ""
            first_line = description.split("\n")[0].strip()
            if len(first_line) > 55:
                first_line = first_line[:52] + "..."
            table.add_row(str(tool_number), tool.name, first_line)
            displayed_tools.append(tool)
            tool_number += 1

    console.print(table)
    console.print()

    return displayed_tools


def print_wizard_header(tool_name: str) -> None:
    """Display header for the parameter configuration form."""
    console.print()
    header = Text()
    header.append("Configuring: ", style="bold")
    header.append(tool_name, style="bold cyan")
    console.print(Panel(header, border_style="cyan"))


def print_parameter_summary(tool_name: str, args: dict[str, Any]) -> None:
    """Display summary of configured parameters before execution."""
    console.print()

    table = Table(
        title=f"Configuration Summary: {tool_name}",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold yellow",
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for key, value in args.items():
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        table.add_row(key, value_str)

    console.print(table)
    console.print()


def _format_output_for_display(output: Any) -> str:
    """Format tool output for human-readable display.

    Handles various output types:
    - Dicts with multiline string values: Extract and display text directly
    - Other dicts: JSON with preserved Unicode
    - Strings: Display directly
    - Other: Convert to string

    Args:
        output: The raw tool output.

    Returns:
        Formatted string for display.
    """
    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        # Find any string values that contain newlines (formatted text)
        # These should be displayed directly, not as escaped JSON
        formatted_parts = []
        other_fields = {}

        for key, value in output.items():
            if isinstance(value, str) and "\n" in value:
                # This is formatted text - display with header
                formatted_parts.append(f"[{key}]\n{value}")
            else:
                other_fields[key] = value

        if formatted_parts:
            # Show non-text fields first, then formatted text
            result_parts = []
            if other_fields:
                result_parts.append(
                    json.dumps(other_fields, indent=2, ensure_ascii=False, default=str)
                )
            result_parts.extend(formatted_parts)
            return "\n\n".join(result_parts)

        # No formatted text found - use JSON with preserved Unicode
        return json.dumps(output, indent=2, ensure_ascii=False, default=str)

    # Fallback for other types
    return str(output)


def print_manual_result(result: ManualToolResult) -> None:
    """Display manual tool execution result in a distinct panel."""
    console.print()

    if result.success:
        # Format output for readable display
        output_str = _format_output_for_display(result.output)

        # Truncate if too long
        if len(output_str) > 2000:
            output_str = output_str[:2000] + "\n... (truncated)"

        panel = Panel(
            output_str,
            title=f"[bold green]Manual Result: {result.tool_name}[/bold green]",
            border_style="green",
            box=box.DOUBLE,
        )
    else:
        panel = Panel(
            result.error or "Unknown error",
            title=f"[bold red]Error: {result.tool_name}[/bold red]",
            border_style="red",
            box=box.DOUBLE,
        )

    console.print(panel)
    console.print()


def print_wizard_cancelled() -> None:
    """Display cancellation message."""
    console.print()
    console.print("[dim]Cancelled. Returning to chat.[/dim]")
    console.print()


def print_context_injected() -> None:
    """Display message confirming context injection."""
    console.print()
    info = Text()
    info.append("Context injected. ", style="bold green")
    info.append("The agent now has access to this data.", style="dim")
    console.print(info)
    console.print()


def prompt_for_parameter(
    param: ToolParameter,
    context: WizardContext,
) -> tuple[Any, bool]:
    """Prompt user for a single parameter value.

    Args:
        param: The parameter to prompt for.
        context: Wizard context with auto-fill values.

    Returns:
        Tuple of (value, was_cancelled).
    """
    # Check for auto-fill
    if param.name in AUTO_FILL_PARAMS:
        auto_value = AUTO_FILL_PARAMS[param.name](context)
        return auto_value, False

    # Build prompt text
    prompt_text = Text()
    prompt_text.append(f"{param.name}", style="cyan")
    if param.required:
        prompt_text.append(" *", style="bold red")
    if param.description:
        desc = param.description
        if len(desc) > 80:
            desc = desc[:77] + "..."
        prompt_text.append(f"\n  {desc}", style="dim")

    console.print(prompt_text)

    try:
        # Handle different parameter types
        if param.param_type == "boolean":
            default = param.default if param.default is not None else False
            value = Confirm.ask("  Value", default=default)

        elif param.param_type == "integer":
            default = param.default
            while True:
                try:
                    if default is not None:
                        value = IntPrompt.ask("  Value", default=default)
                    else:
                        value = IntPrompt.ask("  Value")

                    # Validate constraints
                    if param.minimum is not None and value < param.minimum:
                        console.print(
                            f"  [red]Value must be >= {param.minimum}[/red]"
                        )
                        continue
                    if param.maximum is not None and value > param.maximum:
                        console.print(
                            f"  [red]Value must be <= {param.maximum}[/red]"
                        )
                        continue
                    break
                except ValueError:
                    console.print("  [red]Please enter a valid integer[/red]")

        elif param.choices:
            # Enum type - show choices
            console.print("  Choices:")
            for idx, choice in enumerate(param.choices, 1):
                marker = " (default)" if choice == param.default else ""
                console.print(f"    {idx}. {choice}{marker}")

            default_idx = None
            if param.default in param.choices:
                default_idx = param.choices.index(param.default) + 1

            while True:
                if default_idx:
                    choice_idx = IntPrompt.ask(
                        "  Select", default=default_idx
                    )
                else:
                    choice_idx = IntPrompt.ask("  Select")

                if 1 <= choice_idx <= len(param.choices):
                    value = param.choices[choice_idx - 1]
                    break
                console.print(
                    f"  [red]Please select 1-{len(param.choices)}[/red]"
                )

        else:
            # String type
            default = param.default if param.default is not None else ""
            while True:
                if default:
                    value = Prompt.ask("  Value", default=str(default))
                else:
                    value = Prompt.ask("  Value")

                if value or not param.required:
                    break
                console.print("  [red]This field is required[/red]")

        console.print()  # Spacing between parameters
        return value, False

    except KeyboardInterrupt:
        return None, True


def collect_parameters(
    tool: Tool,
    context: WizardContext,
) -> tuple[dict[str, Any] | None, bool]:
    """Collect all parameters for a tool via interactive prompts.

    Args:
        tool: The tool to collect parameters for.
        context: Wizard context with auto-fill values.

    Returns:
        Tuple of (arguments dict, was_cancelled).
    """
    params = introspect_tool(tool)
    args: dict[str, Any] = {}

    print_wizard_header(tool.name)
    console.print("[dim]Press Ctrl+C to cancel[/dim]")
    console.print()

    # Separate auto-filled and user-prompted params
    user_params = [p for p in params if p.name not in AUTO_FILL_PARAMS]
    auto_params = [p for p in params if p.name in AUTO_FILL_PARAMS]

    # Collect auto-filled params silently
    for param in auto_params:
        value, cancelled = prompt_for_parameter(param, context)
        if cancelled:
            return None, True
        args[param.name] = value

    # Collect user-prompted params
    for param in user_params:
        value, cancelled = prompt_for_parameter(param, context)
        if cancelled:
            return None, True
        args[param.name] = value

    return args, False


async def execute_tool(
    server: MCPServerStdio,
    tool_name: str,
    arguments: dict[str, Any],
) -> ManualToolResult:
    """Execute a tool directly via MCP server.

    Uses direct_call_tool which bypasses the agent context requirement,
    allowing manual tool execution outside the normal agent flow.

    Args:
        server: The MCP server connection.
        tool_name: Name of the tool to execute.
        arguments: Arguments to pass to the tool.

    Returns:
        ManualToolResult with execution outcome.
    """
    try:
        with Status(
            f"[bold cyan]Running {tool_name}...[/bold cyan]",
            console=console,
            spinner="dots",
        ):
            # Use direct_call_tool for manual execution (no RunContext needed)
            # Returns either:
            # - structured content (dict/primitive) if available
            # - mapped content parts (single item or list)
            result = await server.direct_call_tool(tool_name, arguments)

        # Parse result - direct_call_tool can return various types
        output = _parse_tool_result(result)

        return ManualToolResult(
            tool_name=tool_name,
            args=arguments,
            output=output,
            success=True,
            error=None,
        )

    except Exception as e:
        return ManualToolResult(
            tool_name=tool_name,
            args=arguments,
            output=None,
            success=False,
            error=format_exception_message(e),
        )


def _parse_tool_result(result: Any) -> Any:
    """Parse the result from direct_call_tool.

    direct_call_tool can return:
    - A dict (structured content)
    - A primitive value (str, int, etc.)
    - A list of content parts
    - A single content part with 'text' attribute

    Args:
        result: The raw result from direct_call_tool.

    Returns:
        Parsed output suitable for display.
    """
    # If it's already a dict or primitive, return as-is
    if isinstance(result, (dict, str, int, float, bool, type(None))):
        return result

    # If it's a list, process each item
    if isinstance(result, list):
        text_parts = []
        for part in result:
            if hasattr(part, "text"):
                text_parts.append(part.text)
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                text_parts.append(str(part))

        if text_parts:
            combined = "\n".join(text_parts)
            # Try to parse as JSON
            try:
                return json.loads(combined)
            except json.JSONDecodeError:
                return combined
        return result

    # If it has a 'text' attribute (single content part)
    if hasattr(result, "text"):
        text = result.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    # Fallback: convert to string
    return str(result)


def inject_manual_tool_result(
    history: list[ModelMessage],
    result: ManualToolResult,
) -> list[ModelMessage]:
    """Inject a manually executed tool result into conversation history.

    Args:
        history: Current conversation history.
        result: The manual tool execution result.

    Returns:
        Updated history with injected result.
    """
    # Format output for injection
    if isinstance(result.output, dict):
        output_str = json.dumps(result.output, indent=2, default=str)
    else:
        output_str = str(result.output)

    # Truncate very long outputs
    if len(output_str) > 10000:
        output_str = output_str[:10000] + "\n... (output truncated)"

    # Build the injection message
    args_str = json.dumps(result.args, indent=2)

    content = f"""The user manually executed tool '{result.tool_name}' with the following arguments:
```json
{args_str}
```

Here is the output:
```
{output_str}
```

You can now analyze this data or use it to inform your responses. The user may ask questions about this result."""

    # Create the injection message
    injection = ModelRequest(parts=[UserPromptPart(content=content)])

    return list(history) + [injection]


async def run_tool_wizard(
    tools: Sequence[Tool],
    root_directory: str,
    server: MCPServerStdio,
) -> ManualToolResult | None:
    """Run the interactive tool wizard.

    Args:
        tools: Available MCP tools.
        root_directory: Current working directory for context.
        server: MCP server connection for tool execution.

    Returns:
        ManualToolResult if tool was executed, None if cancelled.
    """
    context = WizardContext(
        root_directory=root_directory,
        available_tools=tools,
        server=server,
    )

    try:
        # Step 1: Display menu and get tool selection
        ordered_tools = print_tool_menu(tools)

        console.print("[dim]Enter tool number (or 0 to cancel):[/dim]")
        while True:
            try:
                selection = IntPrompt.ask("Select tool")
                if selection == 0:
                    print_wizard_cancelled()
                    return None
                if 1 <= selection <= len(ordered_tools):
                    break
                console.print(
                    f"[red]Please enter 1-{len(ordered_tools)} (or 0 to cancel)[/red]"
                )
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")

        selected_tool = ordered_tools[selection - 1]

        # Step 2: Collect parameters
        args, cancelled = collect_parameters(selected_tool, context)
        if cancelled or args is None:
            print_wizard_cancelled()
            return None

        # Step 3: Show summary and confirm
        print_parameter_summary(selected_tool.name, args)

        # Simple ENTER-to-confirm prompt (no y/n required)
        console.print("[dim]Press ENTER to execute, Ctrl+C to cancel[/dim]")
        try:
            Prompt.ask("", default="")  # Accept ENTER directly
        except (KeyboardInterrupt, EOFError):
            print_wizard_cancelled()
            return None

        # Step 4: Execute tool
        result = await execute_tool(server, selected_tool.name, args)

        # Step 5: Display result
        print_manual_result(result)

        if result.success:
            print_context_injected()

        return result

    except KeyboardInterrupt:
        print_wizard_cancelled()
        return None
