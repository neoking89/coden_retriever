"""Interactive directory browser for the CLI agent.

Provides a beautiful, keyboard-navigable directory browser using prompt_toolkit
and Rich for the best user experience. Allows users to:
- Navigate directories with arrow keys
- See directory contents preview
- Jump to home, parent, or type paths directly
- Quick filter with search
"""

from pathlib import Path
from typing import Optional

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from rich import box
from rich.panel import Panel
from rich.text import Text

from .rich_console import console
from .ui_utils import calculate_viewport, format_scroll_indicator


class DirectoryBrowser:
    """Interactive directory browser with keyboard navigation."""

    def __init__(self, start_path: Optional[str] = None):
        """Initialize the directory browser.

        Args:
            start_path: Starting directory path. Defaults to cwd.
        """
        self.current_path = Path(start_path or Path.cwd()).resolve()
        self.selected_index = 0
        self.entries: list[Path] = []
        self.filter_text = ""
        self.selected_path: Optional[Path] = None
        self.cancelled = False
        self._refresh_entries()

    def _refresh_entries(self) -> None:
        """Refresh the directory entries list."""
        self.entries = []

        try:
            # Add parent directory option (if not at root)
            if self.current_path.parent != self.current_path:
                pass  # We'll handle ".." specially in display

            # Get directories only, sorted
            dirs = []
            for entry in self.current_path.iterdir():
                try:
                    if entry.is_dir() and not entry.name.startswith('.'):
                        dirs.append(entry)
                except PermissionError:
                    continue

            # Sort directories alphabetically (case-insensitive)
            dirs.sort(key=lambda p: p.name.lower())

            # Apply filter if set
            if self.filter_text:
                filter_lower = self.filter_text.lower()
                dirs = [d for d in dirs if filter_lower in d.name.lower()]

            self.entries = dirs

        except PermissionError:
            self.entries = []

        # Clamp selected index
        if self.entries:
            self.selected_index = min(self.selected_index, len(self.entries) - 1)
        else:
            self.selected_index = 0

    def _get_preview(self) -> list[str]:
        """Get preview of selected directory contents."""
        if not self.entries:
            return ["(empty)"]

        if self.selected_index >= len(self.entries):
            return []

        selected = self.entries[self.selected_index]
        preview = []

        try:
            items = list(selected.iterdir())
            dirs = [i for i in items if i.is_dir() and not i.name.startswith('.')]
            files = [i for i in items if i.is_file() and not i.name.startswith('.')]

            dirs.sort(key=lambda p: p.name.lower())
            files.sort(key=lambda p: p.name.lower())

            # Show up to 8 items
            for d in dirs[:4]:
                preview.append(f"  {d.name}/")
            for f in files[:4]:
                preview.append(f"  {f.name}")

            remaining = len(dirs) + len(files) - len(preview)
            if remaining > 0:
                preview.append(f"  ... +{remaining} more")

        except PermissionError:
            preview.append("  (access denied)")

        return preview if preview else ["  (empty)"]


def run_directory_browser(start_path: Optional[str] = None) -> Optional[str]:
    """Run the interactive directory browser.

    Args:
        start_path: Starting directory path. Defaults to cwd.

    Returns:
        Selected directory path as string, or None if cancelled.
    """
    browser = DirectoryBrowser(start_path)

    # Create key bindings
    kb = KeyBindings()

    @kb.add('up')
    @kb.add('k')
    def move_up(event):
        if browser.selected_index > 0:
            browser.selected_index -= 1

    @kb.add('down')
    @kb.add('j')
    def move_down(event):
        if browser.selected_index < len(browser.entries) - 1:
            browser.selected_index += 1

    @kb.add('enter')
    @kb.add('right')
    @kb.add('l')
    def enter_dir(event):
        if browser.entries and browser.selected_index < len(browser.entries):
            new_path = browser.entries[browser.selected_index]
            if new_path.is_dir():
                browser.current_path = new_path.resolve()
                browser.selected_index = 0
                browser.filter_text = ""
                browser._refresh_entries()

    @kb.add('left')
    @kb.add('h')
    @kb.add('backspace')
    def go_parent(event):
        parent = browser.current_path.parent
        if parent != browser.current_path:
            # Remember current directory name to select it after going up
            current_name = browser.current_path.name
            browser.current_path = parent.resolve()
            browser.filter_text = ""
            browser._refresh_entries()
            # Try to select the directory we came from
            for i, entry in enumerate(browser.entries):
                if entry.name == current_name:
                    browser.selected_index = i
                    break
            else:
                browser.selected_index = 0

    @kb.add('~')
    def go_home(event):
        browser.current_path = Path.home().resolve()
        browser.selected_index = 0
        browser.filter_text = ""
        browser._refresh_entries()

    @kb.add(' ')
    def select_current(event):
        # Space selects current directory (not entering it)
        browser.selected_path = browser.current_path
        event.app.exit()

    @kb.add('.')
    def toggle_hidden(event):
        # Could toggle hidden files, but for now just refresh
        browser._refresh_entries()

    @kb.add('escape')
    @kb.add('q')
    def cancel(event):
        browser.cancelled = True
        event.app.exit()

    @kb.add('c-c')
    def ctrl_c(event):
        browser.cancelled = True
        event.app.exit()

    # Filter input with any printable character
    @kb.add('/')
    def start_filter(event):
        # Start filter mode - subsequent chars will filter
        pass

    def get_content():
        """Generate the browser content."""
        lines = []

        # Header with current path
        path_display = str(browser.current_path)
        if len(path_display) > 60:
            path_display = "..." + path_display[-57:]

        lines.append(('class:header', f' {path_display}\n'))
        lines.append(('class:separator', ' ' + '-' * 62 + '\n'))

        # Parent directory option
        if browser.current_path.parent != browser.current_path:
            lines.append(('class:parent', '   .. (parent directory)\n'))

        # Directory entries
        if browser.entries:
            # Calculate visible range (scrolling)
            max_visible = 15
            start, end = calculate_viewport(browser.selected_index, len(browser.entries), max_visible)

            for i in range(start, end):
                entry = browser.entries[i]
                prefix = ' > ' if i == browser.selected_index else '   '
                style = 'class:selected' if i == browser.selected_index else 'class:dir'
                lines.append((style, f'{prefix}{entry.name}/\n'))

            # Scroll indicator
            if len(browser.entries) > max_visible:
                scroll_info = format_scroll_indicator(browser.selected_index, len(browser.entries))
                lines.append(('class:dim', scroll_info + '\n'))
        else:
            lines.append(('class:dim', '   (no subdirectories)\n'))

        # Preview section
        lines.append(('class:separator', '\n ' + '-' * 62 + '\n'))
        lines.append(('class:preview-header', ' Preview:\n'))

        for preview_line in browser._get_preview()[:5]:
            lines.append(('class:preview', preview_line + '\n'))

        return FormattedText(lines)

    def get_toolbar():
        """Generate the bottom toolbar."""
        return HTML(
            '<b>[Enter]</b> Open  '
            '<b>[Space]</b> Select here  '
            '<b>[Backspace]</b> Parent  '
            '<b>[~]</b> Home  '
            '<b>[q/Esc]</b> Cancel'
        )

    # Create layout
    body = Window(
        content=FormattedTextControl(get_content),
        wrap_lines=False,
    )

    toolbar = Window(
        content=FormattedTextControl(get_toolbar),
        height=1,
        style='class:toolbar',
    )

    root_container = HSplit([
        Frame(
            body,
            title='Directory Browser',
        ),
        toolbar,
    ])

    # Styles
    style = Style.from_dict({
        'header': 'bold #00aa00',
        'separator': '#666666',
        'parent': '#888888 italic',
        'dir': '#00aaff',
        'selected': 'bold reverse #00ff00',
        'dim': '#666666',
        'preview-header': 'bold #888888',
        'preview': '#666666',
        'toolbar': 'bg:#333333 #ffffff',
        'frame.border': '#00aa00',
    })

    # Create and run application
    app = Application(
        layout=Layout(root_container),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=True,
    )

    # Clear screen and show browser
    console.print()
    console.print("[bold green]Directory Browser[/bold green]")
    console.print("[dim]Navigate with arrow keys, Enter to open, Space to select, q to cancel[/dim]")
    console.print()

    app.run()

    if browser.cancelled:
        console.print("[yellow]Cancelled[/yellow]")
        return None

    if browser.selected_path:
        return str(browser.selected_path)

    # If they navigated into a directory and didn't explicitly cancel,
    # return that directory
    return str(browser.current_path)


async def run_directory_browser_async(start_path: Optional[str] = None) -> Optional[str]:
    """Async version of the directory browser.

    Args:
        start_path: Starting directory path. Defaults to cwd.

    Returns:
        Selected directory path as string, or None if cancelled.
    """
    # Run in executor to not block async loop
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_directory_browser, start_path)


def print_directory_changed(old_path: str, new_path: str) -> None:
    """Print a nice directory change confirmation.

    Args:
        old_path: Previous working directory.
        new_path: New working directory.
    """
    console.print()

    # Create a nice panel showing the change
    content = Text()
    content.append("Previous: ", style="dim")
    content.append(f"{old_path}\n", style="dim italic")
    content.append("Current:  ", style="bold green")
    content.append(f"{new_path}", style="bold")

    panel = Panel(
        content,
        title="[bold green]Working Directory Changed[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)
    console.print()
