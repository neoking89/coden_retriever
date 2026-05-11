"""Shared prompt_toolkit style and keybinding constants for picker UIs.

Used by directory_browser, tool_permission_picker, and tool_picker
to ensure consistent look and behavior across all interactive pickers.
"""

# === Keybinding key names ===
KEY_UP = "up"
KEY_DOWN = "down"
KEY_VIM_UP = "k"
KEY_VIM_DOWN = "j"
KEY_ENTER = "enter"
KEY_ESCAPE = "escape"
KEY_QUIT = "q"
KEY_CTRL_C = "c-c"

# === Shared style values ===
# Used in Style.from_dict() — only the values that are identical across all pickers
STYLE_SEPARATOR = "#666666"
STYLE_SELECTED = "bold reverse #00ff00"
STYLE_TOOLBAR = "bg:#333333 #ffffff"

# === Selection prefix indicators ===
SELECTED_PREFIX = " > "
UNSELECTED_PREFIX = "   "

# === Style class names ===
CLASS_HEADER = "class:header"
CLASS_SEPARATOR = "class:separator"
CLASS_SELECTED = "class:selected"
CLASS_DIM = "class:dim"
CLASS_TOOLBAR = "class:toolbar"

# === Content separator widths ===
# Width of dash separators inside picker/browser content areas
PICKER_SEPARATOR_WIDTH = 62   # Used by directory browser and tool picker
PERMISSION_SEPARATOR_WIDTH = 50  # Used by permission picker (narrower dialog)
