"""Constants for the conversation-branching undo picker."""

# WHY 60: fits a one-line `tool_name(arg=..., arg=...)` preview in a typical
# 80-column terminal without wrapping. Longer previews are truncated with `...`.
MAX_ARGS_PREVIEW_LEN: int = 60

# WHY 14: matches MAX_VISIBLE_ITEMS in config_picker_layout.py so the two
# pickers share a consistent viewport height on standard terminals.
MAX_VISIBLE_ROWS: int = 14

# WHY 200: a directive is a single-turn steering instruction; users are
# expected to keep it short. Acts as a soft budget — longer input is still
# accepted but the picker toolbar warns near the limit.
DIRECTIVE_SOFT_LIMIT: int = 200

# Root branch identifier. Every tree starts with exactly one branch with this id.
ROOT_BRANCH_ID: str = "b0"

# WHY "b": matches the label_hint format ("b0", "b1", ...) used throughout the
# picker. Kept short so it fits in row prefixes like "[b3:7]".
BRANCH_ID_PREFIX: str = "b"
