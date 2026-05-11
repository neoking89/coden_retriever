"""Runtime-resolved defaults for MCP tool signatures.

Function defaults are captured at def-time, so they can't directly call
`load_config()` per invocation. We resolve the config value once at
module import and expose it as a constant that each MCP tool uses in
its signature. Changing `search.default_tokens` therefore takes effect
on the next process start (which is the standard contract for server
config — the MCP server reloads on restart anyway).
"""
from ..config_loader import get_default_token_budget

RESOLVED_DEFAULT_TOKEN_BUDGET: int = get_default_token_budget()
