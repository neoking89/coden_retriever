"""Debug adapters package.

Each module here implements a `DebugAdapter` for one language's DAP server.
The `REGISTRY` singleton in `registry.py` is populated at import time by each
adapter module's side-effectful registration line.
"""
from .availability import DebugAvailability, DebugDependencyStatus
from .base import DebugAdapter, IdentityPathMapper, LaunchConfig, PathMapper
from .registry import REGISTRY, AdapterRegistry, check_debug_availability

# Import adapter modules for side-effect registration. Keep this list ordered
# so MRO on extension collisions is deterministic (first registration wins).
from . import python  # noqa: F401  (registers PythonAdapter)
from . import go  # noqa: F401  (registers GoAdapter)
from . import ruby  # noqa: F401  (registers RubyAdapter)
from . import lldb  # noqa: F401  (registers LLDBAdapter + RustAdapter)
from . import netcoredbg  # noqa: F401  (registers NetcoredbgAdapter)
from . import powershell  # noqa: F401  (registers PSESAdapter)
from . import r  # noqa: F401  (registers RAdapter)
from . import lua  # noqa: F401  (registers LuaAdapter)
from . import bash  # noqa: F401  (registers BashAdapter)
from . import php  # noqa: F401  (registers PHPAdapter)
from . import kotlin  # noqa: F401  (registers KotlinAdapter)
from . import jsdebug  # noqa: F401  (registers JSDebugAdapter)
from . import java  # noqa: F401  (registers JavaAdapter)

__all__ = [
    "REGISTRY",
    "AdapterRegistry",
    "DebugAvailability",
    "DebugAdapter",
    "DebugDependencyStatus",
    "IdentityPathMapper",
    "LaunchConfig",
    "PathMapper",
    "check_debug_availability",
]
