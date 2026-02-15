"""Handler for cache management commands."""
import json
import sys
from pathlib import Path

from ...cache import CacheManager
from ...config import get_central_cache_root, get_project_cache_dir


def handle_cache_command(args: list[str]) -> int:
    """Handle cache subcommands: list, clear, status, path."""
    if not args or args[0] == "list":
        return _cache_list()
    elif args[0] == "clear":
        return _cache_clear(args)
    elif args[0] == "status":
        return _cache_status(args)
    elif args[0] == "path":
        return _cache_path(args)
    else:
        _print_cache_usage()
        return 1


def _cache_list() -> int:
    """List all cached projects."""
    caches = CacheManager.list_all_caches()
    if not caches:
        print("No cached projects found.")
        print(f"Cache directory: {get_central_cache_root()}")
        return 0

    print(f"Cached projects ({len(caches)}):")
    print(f"Cache directory: {get_central_cache_root()}\n")

    # Max display length for source paths before truncation
    MAX_PATH_DISPLAY_LEN = 60
    # Characters to keep from end of truncated path (after "...")
    PATH_TAIL_LEN = 57

    total_size = 0
    for cache_info in caches:
        total_size += cache_info["size_mb"]
        source = cache_info["source_dir"]
        if len(source) > MAX_PATH_DISPLAY_LEN:
            source = "..." + source[-PATH_TAIL_LEN:]
        print(f"  {source}")
        print(f"    Entities: {cache_info['entity_count']:,} | Files: {cache_info['file_count']:,} | Size: {cache_info['size_mb']:.1f} MB")
        if cache_info.get("updated_at"):
            print(f"    Updated: {cache_info['updated_at']}")
        print()

    print(f"Total cache size: {total_size:.1f} MB")
    return 0


def _cache_clear(args: list[str]) -> int:
    """Clear cache(s) based on arguments."""
    clear_all = "--all" in args or "-a" in args

    if clear_all:
        count, errors = CacheManager.clear_all_caches()
        if count > 0:
            print(f"Cleared {count} project cache(s)")
        else:
            print("No caches to clear")
        for error in errors:
            print(f"  Warning: {error}", file=sys.stderr)
        return 0 if not errors else 1

    path_arg = None
    for arg in args[1:]:
        if not arg.startswith("-"):
            path_arg = arg
            break

    target_path = Path(path_arg).resolve() if path_arg else Path.cwd()

    if not target_path.exists():
        print(f"Path does not exist: {target_path}", file=sys.stderr)
        return 1

    if not target_path.is_dir():
        print(f"Path is not a directory: {target_path}", file=sys.stderr)
        return 1

    cache_dir = get_project_cache_dir(target_path)
    if not cache_dir.exists():
        print(f"No cache found for: {target_path}")
        return 0

    if CacheManager.clear_cache_by_source_dir(target_path):
        print(f"Cache cleared for: {target_path}")
        return 0
    else:
        print(f"Failed to clear cache for: {target_path}", file=sys.stderr)
        return 1


def _cache_status(args: list[str]) -> int:
    """Show cache status for a project."""
    path_arg = args[1] if len(args) > 1 else None
    target_path = Path(path_arg).resolve() if path_arg else Path.cwd()

    if not target_path.exists():
        print(f"Path does not exist: {target_path}", file=sys.stderr)
        return 1

    if not target_path.is_dir():
        print(f"Path is not a directory: {target_path}", file=sys.stderr)
        return 1

    cache = CacheManager(target_path)
    status = cache.get_cache_status()
    print(json.dumps(status, indent=2))
    return 0


def _cache_path(args: list[str]) -> int:
    """Show cache path for a project."""
    path_arg = args[1] if len(args) > 1 else None
    target_path = Path(path_arg).resolve() if path_arg else Path.cwd()

    cache_dir = get_project_cache_dir(target_path)
    print(f"Project: {target_path}")
    print(f"Cache:   {cache_dir}")
    print(f"Exists:  {cache_dir.exists()}")
    return 0


def _print_cache_usage() -> None:
    """Print cache command usage help."""
    print("Usage: coden cache [list|clear|status|path]")
    print("\nCommands:")
    print("  list              List all cached projects")
    print("  clear             Clear cache for current directory")
    print("  clear <path>      Clear cache for specific project")
    print("  clear --all       Clear ALL cached projects")
    print("  status [path]     Show cache status for project")
    print("  path [path]       Show cache directory path for project")
    print(f"\nCache location: {get_central_cache_root()}")
