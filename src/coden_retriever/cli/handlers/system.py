"""Handlers for config and reset commands."""
import json
import sys
from pathlib import Path

from ...cache import CacheManager
from ...config_loader import (
    AppConfig,
    get_config,
    load_config,
    save_config,
    get_config_file,
    reset_config,
    set_config_value,
    SETTING_LOCATIONS,
    _config_to_dict,
)
from ...daemon.client import stop_daemon
from ...daemon.server import is_daemon_running


def handle_config_command(args: list[str]) -> int:
    """Handle config subcommands: show, path, reset, set."""
    if not args or args[0] == "show":
        config = load_config()
        print(json.dumps(_config_to_dict(config), indent=2))
        return 0

    elif args[0] == "path":
        print(get_config_file())
        return 0

    elif args[0] == "reset":
        if reset_config():
            print("Configuration reset to defaults")
            return 0
        else:
            print("Failed to reset configuration", file=sys.stderr)
            return 1

    elif args[0] == "set" and len(args) >= 3:
        return _handle_config_set(args)

    elif args[0] == "new" and len(args) == 2:
        return _handle_config_new(args[1])

    else:
        _print_config_usage()
        return 1


def _handle_config_new(path_str: str) -> int:
    """Write a fresh AppConfig() defaults JSON to `path_str`.

    Refuses to overwrite an existing file or write into a missing parent
    directory — both are likely typos, and silently creating them would
    hide the user's mistake.
    """
    target = Path(path_str).expanduser().resolve()

    # Directory shorthand: `coden config new .` or `... ~/foo/` means
    # "drop a settings.json into that directory".
    if target.is_dir():
        target = target / "settings.json"

    if target.exists():
        print(f"Error: {target} already exists. Refusing to overwrite.", file=sys.stderr)
        return 1

    if not target.parent.exists():
        print(f"Error: parent directory does not exist: {target.parent}", file=sys.stderr)
        return 1

    if not save_config(AppConfig(), path=target):
        print(f"Error: failed to write {target}", file=sys.stderr)
        return 1

    print(f"Created {target}")
    print(f"Edit it, then run: coden -a --config {target}")
    return 0


def _handle_config_set(args: list[str]) -> int:
    """Handle 'config set <key> <value>' subcommand."""
    key_path = args[1]
    value = args[2]

    config = load_config()
    parts = key_path.split(".")

    if len(parts) == 2:
        section, key = parts
        if key in SETTING_LOCATIONS:
            expected_section = SETTING_LOCATIONS[key][0]
            if section != expected_section:
                print(f"Key '{key}' belongs to section '{expected_section}', not '{section}'", file=sys.stderr)
                return 1
    elif len(parts) == 1:
        key = parts[0]
    else:
        print(f"Invalid key format: {key_path}. Use key or section.key (e.g., debug or agent.debug)", file=sys.stderr)
        return 1

    success, error = set_config_value(config, key, value)
    if not success:
        print(error, file=sys.stderr)
        return 1

    save_config(config)
    print(f"Set {key} = {value}")
    return 0


def _print_config_usage() -> None:
    """Print config command usage help."""
    print("Usage: coden config [show|path|reset|set <key> <value>|new <path>]")
    print("\nCommands:")
    print("  show             Show current configuration")
    print("  path             Show config file path")
    print("  reset            Reset configuration to defaults")
    print("  set <key> <val>  Set a configuration value")
    print("  new <path>       Write a fresh defaults JSON to <path>")
    print("                   Use with: coden -a --config <path>")
    print("\nKeys:")
    print("  model.default, model.base_url")
    print("  agent.max_steps, agent.max_retries, agent.debug")
    print("  daemon.host, daemon.port, daemon.daemon_timeout, daemon.max_projects")
    print("  search.default_tokens, search.default_limit, search.semantic_model_path")


def handle_reset_command(keep_config: bool = False) -> int:
    """Handle reset command: clear all caches, stop daemon, reset config."""
    exit_code = 0
    config = get_config()

    print("Clearing all caches...")
    count, errors = CacheManager.clear_all_caches()
    if count > 0:
        print(f"  Cleared {count} project cache(s)")
    else:
        print("  No caches to clear")
    for error in errors:
        print(f"  Warning: {error}", file=sys.stderr)
        exit_code = 1

    print("Stopping daemon...")
    running, pid = is_daemon_running()
    if not running:
        print("  Daemon is not running")
    else:
        if stop_daemon(config.daemon.address):
            print(f"  Daemon stopped (was PID: {pid})")
        else:
            print(f"  Failed to stop daemon (PID: {pid})", file=sys.stderr)
            exit_code = 1

    if keep_config:
        print("Skipping configuration reset (--keep-config)")
    else:
        print("Resetting configuration...")
        if reset_config():
            print("  Configuration reset to defaults")
        else:
            print("  Failed to reset configuration", file=sys.stderr)
            exit_code = 1

    if exit_code == 0:
        print("\nReset complete.")
    else:
        print("\nReset completed with warnings.", file=sys.stderr)

    return exit_code
