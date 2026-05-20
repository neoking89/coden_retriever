"""Argument parser creation for search and flag-arguments."""
import argparse

from ..config_loader import AppConfig
from ..constants import (
    DEFAULT_CLONE_SEMANTIC_WEIGHT,
    DEFAULT_CLONE_SYNTACTIC_WEIGHT,
    DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
    DEFAULT_SYNTACTIC_LINE_THRESHOLD,
    FLAG_MIN_LINES,
    MAGIC_CONSTANT_DEFAULT_MIN_FILES,
    MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
    TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
    TRAMP_DATA_MIN_GROUP_SIZE,
)
from .threshold_config import THRESHOLD_CONFIGS, add_threshold_argument
from .utils import DefaultValueHelpFormatter

# Validation floor for --min-lines argument
_MIN_LINES_FLOOR = 1

_SEARCH_EPILOG = """\
[bold cyan]Start by running coden on your repo to see what matters:[/]

  coden src                              ranked map of src/
  coden src --stats -n 50 -r             top 50, reversed, with ranking stats
  coden src --map-mode simple            map ranked by git commits per object (linecount fallback)

[bold green]Looking for something specific? Search by keyword, or go semantic:[/] 

  coden src -q "auth"                    keyword search for "auth"
  coden src -q "how does login work" -s  semantic search (bundled MiniLM ONNX)
  coden --find UserAuth --show-deps      find a symbol and its callers/callees

[bold yellow]Now dig deeper. Run analysis to find what needs attention:[/] 

  coden src -H --stats -r                refactoring hotspots (coupling x complexity)
  coden src -C --clone-threshold 0.90    code clones above 90% similarity
  coden src -P --breakdown               architecture health per module
  coden src -D                           dead code - functions nobody calls
  coden src -T --min-occurrences 5       tramp data - params traveling together
  coden src -S                           hardcoded secrets and API keys
  coden src -K                           magic constants - repeated literals

[bold red]Found issues? Flag them in your source code, or fix them directly:[/] 

  coden flag -HPCEDTSK --dry-run         preview all flags before writing
  coden flag -HPCEDTSK --backup          write \\[CODEN] comments (with backup)
  coden flag -E --remove-comments --backup  remove echo comments entirely
  coden flag -S --replace --backup       redact secrets with ***REDACTED***
  coden flag clear                       clean up all \\[CODEN] markers

[bold magenta]For repeated queries, the daemon keeps indices in memory so you skip startup:[/]

  coden daemon start                     start background daemon for caching and faster queries
  coden src -q "auth"                    queries use daemon automatically
  coden daemon status                    check what's cached and how long it's been up
  coden daemon stop                      shut it down

[bold blue]Connect coden to your editor or AI assistant:[/]

  coden serve                            MCP server over stdio
  coden -a                               interactive agent with local LLM
  coden -a --base-url http://localhost:1234/v1 --model my-model
  coden debug-availability python        check whether a language can be debugged here

[bold dim]Starting fresh? Reset clears caches, stops the daemon, and resets config:[/]

  coden reset                            full reset (destructive!)

[dim]Subcommands: serve, agent (-a), daemon, cache, config, debug-availability, flag, reset[/]
"""


def _validate_min_lines(value: str) -> int:
    """Validate --min-lines argument is at least 1."""
    ival = int(value)
    if ival < _MIN_LINES_FLOOR:
        raise argparse.ArgumentTypeError(
            f"min-lines must be at least {_MIN_LINES_FLOOR}, got {ival}"
        )
    return ival


def add_flag_arguments(parser: argparse.ArgumentParser, config: AppConfig) -> None:
    """Add common flag arguments to a parser."""
    parser.add_argument("root", nargs="?", default=".",
                        help="Repository root directory")

    analysis_group = parser.add_argument_group("Analysis Types (at least one required)")
    analysis_group.add_argument("-H", "--hotspots", action="store_true",
                                help="Flag coupling hotspots")
    analysis_group.add_argument("-P", "--propagation", action="store_true",
                                help="Flag high propagation cost functions")
    analysis_group.add_argument("-C", "--clones", action="store_true",
                                help="Flag code clones (combined semantic + syntactic by default)")
    analysis_group.add_argument("--clone-semantic", action="store_true",
                                help="Clone detection: semantic only (MiniLM ONNX embeddings)")
    analysis_group.add_argument("--clone-syntactic", action="store_true",
                                help="Clone detection: syntactic only (line-by-line Jaccard)")
    analysis_group.add_argument("--line-threshold", type=float, default=DEFAULT_SYNTACTIC_LINE_THRESHOLD,
                                help="Line similarity threshold for syntactic clones")
    analysis_group.add_argument("--func-threshold", type=float, default=DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
                                help="Function match threshold for syntactic clones")
    analysis_group.add_argument("--semantic-weight", type=float, default=DEFAULT_CLONE_SEMANTIC_WEIGHT,
                                help="Weight for semantic similarity in combined score")
    analysis_group.add_argument("--syntactic-weight", type=float, default=DEFAULT_CLONE_SYNTACTIC_WEIGHT,
                                help="Weight for syntactic similarity in combined score")
    analysis_group.add_argument("-E", "--echo-comments", action="store_true",
                                help="Detect and flag echo comments")
    analysis_group.add_argument("-D", "--dead-code", action="store_true",
                                help="Flag dead code - functions/methods with no callers in the codebase")
    analysis_group.add_argument("-T", "--tramp-data", action="store_true",
                                help="Flag tramp data - parameters appearing across many functions")
    analysis_group.add_argument("-S", "--sensitive-values", action="store_true",
                                help="Flag sensitive values - hardcoded secrets, API keys, credentials")
    analysis_group.add_argument("-K", "--magic-constants", action="store_true",
                                help="Flag magic constants - repeated literal values across files")

    threshold_group = parser.add_argument_group("Threshold Options")
    for threshold_config in THRESHOLD_CONFIGS.values():
        add_threshold_argument(threshold_group, threshold_config, use_detailed_help=True)
    threshold_group.add_argument("--min-occurrences", type=int, default=TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
                                 help="(-T) Minimum function count for tramp data detection (default: 3)")
    threshold_group.add_argument("--min-group-size", type=int, default=TRAMP_DATA_MIN_GROUP_SIZE,
                                 help="(-T) Minimum parameters in a tramp data group (default: 2)")
    threshold_group.add_argument("--min-constant-occurrences", type=int,
                                 default=MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
                                 help="(-K) Minimum occurrences for magic constant detection (default: 3)")
    threshold_group.add_argument("--min-constant-files", type=int,
                                 default=MAGIC_CONSTANT_DEFAULT_MIN_FILES,
                                 help="(-K) Minimum distinct files for magic constant detection (default: 2)")
    threshold_group.add_argument("--constant-type", choices=["all", "numeric", "string"],
                                 default="all",
                                 help="(-K) Filter magic constants by type (default: all)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying files")
    parser.add_argument("--backup", action="store_true",
                        help="Create .coden-backup files before modifying")
    parser.add_argument("--remove-comments", action="store_true",
                        help="Delete detected echo comments entirely instead of flagging with \\[CODEN] markers (use with -E)")
    parser.add_argument("--remove-dead-code", action="store_true",
                        help="Delete dead code functions entirely instead of flagging with \\[CODEN] markers (DESTRUCTIVE - use with --backup)")
    parser.add_argument("--replace", nargs="?", const="***REDACTED***", default=None,
                        help="Replace detected sensitive values with placeholder (default: ***REDACTED***, or specify custom value)")
    parser.add_argument("--whitelist", nargs="*", default=None, metavar="PATTERN",
                        help="Scan text files matching glob patterns for secrets (e.g. '*.env' '*.json')")
    parser.add_argument("--include-tests", action="store_true",
                        help="Include test files in analysis. By default, test files are excluded to focus on production code")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("-f", "--format", default="tree",
                        choices=["tree", "json"],
                        help="Output format")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse output order (highest severity last)")
    parser.add_argument("--stats", action="store_true",
                        help="Show summary statistics")
    parser.add_argument("-n", "--limit", type=int, default=config.search.default_limit,
                        help="Limit results (default: 20, use -n -1 for all; dry-run preview only)")
    parser.add_argument("--no-daemon", dest="no_daemon", action="store_true",
                        help="Skip the daemon for this invocation; use the in-process direct path.")


def create_search_parser(config: AppConfig) -> argparse.ArgumentParser:
    """Create parser for search (default) mode."""
    parser = argparse.ArgumentParser(
        prog="coden",
        description="Coden - code search and context generation",
        formatter_class=DefaultValueHelpFormatter,
        epilog=_SEARCH_EPILOG,
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root directory")

    search = parser.add_argument_group("Search & Discovery")
    search.add_argument("-q", "--query", default="", help="Search query")
    search.add_argument("--map", action="store_true",
                        help="Generate context map (default when no query)")
    search.add_argument("--map-mode", dest="map_mode",
                        choices=["static", "simple"], default="static",
                        help="Ranking strategy for map output. "
                             "'static' (default): combined signal score. "
                             "'simple': per-object git commit count, linecount fallback. "
                             "Disk-cached; warm runs skip re-parsing. "
                             "Ignored when a query is supplied without --map.")
    search.add_argument("--find", metavar="IDENT", help="Find specific identifier")
    search.add_argument("-s", "--semantic", dest="enable_semantic", action="store_true",
                        help="Enable semantic search (bundled MiniLM ONNX)")
    search.add_argument("--no-daemon", dest="no_daemon", action="store_true",
                        help="Skip the daemon for this invocation; use the in-process direct path.")

    analysis = parser.add_argument_group("Code Analysis")
    analysis.add_argument("-H", "--hotspots", action="store_true",
                          help="Find refactoring hotspots (high coupling + complexity)")
    analysis.add_argument("-C", "--clones", action="store_true",
                          help="Detect code clones (combined semantic + syntactic by default)")
    analysis.add_argument("-P", "--propagation", action="store_true",
                          help="Analyze propagation cost (architecture coupling)")
    analysis.add_argument("-E", "--echo-comments", action="store_true",
                          help="Detect echo comments")
    analysis.add_argument("-D", "--dead-code", action="store_true",
                          help="Detect dead code - functions/methods with no callers")
    analysis.add_argument("-T", "--tramp-data", action="store_true",
                          help="Detect tramp data - parameters appearing across many functions")
    analysis.add_argument("-S", "--sensitive-values", action="store_true",
                          help="Detect sensitive values - hardcoded secrets, API keys, credentials")
    analysis.add_argument("-K", "--magic-constants", action="store_true",
                          help="Detect magic constants - repeated literal values across files")

    clones = parser.add_argument_group("Clone Options (use with -C)")
    mode_group = clones.add_mutually_exclusive_group()
    mode_group.add_argument("--clone-semantic", action="store_true",
                            help="Semantic only (MiniLM ONNX embeddings)")
    mode_group.add_argument("--clone-syntactic", action="store_true",
                            help="Syntactic only (line-by-line Jaccard)")
    clones.add_argument("--line-threshold", type=float, default=DEFAULT_SYNTACTIC_LINE_THRESHOLD,
                        help="Line similarity threshold (0.0-1.0)")
    clones.add_argument("--func-threshold", type=float, default=DEFAULT_SYNTACTIC_FUNC_THRESHOLD,
                        help="Function match threshold (0.0-1.0)")
    clones.add_argument("--semantic-weight", type=float, default=DEFAULT_CLONE_SEMANTIC_WEIGHT,
                        help="Semantic similarity weight in combined score (0.0-1.0)")
    clones.add_argument("--syntactic-weight", type=float, default=DEFAULT_CLONE_SYNTACTIC_WEIGHT,
                        help="Syntactic similarity weight in combined score (0.0-1.0)")

    prop = parser.add_argument_group("Propagation Options (use with -P)")
    prop.add_argument("--breakdown", action=argparse.BooleanOptionalAction, default=True,
                      help="Include per-module breakdown (default: on; pass --no-breakdown to disable)")
    prop.add_argument("--critical-paths", action="store_true",
                      help="Show most connected paths")
    prop.add_argument("--approximate", action="store_true",
                      help="Allow direct-edges approximation when the graph exceeds the closure "
                           "limit. Suppresses the PASS/WARNING/CRITICAL verdict and emits a loud "
                           "[APPROXIMATION] banner.")

    mod = parser.add_argument_group("Modification Options (use with -E/-S)")
    mod.add_argument("--remove-comments", action="store_true",
                     help="Delete detected echo comments (use with -E)")
    mod.add_argument("--replace", nargs="?", const="***REDACTED***", default=None,
                     help="Replace sensitive values (default: ***REDACTED***)")
    mod.add_argument("--whitelist", nargs="*", default=None, metavar="PATTERN",
                     help="Scan text files matching glob patterns for secrets (e.g. '*.env' '*.json')")
    mod.add_argument("--dry-run", action="store_true",
                     help="Preview changes without modifying files")
    mod.add_argument("--backup", action="store_true",
                     help="Create .coden-backup files before modifying")
    mod.add_argument("--include-tests", action="store_true",
                     help="Include test files in analysis")

    thresholds = parser.add_argument_group("Thresholds")
    for key in ["risk", "propagation", "clone", "echo", "dead_code", "sensitive_value"]:
        add_threshold_argument(thresholds, THRESHOLD_CONFIGS[key], use_detailed_help=False)
    thresholds.add_argument("--min-lines", type=_validate_min_lines, default=FLAG_MIN_LINES,
                            metavar="INT", help="(-C/-D) Skip functions shorter than N lines")
    thresholds.add_argument("--min-occurrences", type=int, default=TRAMP_DATA_DEFAULT_MIN_OCCURRENCES,
                            help="(-T) Minimum function count for tramp data detection")
    thresholds.add_argument("--min-group-size", type=int, default=TRAMP_DATA_MIN_GROUP_SIZE,
                            help="(-T) Minimum parameters in a tramp data group")
    thresholds.add_argument("--min-constant-occurrences", type=int,
                            default=MAGIC_CONSTANT_DEFAULT_MIN_OCCURRENCES,
                            help="(-K) Minimum occurrences for magic constant detection")
    thresholds.add_argument("--min-constant-files", type=int,
                            default=MAGIC_CONSTANT_DEFAULT_MIN_FILES,
                            help="(-K) Minimum distinct files for magic constant detection")
    thresholds.add_argument("--constant-type", choices=["all", "numeric", "string"],
                            default="all",
                            help="(-K) Filter magic constants by type (default: all)")

    output = parser.add_argument_group("Output")
    output.add_argument("-f", "--format", choices=["xml", "markdown", "tree", "json"],
                        default="tree", help="Output format")
    output.add_argument("-n", "--limit", type=int, default=config.search.default_limit,
                        help="Max results (use -n -1 for all)")
    output.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse result order (highest score last)")
    output.add_argument("--stats", action="store_true", help="Print ranking statistics")
    output.add_argument("--dir-tree", action=argparse.BooleanOptionalAction,
                        default=True, help="Show directory tree")
    output.add_argument("--show-deps", action="store_true", help="Include dependency context")
    output.add_argument("--tokens", type=int, default=None,
                        help="Token budget (only -n/--limit controls result count)")

    general = parser.add_argument_group("General")
    general.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser
