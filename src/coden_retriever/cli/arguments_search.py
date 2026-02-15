"""Argument parser creation for search and flag-arguments."""
import argparse

from ..config_loader import AppConfig
from .threshold_config import THRESHOLD_CONFIGS, add_threshold_argument
from .utils import DefaultValueHelpFormatter

# Validation floor for --min-lines argument
_MIN_LINES_FLOOR = 1

# Line-level similarity threshold for syntactic clone detection
# Selected to balance precision (avoid false positives) with recall (find actual clones)
_DEFAULT_LINE_THRESHOLD = 0.70

# Function-level match threshold - requires at least 50% of lines to match
# Lower than line threshold to account for structural variations in similar functions
_DEFAULT_FUNC_THRESHOLD = 0.50

# Semantic similarity weight in combined clone detection
# Weighted higher (0.65) because semantic similarity is more reliable than syntax
_DEFAULT_SEMANTIC_WEIGHT = 0.65

# Syntactic similarity weight in combined clone detection
# Complementary weight (0.35) to semantic, ensuring weights sum to 1.0
_DEFAULT_SYNTACTIC_WEIGHT = 0.35

# Minimum function lines to consider for analysis
# Filters out trivial functions (getters, simple returns, one-liners)
_DEFAULT_MIN_LINES = 3

# Minimum function occurrences for tramp data detection
# At least 3 functions needed to identify a meaningful data flow pattern
_DEFAULT_MIN_OCCURRENCES = 3

# Minimum parameters in a tramp data group
# At least 2 parameters traveling together required to form a cohesive group
_DEFAULT_MIN_GROUP_SIZE = 2

_SEARCH_EPILOG = """\
[bold cyan]Start by running coden on your repo to see what matters:[/] 

  coden src                              ranked map of src/
  coden src --stats -n 50 -r             top 50, reversed, with ranking stats

[bold green]Looking for something specific? Search by keyword, or go semantic:[/] 

  coden src -q "auth"                    keyword search for "auth"
  coden src -q "how does login work" -s  semantic search (needs \\[semantic] extra)
  coden --find UserAuth --show-deps      find a symbol and its callers/callees

[bold yellow]Now dig deeper. Run analysis to find what needs attention:[/] 

  coden src -H --stats -r                refactoring hotspots (coupling x complexity)
  coden src -C --clone-threshold 0.90    code clones above 90% similarity
  coden src -P --breakdown               architecture health per module
  coden src -D                           dead code - functions nobody calls
  coden src -T --min-occurrences 5       tramp data - params traveling together
  coden src -S                           hardcoded secrets and API keys

[bold red]Found issues? Flag them in your source code, or fix them directly:[/] 

  coden flag -HPCEDTS --dry-run          preview all flags before writing
  coden flag -HPCEDTS --backup           write \\[CODEN] comments (with backup)
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

[bold dim]Starting fresh? Reset clears caches, stops the daemon, and resets config:[/]

  coden reset                            full reset (destructive!)

[dim]Subcommands: serve, agent (-a), daemon, cache, config, flag, reset[/]
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
                                help="Flag code clones. Requires \\[semantic] extra for semantic/combined modes")
    analysis_group.add_argument("--clone-semantic", action="store_true",
                                help="Clone detection: semantic only (Model2Vec embeddings)")
    analysis_group.add_argument("--clone-syntactic", action="store_true",
                                help="Clone detection: syntactic only (line-by-line Jaccard)")
    analysis_group.add_argument("--line-threshold", type=float, default=_DEFAULT_LINE_THRESHOLD,
                                help="Line similarity threshold for syntactic clones")
    analysis_group.add_argument("--func-threshold", type=float, default=_DEFAULT_FUNC_THRESHOLD,
                                help="Function match threshold for syntactic clones")
    analysis_group.add_argument("--semantic-weight", type=float, default=_DEFAULT_SEMANTIC_WEIGHT,
                                help="Weight for semantic similarity in combined score")
    analysis_group.add_argument("--syntactic-weight", type=float, default=_DEFAULT_SYNTACTIC_WEIGHT,
                                help="Weight for syntactic similarity in combined score")
    analysis_group.add_argument("-E", "--echo-comments", action="store_true",
                                help="Detect and flag echo comments. Requires \\[semantic] extra")
    analysis_group.add_argument("-D", "--dead-code", action="store_true",
                                help="Flag dead code - functions/methods with no callers in the codebase")
    analysis_group.add_argument("-T", "--tramp-data", action="store_true",
                                help="Flag tramp data - parameters appearing across many functions")
    analysis_group.add_argument("-S", "--sensitive-values", action="store_true",
                                help="Flag sensitive values - hardcoded secrets, API keys, credentials")

    threshold_group = parser.add_argument_group("Threshold Options")
    for threshold_config in THRESHOLD_CONFIGS.values():
        add_threshold_argument(threshold_group, threshold_config, use_detailed_help=True)
    threshold_group.add_argument("--min-occurrences", type=int, default=_DEFAULT_MIN_OCCURRENCES,
                                 help="(-T) Minimum function count for tramp data detection (default: 3)")
    threshold_group.add_argument("--min-group-size", type=int, default=_DEFAULT_MIN_GROUP_SIZE,
                                 help="(-T) Minimum parameters in a tramp data group (default: 2)")

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
    search.add_argument("--find", metavar="IDENT", help="Find specific identifier")
    search.add_argument("-s", "--semantic", dest="enable_semantic", action="store_true",
                        help="Enable semantic search (Model2Vec). Requires \\[semantic] extra")

    analysis = parser.add_argument_group("Code Analysis")
    analysis.add_argument("-H", "--hotspots", action="store_true",
                          help="Find refactoring hotspots (high coupling + complexity)")
    analysis.add_argument("-C", "--clones", action="store_true",
                          help="Detect code clones. Requires \\[semantic] extra for semantic/combined")
    analysis.add_argument("-P", "--propagation", action="store_true",
                          help="Analyze propagation cost (architecture coupling)")
    analysis.add_argument("-E", "--echo-comments", action="store_true",
                          help="Detect echo comments. Requires \\[semantic] extra")
    analysis.add_argument("-D", "--dead-code", action="store_true",
                          help="Detect dead code - functions/methods with no callers")
    analysis.add_argument("-T", "--tramp-data", action="store_true",
                          help="Detect tramp data - parameters appearing across many functions")
    analysis.add_argument("-S", "--sensitive-values", action="store_true",
                          help="Detect sensitive values - hardcoded secrets, API keys, credentials")

    clones = parser.add_argument_group("Clone Options (use with -C)")
    clones.add_argument("--clone-semantic", action="store_true",
                        help="Semantic only (Model2Vec embeddings)")
    clones.add_argument("--clone-syntactic", action="store_true",
                        help="Syntactic only (line-by-line Jaccard)")
    clones.add_argument("--line-threshold", type=float, default=_DEFAULT_LINE_THRESHOLD,
                        help="Line similarity threshold (0.0-1.0)")
    clones.add_argument("--func-threshold", type=float, default=_DEFAULT_FUNC_THRESHOLD,
                        help="Function match threshold (0.0-1.0)")
    clones.add_argument("--semantic-weight", type=float, default=_DEFAULT_SEMANTIC_WEIGHT,
                        help="Semantic similarity weight in combined score (0.0-1.0)")
    clones.add_argument("--syntactic-weight", type=float, default=_DEFAULT_SYNTACTIC_WEIGHT,
                        help="Syntactic similarity weight in combined score (0.0-1.0)")

    prop = parser.add_argument_group("Propagation Options (use with -P)")
    prop.add_argument("--breakdown", action="store_true",
                      help="Include per-module breakdown")
    prop.add_argument("--critical-paths", action="store_true",
                      help="Show most connected paths")

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
    thresholds.add_argument("--min-lines", type=_validate_min_lines, default=_DEFAULT_MIN_LINES,
                            metavar="INT", help="(-C/-D) Skip functions shorter than N lines")
    thresholds.add_argument("--min-occurrences", type=int, default=_DEFAULT_MIN_OCCURRENCES,
                            help="(-T) Minimum function count for tramp data detection")
    thresholds.add_argument("--min-group-size", type=int, default=_DEFAULT_MIN_GROUP_SIZE,
                            help="(-T) Minimum parameters in a tramp data group")

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
