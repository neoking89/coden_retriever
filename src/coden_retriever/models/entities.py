"""
Code entity models.

Contains data structures representing code entities and their dependencies.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..token_estimator import count_tokens

# Pre-compiled pattern for CamelCase to words conversion (e.g., "ProjectCache" -> "Project Cache")
_CAMEL_CASE_PATTERN = re.compile(r'([a-z])([A-Z])')


@dataclass
class DependencyContext:
    """Dependency information for an entity."""
    callers: list[tuple[str, str, str, float]] = field(default_factory=list)  # (node_id, name, type, weight)
    callees: list[tuple[str, str, str, float]] = field(default_factory=list)  # (node_id, name, type, weight)

    def format_compact(self, max_items: int = 3) -> str:
        """Format dependencies compactly for inclusion in output."""
        parts = []

        if self.callers:
            caller_strs = []
            for _, name, etype, _ in self.callers[:max_items]:
                caller_strs.append(f"{name}")
            if len(self.callers) > max_items:
                caller_strs.append(f"+{len(self.callers) - max_items} more")
            parts.append(f"<- called by: {', '.join(caller_strs)}")

        if self.callees:
            callee_strs = []
            for _, name, etype, _ in self.callees[:max_items]:
                callee_strs.append(f"{name}")
            if len(self.callees) > max_items:
                callee_strs.append(f"+{len(self.callees) - max_items} more")
            parts.append(f"-> calls: {', '.join(callee_strs)}")

        return " | ".join(parts) if parts else ""

    def is_empty(self) -> bool:
        """Check if there are no dependencies."""
        return not self.callers and not self.callees


@dataclass
class PathTraceResult:
    """Result of tracing execution or dependency paths between symbols."""
    source: str
    target: str | None
    direction: str
    paths: list[list[tuple[str, str, str]]] = field(default_factory=list)  # List of paths, each path is [(node_id, name, type), ...]
    reachable_nodes: list[tuple[str, str, str]] = field(default_factory=list)  # [(node_id, name, type), ...]
    max_depth_reached: int = 0
    total_affected: int = 0
    paths_found: int = 0
    requested_max_depth: int = 5  # The depth requested by the user
    requested_path_limit: int = 10  # The number of paths requested by the user

    def _extract_file_path(self, node_id: str) -> str:
        """Extract file path from node_id format 'file_path::name::line'."""
        parts = node_id.split("::")
        if parts:
            return parts[0]
        return "unknown"

    def _get_relationship_type(self, from_type: str, to_type: str) -> str:
        """Determine relationship type based on entity types."""
        if from_type == "function" and to_type == "class":
            return "instantiates"
        elif from_type == "method" and to_type == "method":
            return "calls"
        else:
            return "calls"

    def _deduplicate_paths(self) -> list[list[tuple[str, str, str]]]:
        """Remove duplicate paths."""
        seen = set()
        unique_paths = []
        for path in self.paths:
            # Create a hashable representation of the path
            path_key = tuple((name, etype) for _, name, etype in path)
            if path_key not in seen:
                seen.add(path_key)
                unique_paths.append(path)
        return unique_paths

    def _deduplicate_nodes(self) -> list[tuple[str, str, str]]:
        """Remove duplicate reachable nodes."""
        seen = set()
        unique_nodes = []
        for node in self.reachable_nodes:
            node_id, name, etype = node
            node_key = (name, etype)
            if node_key not in seen:
                seen.add(node_key)
                unique_nodes.append(node)
        return unique_nodes

    def _count_node_frequency(self, unique_paths: list[list[tuple[str, str, str]]]) -> dict[tuple[str, str], int]:
        """Count how many times each node appears in paths."""
        frequency: dict[tuple[str, str], int] = {}
        for path in unique_paths:
            for _, name, etype in path:
                key = (name, etype)
                frequency[key] = frequency.get(key, 0) + 1
        return frequency

    def _group_nodes_by_depth(self, unique_paths: list[list[tuple[str, str, str]]]) -> dict:
        """Group nodes by their depth in the dependency tree."""
        depth_map = {}
        for path in unique_paths:
            for depth, (node_id, name, etype) in enumerate(path):
                key = (name, etype)
                if key not in depth_map:
                    depth_map[key] = depth
                else:
                    # Keep the minimum depth where this node appears
                    depth_map[key] = min(depth_map[key], depth)
        return depth_map

    def format_enhanced_output(self, root_directory: str = "") -> dict:
        """Format the complete enhanced output for LLM consumption."""
        unique_paths = self._deduplicate_paths()
        unique_nodes = self._deduplicate_nodes()
        frequency = self._count_node_frequency(unique_paths)
        depth_map = self._group_nodes_by_depth(unique_paths)

        # Build dependency paths with file locations
        paths_lines = []
        for i, path in enumerate(unique_paths, 1):
            paths_lines.append(f"\nPath {i} (depth: {len(path) - 1}):")
            for depth, (node_id, name, etype) in enumerate(path):
                file_path = self._extract_file_path(node_id)
                indent = "  " * (depth + 1)
                rel_type = ""
                if depth > 0:
                    prev_type = path[depth - 1][2]
                    rel_type = f" [{self._get_relationship_type(prev_type, etype)}]"

                paths_lines.append(f"{indent}{name} ({etype}, {file_path}){rel_type}")

        dependency_paths = "\n".join(paths_lines) if paths_lines else "No paths found."

        # Group nodes by depth with frequency
        depth_groups: dict[int, list[tuple[str, str, str, int]]] = {}
        for node_id, name, etype in unique_nodes:
            key = (name, etype)
            node_depth = depth_map.get(key, 0)
            if node_depth not in depth_groups:
                depth_groups[node_depth] = []
            file_path = self._extract_file_path(node_id)
            freq = frequency.get(key, 1)
            depth_groups[node_depth].append((name, etype, file_path, freq))

        # Format nodes by depth
        nodes_by_depth_lines = []
        for depth in sorted(depth_groups.keys()):
            nodes = depth_groups[depth]
            depth_label = "Depth 1 (direct dependencies)" if depth == 1 else f"Depth {depth}"
            nodes_by_depth_lines.append(f"\n{depth_label}:")
            for name, etype, file_path, freq in nodes:
                freq_str = f" - appears in {freq} path{'s' if freq > 1 else ''}" if freq > 1 else ""
                nodes_by_depth_lines.append(f"  * {name} ({etype}, {file_path}){freq_str}")

        reachable_by_depth = "\n".join(nodes_by_depth_lines) if nodes_by_depth_lines else "No reachable nodes."

        # Format all reachable nodes alphabetically
        classes = []
        functions = []
        for node_id, name, etype in sorted(unique_nodes, key=lambda x: x[1].lower()):
            file_path = self._extract_file_path(node_id)
            if etype == "class":
                classes.append(f"  * {name} - {file_path}")
            else:
                functions.append(f"  * {name} - {file_path}")

        all_nodes_lines = []
        all_nodes_lines.append(f"{len(unique_nodes)} unique nodes reachable from '{self.source}' (duplicates removed):\n")
        if classes:
            all_nodes_lines.append(f"Classes ({len(classes)}):")
            all_nodes_lines.extend(classes)
            all_nodes_lines.append("")
        if functions:
            all_nodes_lines.append(f"Functions ({len(functions)}):")
            all_nodes_lines.extend(functions)

        all_reachable_alphabetical = "\n".join(all_nodes_lines) if all_nodes_lines else "No reachable nodes."

        # Identify critical path nodes (appear in 3+ paths)
        critical_nodes = [(name, etype) for (name, etype), count in frequency.items() if count >= 3]
        critical_names = [name for name, _ in critical_nodes]

        # Identify branching points (nodes with multiple children in paths)
        branching_nodes: list[tuple[str, str]] = []
        for path in unique_paths:
            for i, (_, name, etype) in enumerate(path[:-1]):  # Exclude leaf nodes
                # Count how many different successors this node has across all paths
                successors = set()
                for p in unique_paths:
                    for j, (_, n, e) in enumerate(p[:-1]):
                        if n == name and e == etype and j + 1 < len(p):
                            successors.add(p[j + 1][1])  # Add successor name
                if len(successors) > 1 and name not in [n for n, _ in branching_nodes]:
                    branching_nodes.append((name, etype))

        branching_names = [name for name, _ in branching_nodes]

        # Identify leaf nodes (appear only at path ends)
        leaf_nodes = set()
        for path in unique_paths:
            if path:
                leaf_nodes.add((path[-1][1], path[-1][2]))
        leaf_names = [name for name, _ in leaf_nodes]

        # Build summary based on direction and target
        direction_text = {
            "downstream": "Downstream",
            "upstream": "Upstream",
            "both": "Bidirectional"
        }.get(self.direction.lower(), "Downstream")

        if self.target:
            summary = f"{direction_text} dependency analysis from '{self.source}' to '{self.target}'"
        else:
            summary = f"{direction_text} dependency analysis from '{self.source}' function"

        query_params = {
            "start_identifier": self.source,
            "direction": self.direction,
            "requested_max_depth": self.requested_max_depth,
            "requested_path_limit": self.requested_path_limit
        }
        if self.target:
            query_params["end_identifier"] = self.target

        return {
            "summary": summary,
            "query_parameters": query_params,
            "overview": {
                "unique_paths_found": len(unique_paths),
                "unique_reachable_nodes": len(unique_nodes),
                "actual_max_depth": self.max_depth_reached,
                "note": f"Actual max depth ({self.max_depth_reached}) {'exceeds' if self.max_depth_reached > self.requested_max_depth else 'is within'} requested limit ({self.requested_max_depth})" + (" - paths extended to completion" if self.max_depth_reached > self.requested_max_depth else "")
            },
            "dependency_paths": dependency_paths,
            "reachable_nodes_by_depth": reachable_by_depth,
            "all_reachable_nodes_alphabetical": all_reachable_alphabetical,
            "impact_analysis": {
                "critical_path_nodes": critical_names,
                "note": "These nodes appear in 3+ paths - changes here have highest downstream impact" if critical_names else "No critical path nodes identified",
                "leaf_nodes": leaf_names,
                "branching_points": branching_names,
                "branching_note": "These nodes are where execution branches to multiple destinations" if branching_names else "No branching points identified"
            },
            "stats": {
                "paths_returned": len(unique_paths),
                "paths_requested": self.requested_path_limit,
                "unique_nodes_reached": len(unique_nodes),
                "total_node_occurrences": self.total_affected,
                "max_depth_reached": self.max_depth_reached,
                "max_depth_requested": self.requested_max_depth
            }
        }

    def format_summary(self) -> str:
        """Format a summary of the trace result."""
        parts = []
        parts.append(f"Source: {self.source}")
        if self.target:
            parts.append(f"Target: {self.target}")
        parts.append(f"Direction: {self.direction}")
        parts.append(f"Paths found: {self.paths_found}")
        parts.append(f"Total affected nodes: {self.total_affected}")
        parts.append(f"Max depth reached: {self.max_depth_reached}")
        return " | ".join(parts)

    def format_paths(self, max_paths: int = 5) -> str:
        """Format paths in a readable way."""
        if not self.paths:
            return "No paths found."

        lines = []
        for i, path in enumerate(self.paths[:max_paths]):
            path_str = " -> ".join([f"{name}({etype})" for _, name, etype in path])
            lines.append(f"Path {i+1}: {path_str}")

        if len(self.paths) > max_paths:
            lines.append(f"... and {len(self.paths) - max_paths} more paths")

        return "\n".join(lines)

    def format_reachable(self) -> str:
        """Format reachable nodes in a readable way."""
        if not self.reachable_nodes:
            return "No reachable nodes found."

        lines = [f"Reachable from {self.source}:"]
        for _, name, etype in self.reachable_nodes:
            lines.append(f"  * {name} ({etype})")

        return "\n".join(lines)


@dataclass
class CodeEntity:
    """Represents a code entity (class, function, method) in the repository."""

    name: str
    entity_type: str
    file_path: str
    language: str
    line_start: int
    line_end: int
    source_code: str
    body_start: int | None = None
    body_end: int | None = None
    docstring: str | None = None
    parent_class: str | None = None
    cyclomatic_complexity: int | None = None
    is_stub: bool = False
    is_decorated: bool = False  # True if function/method has decorators/annotations (AST-detected)

    @property
    def node_id(self) -> str:
        """Unique identifier for this entity in the graph."""
        return f"{self.file_path}::{self.name}::{self.line_start}"

    @property
    def qualified_name(self) -> str:
        """Human-readable qualified name."""
        rel_path = Path(self.file_path).name
        if self.parent_class:
            return f"{rel_path}::{self.parent_class}.{self.name}"
        return f"{rel_path}::{self.name}"

    @property
    def is_private(self) -> bool:
        """Determine if this entity is private based on language conventions."""
        if self.language == "go":
            return len(self.name) > 0 and self.name[0].islower()
        elif self.language in ("javascript", "typescript"):
            return self.name.startswith("_") or self.name.startswith("#")
        elif self.language == "python":
            return self.name.startswith("_") and not self.name.startswith("__")
        elif self.language in ("java", "c_sharp", "kotlin"):
            return self.name.startswith("_")
        else:
            return self.name.startswith("_")

    @property
    def is_test(self) -> bool:
        """Check if this is a test function/file.

        Uses only reliable name-based patterns to avoid false positives from
        absolute paths containing 'test' (e.g., /users/bob/test/myproject/).
        """
        name_lower = self.name.lower()

        # Check function/class name patterns (e.g., test_foo, FooTest)
        if (name_lower.startswith("test") or
            name_lower.endswith("test") or
            name_lower.startswith("spec") or
            name_lower.endswith("spec")):
            return True

        # Check filename patterns (e.g., test_foo.py, foo_test.py)
        filename = Path(self.file_path).name.lower()
        if (filename.startswith("test_") or
            filename.endswith("_test.py") or
            filename.endswith("_spec.py") or
            filename.startswith("spec_")):
            return True

        return False

    @property
    def line_count(self) -> int:
        """Number of lines in this entity."""
        return max(1, self.line_end - self.line_start + 1)

    @property
    def token_count(self) -> int:
        """Accurate token count for this entity. Cached for performance."""
        cache = self.__dict__.get('_token_count_cache')
        if cache is not None:
            return cache

        result = count_tokens(self.source_code, is_code=True)
        object.__setattr__(self, '_token_count_cache', result)
        return result

    @property
    def is_tiny(self) -> bool:
        """Check if this is a trivially small function."""
        return self.line_count < Config.TINY_FUNCTION_LINES

    @property
    def is_utility(self) -> bool:
        """Check if this is a known utility function."""
        return self.name.lower() in Config.UTILITY_NAMES

    @property
    def searchable_text(self) -> str:
        """Text corpus for BM25 indexing. Cached for performance."""
        # Use object's __dict__ directly to cache without triggering frozen dataclass issues
        cache = self.__dict__.get('_searchable_text_cache')
        if cache is not None:
            return cache

        name_expanded = _CAMEL_CASE_PATTERN.sub(r"\1 \2", self.name)
        name_expanded = name_expanded.replace("_", " ")

        # Boost name by repeating it - helps entities with matching names rank higher
        boost = Config.BM25_NAME_BOOST
        name_boost = f"{self.name} " * boost + f"{name_expanded} " * boost

        parts = [
            name_boost,
            self.entity_type,
            self.docstring or "",
            self.source_code,
        ]
        result = " ".join(parts)
        object.__setattr__(self, '_searchable_text_cache', result)
        return result

    @property
    def semantic_searchable_text(self) -> str:
        """Text optimized for semantic search using tree-sitter AST.

        Returns expanded name + signature + bounded body content (first N lines)
        to improve semantic matching on implementation-specific keywords.

        Note: Docstrings were tested (Iteration 3) but caused regressions due to
        semantic noise from verbose descriptions matching unrelated queries.
        """
        cache = self.__dict__.get('_semantic_searchable_text_cache')
        if cache is not None:
            return cache

        # Expand name for natural language matching (CamelCase/snake_case -> words)
        name_expanded = _CAMEL_CASE_PATTERN.sub(r'\1 \2', self.name)
        name_expanded = name_expanded.replace('_', ' ').lower()

        sig = self.signature
        body_lines = self._get_bounded_body_lines(max_lines=5)
        result = f"{self.name} {name_expanded}\n{sig}"
        if body_lines:
            result += "\n" + body_lines
        object.__setattr__(self, '_semantic_searchable_text_cache', result)
        return result

    def _get_bounded_body_lines(self, max_lines: int = 5) -> str:
        """Extract first N lines of body content after the signature."""
        if self.body_start is None:
            return ""
        lines = self.source_code.splitlines()
        if not lines:
            return ""
        body_line_index = self.body_start - self.line_start
        if body_line_index < 0 or body_line_index >= len(lines):
            return ""
        body_end_index = min(body_line_index + max_lines, len(lines))
        return "\n".join(lines[body_line_index:body_end_index]).strip()

    @property
    def signature(self) -> str:
        """Extracts the signature/header using tree-sitter body_start (language agnostic).

        Uses body_start from tree-sitter AST to determine where the signature ends.
        Falls back to first few lines if body_start is not available.
        """
        lines = self.source_code.splitlines()
        if not lines:
            return self.name

        if self.body_start is not None:
            # Use tree-sitter's body_start: signature is from line_start to body_start
            # body_start is 1-indexed line number where body begins
            sig_line_count = self.body_start - self.line_start
            if sig_line_count > 0:
                return '\n'.join(lines[:sig_line_count]).strip()

        # Fallback: use first few lines (for entities without body_start)
        max_lines = min(5, len(lines))
        return '\n'.join(lines[:max_lines]).strip()

    def get_context_snippet(self, max_lines: int = Config.MAX_BODY_LINES) -> str:
        """Get a meaningful code snippet with smart truncation."""
        lines = self.source_code.splitlines()

        if len(lines) <= max_lines:
            return self.source_code

        sig_end = 1
        for i, line in enumerate(lines):
            if ':' in line or '{' in line:
                sig_end = i + 1
                break

        body_lines_available = max_lines - sig_end - 1

        if body_lines_available <= 0:
            return '\n'.join(lines[:sig_end]) + '\n    # ...'

        result_lines = lines[:sig_end + body_lines_available]
        remaining = len(lines) - len(result_lines)

        if remaining > 0:
            result_lines.append(f"    # ... ({remaining} more lines)")

        return '\n'.join(result_lines)
