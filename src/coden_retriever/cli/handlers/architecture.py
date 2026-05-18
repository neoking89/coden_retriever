"""CLI handler for `coden architecture`: orchestrates parse → audit → render → exit."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from ...architecture import render_json, render_text, run_audit
from ..arguments_architecture import create_architecture_parser

logger = logging.getLogger(__name__)


def handle_architecture_command(argv: list[str]) -> int:
    """Entry point reached from `__main__.py` when `cmd == "architecture"`."""
    parser = create_architecture_parser()
    args = parser.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"architecture: path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"architecture: path is not a directory: {root}", file=sys.stderr)
        return 2

    excludes = tuple(part.strip() for part in args.exclude.split(",") if part.strip())
    report, err = run_audit(root=root, lang=args.lang, top=args.top, excludes=excludes)

    if err is not None:
        print(err, file=sys.stderr)
        return 0

    assert report is not None
    if args.as_json:
        print(render_json(report))
    else:
        print(render_text(report))

    return 1 if report.cycles else 0
