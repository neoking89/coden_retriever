"""Port for git blame/history subprocess calls used by GitHistoryContextService.

The Protocol decouples GitHistoryContextService from subprocess so the service
can be tested with a fake source. SubprocessGitBlameSource is the production
adapter that wraps run_git_command.

Each method returns (returncode, stdout, stderr); returncode -1 signals
subprocess failure (timeout, missing git, etc.) — same contract as
run_git_command.
"""

from __future__ import annotations

from typing import Protocol

from coden_retriever.git.commands import run_git_command


class GitBlameSource(Protocol):
    """Subprocess interface required by GitHistoryContextService."""

    async def rev_parse_git_dir(self, cwd: str) -> tuple[int, str, str]:
        ...

    async def blame_porcelain(
        self,
        cwd: str,
        file_path: str,
        start_line: int,
        end_line: int,
        follow_renames: bool,
    ) -> tuple[int, str, str]:
        ...

    async def show_commit_message(
        self, cwd: str, commit_hash: str
    ) -> tuple[int, str, str]:
        ...

    async def show_commit_diff(
        self, cwd: str, commit_hash: str, file_path: str
    ) -> tuple[int, str, str]:
        ...

    async def log_renames(self, cwd: str, file_path: str) -> tuple[int, str, str]:
        ...


class SubprocessGitBlameSource:
    """Production adapter that delegates to run_git_command."""

    async def rev_parse_git_dir(self, cwd: str) -> tuple[int, str, str]:
        return await run_git_command(["rev-parse", "--git-dir"], cwd)

    async def blame_porcelain(
        self,
        cwd: str,
        file_path: str,
        start_line: int,
        end_line: int,
        follow_renames: bool,
    ) -> tuple[int, str, str]:
        args = ["blame", "-L", f"{start_line},{end_line}", "--porcelain"]
        if follow_renames:
            # -C -C -C: detect copies/renames across files, even in different commits
            args.extend(["-C", "-C", "-C"])
        args.append(file_path)
        return await run_git_command(args, cwd)

    async def show_commit_message(
        self, cwd: str, commit_hash: str
    ) -> tuple[int, str, str]:
        return await run_git_command(["show", "-s", "--format=%B", commit_hash], cwd)

    async def show_commit_diff(
        self, cwd: str, commit_hash: str, file_path: str
    ) -> tuple[int, str, str]:
        return await run_git_command(
            ["show", commit_hash, "--format=", "-p", "--", file_path],
            cwd,
        )

    async def log_renames(self, cwd: str, file_path: str) -> tuple[int, str, str]:
        return await run_git_command(
            [
                "log",
                "--follow",
                "--name-status",
                "--format=%H",
                "--diff-filter=R",
                "--",
                file_path,
            ],
            cwd,
        )
