#!/usr/bin/env python3
"""End-to-end automated release script for coden-retriever.

Drives the full release from local source to live PyPI + GitHub release:

  0.  pre_release_sync.py        copy private repo changes into public
  1.  create release branch      release/vX.Y.Z
  2.  pytest                     SKIPPED by default (tests live in private repo);
                                 use --run-tests to opt in (pytest runs in PRIVATE_REPO)
  3.  bump pyproject.toml        version line in [project]
  4.  commit                     "Update version to X.Y.Z"
  5.  tag                        vX.Y.Z annotated
  6.  push                       branch + tags
  7.  build                      wheel + sdist via `python -m build`
  8.  twine check + upload       PyPI (point of no return)
  9.  poll PyPI                  JSON API (fast) + /simple/ index (what pip uses)
 10.  fresh-venv smoke           install + version + import + CLI in a temp venv
 11.  merge → main + delete      fast-forward, push, drop the release branch
 12.  GitHub release             gh release create with wheel + sdist + notes
 13.  final verification         re-check PyPI JSON API

Usage:
    python release.py [--yes] [--dry-run]
                      [--skip-sync] [--run-tests] [--skip-docker]
                      [--skip-smoke] [--skip-merge] [--skip-gh-release]
                      [--notes-file PATH] [--force]

Flags:
    --yes              Auto-confirm every prompt with its default (default = Y).
                       Removes the stdin-pipe foot-gun that broke 2.1.0.
    --dry-run          Skip all side-effecting commands.
    --skip-sync        Don't run pre_release_sync.py at the start.
    --skip-tests       Don't run pytest. ON by default — tests are private-repo only.
    --run-tests        Opt-in: run pytest from PRIVATE_REPO. Overrides --skip-tests.
    --skip-docker      Don't run Docker tests.
    --skip-smoke       Don't run the post-upload fresh-venv smoke test.
    --skip-merge       Leave release branch in place; don't merge to main.
    --skip-gh-release  Don't create the GitHub release.
    --notes-file PATH  Use these release notes instead of auto-generating from git log.
    --force            Skip the preflight check that PyPI doesn't already have this
                       version. Use only when retrying after a partial failure.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv as venv_module
from pathlib import Path


PACKAGE = "coden-retriever"
REPO_SLUG = "neoking89/coden_retriever"

AUTO_YES = False
DRY_RUN = False


def load_env(env_path: Path) -> dict[str, str]:
    """Load environment variables from .env file."""
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        print("Create one with CODEN_RETRIEVER_VERSION and PYPI_API_TOKEN")
        sys.exit(1)

    env_vars: dict[str, str] = {}
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


def run_command(
    cmd: str | list[str],
    check: bool = True,
    capture_output: bool = False,
    env: dict | None = None,
    input_text: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess | None:
    """Run a shell command. Honors module-level DRY_RUN."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    cwd_note = f"  (cwd={cwd})" if cwd else ""
    print(f"\n>>> Running: {cmd_str}{cwd_note}")

    if DRY_RUN:
        print("    [DRY-RUN] Skipping actual execution")
        return None

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        capture_output=capture_output,
        text=True,
        env=merged_env,
        input=input_text,
        cwd=str(cwd) if cwd else None,
    )


def confirm(message: str, default: bool = True) -> bool:
    """Ask user for confirmation. In AUTO_YES mode, returns default without reading stdin.

    This is the single source of truth for prompts. The 2.1.0 release broke when a
    piped stdin was consumed by a child subprocess (pytest -vvv) and the next
    input() call hit EOFError. AUTO_YES sidesteps that entirely.
    """
    if AUTO_YES:
        ans = "Y" if default else "N"
        print(f"\n{message} [{'Y/n' if default else 'y/N'}]: {ans} (auto)")
        return default

    suffix = " [Y/n]: " if default else " [y/N]: "
    response = input(f"\n{message}{suffix}").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


def update_pyproject_version(pyproject_path: Path, new_version: str) -> None:
    """Rewrite the [project] version line in pyproject.toml."""
    print(f"\n>>> Updating version in pyproject.toml to {new_version}")
    if DRY_RUN:
        print("    [DRY-RUN] Skipping actual update")
        return

    content = pyproject_path.read_text()
    updated = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    pyproject_path.write_text(updated)
    print(f"    Updated version to {new_version}")


def clean_dist(dist_path: Path) -> None:
    print("\n>>> Cleaning dist directory")
    if DRY_RUN:
        print("    [DRY-RUN] Skipping cleanup")
        return
    if dist_path.exists():
        import shutil
        shutil.rmtree(dist_path)
        print(f"    Removed {dist_path}")
    else:
        print(f"    {dist_path} does not exist, nothing to clean")


def check_pypi_version_exists(version: str) -> bool:
    """Return True if PACKAGE==version is already on PyPI."""
    url = f"https://pypi.org/pypi/{PACKAGE}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def poll_pypi_json(version: str, max_wait: int = 180) -> dict:
    """Block until PyPI's JSON API has the new version. Returns the metadata."""
    url = f"https://pypi.org/pypi/{PACKAGE}/{version}/json"
    print(f"\n>>> Polling JSON API: {url} (max {max_wait}s)")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    elapsed = int(time.time() - start)
                    print(f"    JSON API indexed after {elapsed}s")
                    return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        time.sleep(5)
    raise TimeoutError(f"PyPI JSON API did not index {PACKAGE}=={version} within {max_wait}s")


def poll_pypi_simple_index(version: str, max_wait: int = 240) -> None:
    """Block until PyPI's /simple/ index lists the new wheel.

    PyPI's JSON API updates within ~30-45 s of upload, but the /simple/ HTML
    index (which `pip install` resolves against) lags an additional 1-2 minutes
    on average. Smoke tests using `pip install` need this to be ready, not just
    the JSON API.
    """
    url = f"https://pypi.org/simple/{PACKAGE}/"
    needle = f"{PACKAGE.replace('-', '_')}-{version}-"
    print(f">>> Polling simple index: {url} for {needle}* (max {max_wait}s)")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=10) as r:
                html = r.read().decode("utf-8", errors="ignore")
                if needle in html:
                    elapsed = int(time.time() - start)
                    print(f"    Simple index updated after {elapsed}s")
                    return
        except urllib.error.URLError:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"PyPI simple index did not list {PACKAGE}=={version} within {max_wait}s "
        f"(JSON was indexed, but pip can't resolve it yet)"
    )


def fresh_venv_smoke(version: str) -> None:
    """Install PACKAGE in a fresh temp venv and run a minimal smoke test.

    Verifies the published wheel resolves, declares the right version, imports
    the new architecture surface, and the `coden` CLI launches.
    """
    with tempfile.TemporaryDirectory(prefix=f"{PACKAGE}_smoke_") as tmp:
        venv_dir = Path(tmp) / "venv"
        print(f"\n>>> Creating fresh venv at {venv_dir}")
        venv_module.create(venv_dir, with_pip=True)

        if sys.platform == "win32":
            py = venv_dir / "Scripts" / "python.exe"
            cli = venv_dir / "Scripts" / "coden.exe"
        else:
            py = venv_dir / "bin" / "python"
            cli = venv_dir / "bin" / "coden"

        print(f">>> pip install {PACKAGE}=={version}")
        subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", f"{PACKAGE}=={version}"],
            check=True,
        )

        check_version = subprocess.run(
            [str(py), "-c",
             f"import importlib.metadata as m;"
             f"v=m.version('{PACKAGE}');"
             f"assert v=='{version}', f'expected {version}, got {{v}}';"
             f"print(v)"],
            check=True, capture_output=True, text=True,
        )
        print(f"    version: {check_version.stdout.strip()}")

        subprocess.run(
            [str(py), "-c",
             "import coden_retriever.architecture.adapters as a;"
             "names=sorted(x for x in dir(a) if not x.startswith('_'));"
             "print('adapters:', names)"],
            check=True,
        )

        subprocess.run([str(cli), "--help"], check=True, capture_output=True)
        print("    CLI: OK")


def merge_release_branch_to_main(branch_name: str) -> None:
    """Fast-forward main to the release branch, push, and delete the branch."""
    run_command("git checkout main")
    run_command(f"git merge --ff-only {branch_name}")
    run_command("git push origin main")
    run_command(f"git branch -d {branch_name}", check=False)
    run_command(f"git push origin --delete {branch_name}", check=False)


PRIVATE_REPO = Path(r"C:\Users\Vincent\OneDrive\Bureaublad\test_projects\code_retriever")


def previous_tag(current_tag: str) -> str | None:
    """Return the most recently-created v-tag that isn't current_tag."""
    result = subprocess.run(
        ["git", "tag", "--sort=-creatordate"],
        capture_output=True, text=True, check=True,
    )
    for line in result.stdout.splitlines():
        t = line.strip()
        if t.startswith("v") and t != current_tag:
            return t
    return None


def _commits_in_public_since(prev_tag: str) -> str:
    return subprocess.run(
        ["git", "log", f"{prev_tag}..HEAD", "--pretty=format:- %s", "--no-merges"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _commits_in_private_since_prev_release(prev_tag: str | None) -> str | None:
    """Pull private-repo commits landed since the previous public release.

    The public repo only carries bulk-synced commits, so its log is sparse.
    Private holds the granular `feat(...)` / `fix(...)` history. We anchor by
    the previous public tag's commit date and grab everything in private since.
    """
    if not PRIVATE_REPO.exists() or not prev_tag:
        return None

    prev_date = subprocess.run(
        ["git", "log", "-1", "--format=%aI", prev_tag],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not prev_date:
        return None

    result = subprocess.run(
        ["git", "-C", str(PRIVATE_REPO), "log",
         f"--since={prev_date}", "--pretty=format:- %s", "--no-merges"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def generate_release_notes(version: str) -> str:
    """Build release notes from commit log since the previous release.

    Prefers the richer private-repo history (granular feat/fix commits).
    Falls back to public's log, then to the last 50 commits.
    """
    current_tag = f"v{version}"
    prev = previous_tag(current_tag)

    private_log = _commits_in_private_since_prev_release(prev)
    if private_log:
        body = private_log
        header = f"## Changes since {prev}" if prev else "## Changes"
        source = "(from private repo log)"
    elif prev:
        body = _commits_in_public_since(prev) or "_No commits since previous tag._"
        header = f"## Changes since {prev}"
        source = "(from public repo log)"
    else:
        body = subprocess.run(
            ["git", "log", "--pretty=format:- %s", "--no-merges", "-50"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or "_No commit log available._"
        header = "## Changes"
        source = "(from public repo log, last 50 commits)"

    return (
        f"{header} {source}\n\n"
        f"{body}\n\n"
        f"## Install\n\n"
        f"```bash\n"
        f"pip install -U {PACKAGE}\n"
        f"```\n\n"
        f"PyPI: https://pypi.org/project/{PACKAGE}/{version}/\n"
    )


def create_github_release(version: str, notes_path: Path, wheel: Path, sdist: Path) -> None:
    """Invoke `gh release create` with the wheel + sdist + notes file."""
    cmd = [
        "gh", "release", "create", f"v{version}",
        str(wheel), str(sdist),
        "--title", f"v{version}",
        "--notes-file", str(notes_path),
    ]
    print(f"\n>>> Running: {' '.join(cmd)}")
    if DRY_RUN:
        print("    [DRY-RUN] Skipping actual gh call")
        return
    subprocess.run(cmd, check=True)


def run_pre_release_sync(script_dir: Path) -> None:
    """Spawn pre_release_sync.py with `y\\n` piped to its confirmation prompt."""
    sync_script = script_dir / "pre_release_sync.py"
    if not sync_script.exists():
        print("    pre_release_sync.py not found in script dir; skipping")
        return
    print(f"\n>>> Running {sync_script.name} (auto-confirm)")
    if DRY_RUN:
        print("    [DRY-RUN] Skipping actual sync")
        return
    subprocess.run(
        [sys.executable, str(sync_script)],
        input="y\n",
        text=True,
        check=True,
    )


def main() -> None:
    global AUTO_YES, DRY_RUN

    parser = argparse.ArgumentParser(
        description="End-to-end release script for coden-retriever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yes", action="store_true",
                        help="Auto-confirm every prompt with its default")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all side-effecting commands")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Don't run pre_release_sync.py at the start")
    parser.add_argument("--skip-tests", dest="skip_tests", action="store_true",
                        default=True,
                        help="Skip pytest (default: on — tests live in private repo only)")
    parser.add_argument("--run-tests", dest="skip_tests", action="store_false",
                        help="Run pytest from the private repo (opt-in; overrides default)")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Don't run Docker tests")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Don't run the post-upload fresh-venv smoke test")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Leave the release branch in place")
    parser.add_argument("--skip-gh-release", action="store_true",
                        help="Don't create the GitHub release")
    parser.add_argument("--notes-file", type=Path, default=None,
                        help="Path to a release-notes file (otherwise auto-generated)")
    parser.add_argument("--force", action="store_true",
                        help="Skip the preflight check that PyPI lacks this version")
    args = parser.parse_args()

    AUTO_YES = args.yes
    DRY_RUN = args.dry_run

    script_dir = Path(__file__).parent.resolve()
    env_path = script_dir / ".env"
    pyproject_path = script_dir / "pyproject.toml"
    dist_path = script_dir / "dist"
    docker_test_script = script_dir / "docker" / "run_tests.ps1"

    print("=" * 60)
    print("  Coden Retriever - End-to-End Release Script")
    print("=" * 60)
    if DRY_RUN:
        print("\n*** DRY-RUN MODE — no side effects ***")
    if AUTO_YES:
        print("\n*** AUTO-YES — prompts default-confirm; no stdin reads ***")

    env_vars = load_env(env_path)
    version = env_vars.get("CODEN_RETRIEVER_VERSION")
    pypi_token = env_vars.get("PYPI_API_TOKEN")

    if not version:
        print("Error: CODEN_RETRIEVER_VERSION not set in .env")
        sys.exit(1)
    if not pypi_token or pypi_token == "pypi-YOUR_API_TOKEN_HERE":
        print("Error: PYPI_API_TOKEN not set in .env")
        print("Get one at https://pypi.org/manage/account/token/")
        sys.exit(1)

    print(f"\nRelease Configuration:")
    print(f"  Package: {PACKAGE}")
    print(f"  Version: {version}")
    print(f"  PyPI Token: {'*' * 10}...{pypi_token[-4:]}")

    # Preflight: ensure PyPI doesn't already have this version (immutable).
    print("\n" + "=" * 60)
    print("  Preflight")
    print("=" * 60)
    if not args.force:
        if check_pypi_version_exists(version):
            print(f"\n  ERROR: {PACKAGE} {version} is already on PyPI.")
            print("  PyPI versions are immutable. Bump CODEN_RETRIEVER_VERSION in .env")
            print("  to a fresh version and retry. Use --force only when retrying after")
            print("  a partial-failure recovery (rarely correct).")
            sys.exit(1)
        print(f"    PyPI does not yet have {PACKAGE}=={version} - OK to proceed.")
    else:
        print("    Preflight skipped via --force")

    if not confirm("Proceed with release?"):
        print("Release cancelled.")
        sys.exit(0)

    # Step 0: Sync from private repo (NEW: auto by default)
    if not args.skip_sync:
        print("\n" + "=" * 60)
        print("  Step 0: Sync from private repo")
        print("=" * 60)
        run_pre_release_sync(script_dir)
    else:
        print("\n[Step 0 skipped via --skip-sync]")

    # Step 1: Create release branch
    print("\n" + "=" * 60)
    print("  Step 1: Create release branch")
    print("=" * 60)
    branch_name = f"release/v{version}"
    run_command(f"git checkout -b {branch_name}", check=False)

    # Step 2: Tests (default skipped — tests live in the private repo only)
    if not args.skip_tests:
        print("\n" + "=" * 60)
        print("  Step 2: Run pytest (in private repo)")
        print("=" * 60)
        if not PRIVATE_REPO.exists() or not (PRIVATE_REPO / "tests").exists():
            print(f"    No tests/ found at {PRIVATE_REPO}; skipping.")
        else:
            try:
                run_command(["pytest", "-vvv"], cwd=PRIVATE_REPO)
                print("    Tests passed!")
            except subprocess.CalledProcessError:
                print("    Tests FAILED!")
                if not confirm("Tests failed. Continue anyway?", default=False):
                    print("Aborting.")
                    sys.exit(1)
    else:
        print("\n[Step 2 skipped — tests are private-repo dev-only; use --run-tests to opt in]")

    # Step 2b: Docker tests (kept optional, not auto-confirmed if interactive)
    if not args.skip_docker and docker_test_script.exists():
        print("\n" + "=" * 60)
        print("  Step 2b: Run Docker tests")
        print("=" * 60)
        if confirm("Run Docker tests?"):
            try:
                run_command(
                    f'powershell -ExecutionPolicy Bypass -File "{docker_test_script}"'
                )
                print("    Docker tests passed!")
            except subprocess.CalledProcessError:
                print("    Docker tests FAILED!")
                if not confirm("Docker tests failed. Continue anyway?", default=False):
                    sys.exit(1)

    # Step 3: Bump pyproject.toml
    print("\n" + "=" * 60)
    print("  Step 3: Bump pyproject.toml")
    print("=" * 60)
    update_pyproject_version(pyproject_path, version)

    # Step 4: Commit
    print("\n" + "=" * 60)
    print("  Step 4: Commit")
    print("=" * 60)
    run_command("git add -A")
    run_command(f'git commit -am "Update version to {version}"', check=False)

    # Step 5: Tag
    print("\n" + "=" * 60)
    print("  Step 5: Create git tag")
    print("=" * 60)
    tag_name = f"v{version}"
    tag_check = subprocess.run(
        f"git tag -l {tag_name}", shell=True, capture_output=True, text=True,
    )
    if tag_check.stdout.strip() == tag_name:
        if confirm(f"Tag {tag_name} already exists. Delete and recreate?", default=False):
            run_command(f"git tag -d {tag_name}")
            run_command(f"git push origin --delete {tag_name}", check=False)
            run_command(f'git tag -a {tag_name} -m "Release {version}"')
        else:
            print(f"    Keeping existing tag {tag_name}")
    else:
        run_command(f'git tag -a {tag_name} -m "Release {version}"')

    # Step 6: Push branch + tags
    print("\n" + "=" * 60)
    print("  Step 6: Push to remote")
    print("=" * 60)
    if confirm("Push branch + tags to remote?"):
        run_command(f"git push -u origin {branch_name}", check=False)
        run_command("git push --tags")

    # Step 7: Build
    print("\n" + "=" * 60)
    print("  Step 7: Build distribution")
    print("=" * 60)
    clean_dist(dist_path)
    run_command("python -m build")

    # Step 8: Upload (point of no return)
    print("\n" + "=" * 60)
    print("  Step 8: Upload to PyPI (IRREVERSIBLE)")
    print("=" * 60)
    run_command("twine check dist/*")
    if not confirm("Upload to PyPI?"):
        print("\nSkipped upload. Stopping here. Dist artifacts left in dist/.")
        sys.exit(0)
    upload_env = {"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": pypi_token}
    run_command("twine upload dist/* --verbose", env=upload_env)
    print("\n    Uploaded to PyPI.")

    # Step 9: Poll PyPI for index propagation (JSON + simple index)
    if not DRY_RUN:
        print("\n" + "=" * 60)
        print("  Step 9: Poll PyPI for propagation")
        print("=" * 60)
        try:
            poll_pypi_json(version)
            poll_pypi_simple_index(version)
        except TimeoutError as e:
            print(f"    {e}")
            if not confirm("Continue without confirmed propagation?", default=False):
                sys.exit(1)
    else:
        print("\n[Step 9 skipped in dry-run]")

    # Step 10: Fresh-venv smoke
    if not args.skip_smoke and not DRY_RUN:
        print("\n" + "=" * 60)
        print("  Step 10: Fresh-venv smoke test")
        print("=" * 60)
        try:
            fresh_venv_smoke(version)
            print("    Smoke test passed.")
        except (subprocess.CalledProcessError, AssertionError, Exception) as e:
            print(f"\n    SMOKE TEST FAILED: {type(e).__name__}: {e}")
            if not confirm("Continue with merge + GitHub release anyway?", default=False):
                print("Aborting before merge. Release is already on PyPI.")
                sys.exit(1)
    else:
        print("\n[Step 10 skipped]")

    # Step 11: Merge to main + delete release branch
    if not args.skip_merge:
        print("\n" + "=" * 60)
        print("  Step 11: Merge release branch -> main")
        print("=" * 60)
        if confirm("Fast-forward main and delete the release branch?"):
            try:
                merge_release_branch_to_main(branch_name)
            except subprocess.CalledProcessError as e:
                print(f"\n    Merge failed: {e}")
                if not confirm("Continue?", default=False):
                    sys.exit(1)
    else:
        print("\n[Step 11 skipped via --skip-merge]")

    # Step 12: GitHub release
    if not args.skip_gh_release:
        print("\n" + "=" * 60)
        print("  Step 12: GitHub release")
        print("=" * 60)
        if args.notes_file:
            notes = args.notes_file.read_text(encoding="utf-8")
            print(f"    Using notes from {args.notes_file}")
        else:
            notes = generate_release_notes(version)
            print("    Generated notes from commit log")

        if confirm("Create GitHub release with wheel + sdist?"):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8",
            ) as f:
                f.write(notes)
                notes_path = Path(f.name)
            try:
                wheel = next(dist_path.glob("*.whl"))
                sdist = next(dist_path.glob("*.tar.gz"))
                create_github_release(version, notes_path, wheel, sdist)
            except StopIteration:
                print("    ERROR: wheel or sdist not found in dist/")
            except subprocess.CalledProcessError as e:
                print(f"    gh release create failed: {e}")
            finally:
                notes_path.unlink(missing_ok=True)
    else:
        print("\n[Step 12 skipped via --skip-gh-release]")

    # Step 13: Final verification
    print("\n" + "=" * 60)
    print("  Step 13: Final verification")
    print("=" * 60)
    if not DRY_RUN:
        try:
            data = poll_pypi_json(version, max_wait=30)
            info = data["info"]
            urls = data["urls"]
            print(f"    PyPI version: {info['version']}")
            for u in urls:
                print(f"    {u['packagetype']:>12s}: {u['size']:>10d}  {u['filename']}")
            extras = info.get("provides_extra") or []
            stale = set(extras) & {"semantic", "mcp", "agent", "all"}
            if stale:
                print(f"    WARNING: stale extras present: {sorted(stale)}")
            else:
                print(f"    Extras: {extras} (clean)")
        except Exception as e:
            print(f"    Verification failed: {e}")
    else:
        print("[Step 13 skipped in dry-run]")

    print("\n" + "=" * 60)
    print("  Release complete!")
    print("=" * 60)
    print(f"\n  PyPI:    https://pypi.org/project/{PACKAGE}/{version}/")
    print(f"  GitHub:  https://github.com/{REPO_SLUG}/releases/tag/v{version}")
    print(f"  Install: pip install -U {PACKAGE}")


if __name__ == "__main__":
    main()
