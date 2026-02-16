#!/usr/bin/env python3
"""
Automated release script for coden-retriever.

This script automates the process of releasing a new version to PyPI.
It reads configuration from .env file and executes each step with user confirmation.

Usage:
    python release.py [--skip-tests] [--skip-docker] [--dry-run]

Options:
    --skip-tests    Skip running pytest
    --skip-docker   Skip running Docker tests
    --dry-run       Run without making actual changes (for testing the script)
"""

import os
import re
import sys
import subprocess
import argparse
from pathlib import Path


def load_env(env_path: Path) -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        print("Please create a .env file with CODEN_RETRIEVER_VERSION and PYPI_API_TOKEN")
        sys.exit(1)

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


def run_command(cmd: str | list[str], dry_run: bool = False, check: bool = True,
                capture_output: bool = False, env: dict | None = None) -> subprocess.CompletedProcess | None:
    """Run a shell command with optional dry-run mode."""
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_str = " ".join(cmd)

    print(f"\n>>> Running: {cmd_str}")

    if dry_run:
        print("    [DRY-RUN] Skipping actual execution")
        return None

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        capture_output=capture_output,
        text=True,
        env=merged_env
    )
    return result


def confirm(message: str, default: bool = True) -> bool:
    """Ask user for confirmation."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = input(f"\n{message}{suffix}").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


def update_pyproject_version(pyproject_path: Path, new_version: str, dry_run: bool = False) -> None:
    """Update the version in pyproject.toml."""
    print(f"\n>>> Updating version in pyproject.toml to {new_version}")

    if dry_run:
        print("    [DRY-RUN] Skipping actual update")
        return

    content = pyproject_path.read_text()
    updated_content = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE
    )
    pyproject_path.write_text(updated_content)
    print(f"    Updated version to {new_version}")


def clean_dist(dist_path: Path, dry_run: bool = False) -> None:
    """Clean the dist directory before building."""
    print(f"\n>>> Cleaning dist directory")

    if dry_run:
        print("    [DRY-RUN] Skipping cleanup")
        return

    if dist_path.exists():
        import shutil
        shutil.rmtree(dist_path)
        print(f"    Removed {dist_path}")
    else:
        print(f"    {dist_path} does not exist, nothing to clean")


def main():
    parser = argparse.ArgumentParser(description="Automated release script for coden-retriever")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pytest")
    parser.add_argument("--skip-docker", action="store_true", help="Skip running Docker tests")
    parser.add_argument("--dry-run", action="store_true", help="Run without making actual changes")
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    env_path = script_dir / ".env"
    pyproject_path = script_dir / "pyproject.toml"
    dist_path = script_dir / "dist"
    docker_test_script = script_dir / "docker" / "run_tests.ps1"

    print("=" * 60)
    print("  Coden Retriever - Automated Release Script")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY-RUN MODE - No actual changes will be made ***")

    # Load environment variables
    env_vars = load_env(env_path)
    version = env_vars.get("CODEN_RETRIEVER_VERSION")
    pypi_token = env_vars.get("PYPI_API_TOKEN")

    if not version:
        print("Error: CODEN_RETRIEVER_VERSION not set in .env file")
        sys.exit(1)

    if not pypi_token or pypi_token == "pypi-YOUR_API_TOKEN_HERE":
        print("Error: PYPI_API_TOKEN not set or is still placeholder in .env file")
        print("Please get your API token from https://pypi.org/manage/account/token/")
        sys.exit(1)

    print(f"\nRelease Configuration:")
    print(f"  Version: {version}")
    print(f"  PyPI Token: {'*' * 10}...{pypi_token[-4:]}")

    if not confirm("Proceed with release?"):
        print("Release cancelled.")
        sys.exit(0)

    # Step 0: Create release branch
    print("\n" + "=" * 60)
    print("  Step 0: Create release branch")
    print("=" * 60)

    branch_name = f"release/v{version}"
    run_command(f"git checkout -b {branch_name}", dry_run=args.dry_run, check=False)

    # Step 1: Run tests
    print("\n" + "=" * 60)
    print("  Step 1: Run tests")
    print("=" * 60)

    if not args.skip_tests:
        if confirm("Run pytest locally?"):
            try:
                run_command("pytest -vvv", dry_run=args.dry_run)
                print("    Tests passed!")
            except subprocess.CalledProcessError:
                print("    Tests FAILED!")
                if not confirm("Tests failed. Continue anyway?", default=False):
                    sys.exit(1)
    else:
        print("    Skipping pytest (--skip-tests)")

    if not args.skip_docker:
        if docker_test_script.exists():
            if confirm("Run Docker tests (docker/run_tests.ps1)?"):
                try:
                    run_command(f"powershell -ExecutionPolicy Bypass -File \"{docker_test_script}\"",
                               dry_run=args.dry_run)
                    print("    Docker tests passed!")
                except subprocess.CalledProcessError:
                    print("    Docker tests FAILED!")
                    if not confirm("Docker tests failed. Continue anyway?", default=False):
                        sys.exit(1)
        else:
            print(f"    Docker test script not found at {docker_test_script}, skipping")
    else:
        print("    Skipping Docker tests (--skip-docker)")

    # Step 2: Update version in pyproject.toml
    print("\n" + "=" * 60)
    print("  Step 2: Update version")
    print("=" * 60)

    update_pyproject_version(pyproject_path, version, dry_run=args.dry_run)

    # Step 3: Commit changes
    print("\n" + "=" * 60)
    print("  Step 3: Commit changes")
    print("=" * 60)

    run_command("git add -A", dry_run=args.dry_run)
    run_command(f'git commit -am "Update version to {version}"', dry_run=args.dry_run, check=False)

    # Step 4: Create git tag
    print("\n" + "=" * 60)
    print("  Step 4: Create git tag")
    print("=" * 60)

    tag_name = f"v{version}"
    # Check if tag already exists
    tag_check = subprocess.run(f"git tag -l {tag_name}", shell=True, capture_output=True, text=True)
    if tag_check.stdout.strip() == tag_name:
        if confirm(f"Tag {tag_name} already exists. Delete and recreate it?", default=False):
            run_command(f"git tag -d {tag_name}", dry_run=args.dry_run)
        else:
            print(f"    Keeping existing tag {tag_name}")
            tag_name = None  # Skip tag creation

    if tag_name:
        run_command(f'git tag -a {tag_name} -m "New release version {version}"', dry_run=args.dry_run)

    # Step 5: Push changes and tags
    print("\n" + "=" * 60)
    print("  Step 5: Push to remote")
    print("=" * 60)

    if confirm("Push changes and tags to remote?"):
        run_command("git push", dry_run=args.dry_run, check=False)
        run_command("git push --tags", dry_run=args.dry_run)

    # Step 6: Build distribution packages
    print("\n" + "=" * 60)
    print("  Step 6: Build distribution packages")
    print("=" * 60)

    clean_dist(dist_path, dry_run=args.dry_run)
    run_command("python -m build", dry_run=args.dry_run)

    # Step 7: Upload to PyPI
    print("\n" + "=" * 60)
    print("  Step 7: Upload to PyPI")
    print("=" * 60)

    # Check distribution files
    run_command("twine check dist/*", dry_run=args.dry_run)

    if confirm("Upload to PyPI?"):
        # Pass the token via environment variable for security
        upload_env = {"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": pypi_token}
        run_command("twine upload dist/* --verbose", dry_run=args.dry_run, env=upload_env)
        print("\n    Package uploaded successfully!")

    # Step 8: Merge instructions
    print("\n" + "=" * 60)
    print("  Step 8: Merge release branch")
    print("=" * 60)
    print(f"""
    If everything went well, merge the release branch:

    git checkout main
    git merge {branch_name}
    git push
    git branch -d {branch_name}
    """)

    print("\n" + "=" * 60)
    print("  Release complete!")
    print("=" * 60)
    print(f"\n  Package: coden-retriever v{version}")
    print(f"  PyPI: https://pypi.org/project/coden-retriever/{version}/")


if __name__ == "__main__":
    main()
