#!/usr/bin/env python3
"""
bump_versions.py

CLI utility to set the SAME version across all deepdoctection packages.

Expected repo layout:

- `deepdoctection/`
  - `scripts/`
    - `bump_versions.py`   (this file)
  - `packages/`
    - `dd_core/pyproject.toml`
    - `dd_core/src/dd_core/__init__.py`
    - `dd_datasets/pyproject.toml`
    - `dd_datasets/src/dd_datasets/__init__.py`
    - `deepdoctection/pyproject.toml`
    - `deepdoctection/src/deepdoctection/__init__.py`

What it updates:
- In each `pyproject.toml`: `[project]` section `version = "..."`.
- In each package `__init__.py`: `__version__ = "..."`.

Usage:
- From repository root:
    python scripts/bump_version.py 1.0.1

- From anywhere:
    python /path/to/repo/deepdoctection/scripts/bump_versions.py 1.0.1

Exit codes:
- 0: success (no changes or applied changes)
- 2: invalid version format
- 3: repo layout not found
- 4: missing expected file(s)
- 5: expected key not found in a target file
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PACKAGE_DIRS = ("dd_core", "dd_datasets", "deepdoctection")


def find_deepdoctection_root(start: Path) -> Path:
    """
    Find the `deepdoctection/` directory that contains `packages/` and `scripts/`.
    Walks upward from `start`.
    """
    cur = start.resolve()
    for parent in (cur, *cur.parents):
        if (parent / "packages").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate `deepdoctection/` root (missing `packages/` and `scripts/`).")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def update_pyproject_version(pyproject_path: Path, new_version: str, *, dry_run: bool) -> bool:
    content = read_text(pyproject_path)

    section_re = re.compile(r"(?ms)^\[project\]\s*$.*?(?=^\[|\Z)")
    m = section_re.search(content)
    if not m:
        raise KeyError(f"Missing `[project]` section in {pyproject_path}")

    project_block = m.group(0)
    version_re = re.compile(r'(?m)^(version\s*=\s*)"(.*?)"\s*$')
    if not version_re.search(project_block):
        raise KeyError(f"Missing `version = \"...\"` in `[project]` section of {pyproject_path}")

    updated_block = version_re.sub(rf'\1"{new_version}"', project_block, count=1)
    if updated_block == project_block:
        return False

    updated_content = content[: m.start()] + updated_block + content[m.end() :]
    if not dry_run:
        write_text(pyproject_path, updated_content)
    return True


def update_init_py_version(init_path: Path, new_version: str, *, dry_run: bool) -> bool:
    content = read_text(init_path)

    version_re = re.compile(r'(?m)^(__version__\s*=\s*)"(.*?)"\s*$')
    if not version_re.search(content):
        raise KeyError(f"Missing `__version__ = \"...\"` in {init_path}")

    updated = version_re.sub(rf'\1"{new_version}"', content, count=1)
    if updated == content:
        return False

    if not dry_run:
        write_text(init_path, updated)
    return True

def update_dockerfile_version(dockerfile_path: Path, new_version: str, *, dry_run: bool) -> bool:
    content = read_text(dockerfile_path)

    # Match e.g. `ARG DEEPDOCTECTION_VERSION=1.0.4` with flexible whitespace
    arg_re = re.compile(r"(?m)^(ARG\s+DEEPDOCTECTION_VERSION\s*=\s*)(\S+)\s*$")
    if not arg_re.search(content):
        raise KeyError(f"Missing `ARG DEEPDOCTECTION_VERSION=...` in {dockerfile_path}")

    # Use \g<1> to avoid ambiguity with digits (e.g. "\11" being parsed as group 11).
    updated = arg_re.sub(rf"\g<1>{new_version}", content, count=1)
    if updated == content:
        return False

    if not dry_run:
        write_text(dockerfile_path, updated)
    return True

def is_valid_version(v: str) -> bool:
    # Simple PEP 440-ish guard; accepts common forms like 1.0.1, 1.0.1rc1, 1.0.1.dev1
    return bool(re.fullmatch(r"[0-9]+(\.[0-9]+)*([a-zA-Z0-9\.\-\+]+)?", v))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bump_versions.py",
        description="Set the same version in all deepdoctection packages under `deepdoctection/packages`.",
    )
    parser.add_argument("version", help="New version, e.g. 1.0.1")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    args = parser.parse_args(argv)

    new_version = args.version.strip()
    if not is_valid_version(new_version):
        print(f"Invalid version format: {new_version}", file=sys.stderr)
        return 2

    try:
        dd_root = find_deepdoctection_root(Path(__file__).parent)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 3

    packages_root = dd_root / "packages"
    dockerfile_path = dd_root / "docker" / "gpu" / "Dockerfile"

    targets_pyproject: list[Path] = []
    targets_init: list[Path] = []

    for pkg in PACKAGE_DIRS:
        targets_pyproject.append(packages_root / pkg / "pyproject.toml")
        targets_init.append(packages_root / pkg / "src" / pkg / "__init__.py")

    missing = [p for p in (*targets_pyproject, *targets_init) if not p.is_file()]
    if missing:
        print("Missing expected file(s):", file=sys.stderr)
        for p in missing:
            print(f"- {p}", file=sys.stderr)
        return 4

    changed: list[Path] = []
    try:
        for p in targets_pyproject:
            if update_pyproject_version(p, new_version, dry_run=args.dry_run):
                changed.append(p)
        for p in targets_init:
            if update_init_py_version(p, new_version, dry_run=args.dry_run):
                changed.append(p)
        if update_dockerfile_version(dockerfile_path, new_version, dry_run=args.dry_run):
            changed.append(dockerfile_path)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 5

    if args.dry_run:
        if changed:
            print("Would update:")
            for p in changed:
                print(f"- {p}")
        else:
            print("No changes needed.")
        return 0

    if changed:
        print("Updated:")
        for p in changed:
            print(f"- {p}")
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
