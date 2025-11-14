# -*- coding: utf-8 -*-
"""
CLI tool to add or update entries in the asset manifest.

Provides the dd-add-asset command for managing the assets_manifest.yaml file.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

from ..assets import get_repo_root, get_default_manifest_path



def _compute_sha256(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file using streaming reads.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal SHA-256 digest string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _normalize_path(input_path: str, repo_root: Path) -> tuple[Path, str]:
    """
    Normalize an input path to absolute and compute repo-relative version.

    Args:
        input_path: User-provided path (absolute or relative)
        repo_root: Repository root directory

    Returns:
        Tuple of (absolute_path, relative_path_string)
        relative_path_string uses forward slashes
    """
    path = Path(input_path)

    # Convert to absolute
    if not path.is_absolute():
        # Treat as relative to repo root
        abs_path = (repo_root / path).resolve()
    else:
        abs_path = path.resolve()

    # Compute relative path from repo root with forward slashes
    try:
        rel_path = abs_path.relative_to(repo_root)
        rel_path_str = rel_path.as_posix()
    except ValueError:
        # Path is outside repo root, store as absolute with forward slashes
        rel_path_str = abs_path.as_posix()
        print(
            f"Warning: Path {abs_path} is outside repository root {repo_root}. "
            "Storing as absolute path.",
            file=sys.stderr,
        )

    return abs_path, rel_path_str


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """
    Load existing manifest or return empty dict if file doesn't exist.

    Args:
        manifest_path: Path to manifest file

    Returns:
        Dictionary of manifest entries
    """
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        print(f"Error: Manifest must be a YAML mapping, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)

    return data


def _write_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    """
    Write manifest to YAML file with deterministic formatting.

    Args:
        manifest_path: Path to manifest file
        data: Dictionary to write
    """
    # Sort top-level keys for deterministic output
    sorted_data = {k: data[k] for k in sorted(data.keys())}

    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(
            sorted_data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,  # We pre-sorted, keep our order
            width=120,
        )


def main() -> None:
    """
    Entry point for dd-add-asset CLI command.

    Parses arguments and adds/updates an asset entry in the manifest.
    """
    parser = argparse.ArgumentParser(
        description="Add or update an asset entry in the Tier-A manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dd-add-asset --key sample_pdf --path tests/data/sample.pdf --license Apache-2.0
  dd-add-asset --key sample_pdf --path tests/data/sample.pdf --license Apache-2.0 --tags "pdf,test" --description "Sample PDF for testing"
  dd-add-asset --key sample_pdf --path /abs/path/to/sample.pdf --license MIT --force
        """,
    )

    parser.add_argument(
        "--key",
        required=True,
        help="Unique logical identifier for this asset",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the asset file (absolute or relative to repo root)",
    )
    parser.add_argument(
        "--license",
        required=True,
        help="License identifier (e.g., Apache-2.0, MIT, CC0-1.0)",
    )
    parser.add_argument(
        "--tags",
        help="Comma-separated tags for categorization",
    )
    parser.add_argument(
        "--description",
        help="Human-readable description of the asset",
    )
    parser.add_argument(
        "--manifest",
        help="Path to manifest file (default: packaged assets_manifest.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing entry if key already exists",
    )

    args = parser.parse_args()

    # Determine paths
    repo_root = get_repo_root()
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = manifest_path.resolve()
    else:
        manifest_path = get_default_manifest_path()

    # Normalize and validate asset path
    abs_asset_path, rel_asset_path = _normalize_path(args.path, repo_root)

    if not abs_asset_path.exists():
        print(f"Error: Asset file does not exist: {abs_asset_path}", file=sys.stderr)
        sys.exit(1)

    if not abs_asset_path.is_file():
        print(f"Error: Asset path is not a file: {abs_asset_path}", file=sys.stderr)
        sys.exit(1)

    # Load existing manifest
    manifest = _load_manifest(manifest_path)

    # Check for existing key
    if args.key in manifest and not args.force:
        print(
            f"Error: Key '{args.key}' already exists in manifest. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compute file metadata
    print(f"Computing SHA-256 hash for {abs_asset_path}...", file=sys.stderr)
    sha256_digest = _compute_sha256(abs_asset_path)
    file_size = abs_asset_path.stat().st_size

    # Build entry
    entry: dict[str, Any] = {
        "path": rel_asset_path,
        "sha256": sha256_digest,
        "size": file_size,
        "license": args.license,
    }

    if args.tags:
        # Split and strip whitespace
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        if tags:
            entry["tags"] = tags

    if args.description:
        entry["description"] = args.description

    # Update manifest
    manifest[args.key] = entry

    # Write back
    _write_manifest(manifest_path, manifest)

    action = "Updated" if args.key in manifest else "Added"
    print(f"{action} asset '{args.key}' in {manifest_path}", file=sys.stderr)
    print(f"  Path: {rel_asset_path}", file=sys.stderr)
    print(f"  SHA-256: {sha256_digest}", file=sys.stderr)
    print(f"  Size: {file_size} bytes", file=sys.stderr)
    print(f"  License: {args.license}", file=sys.stderr)


if __name__ == "__main__":
    main()

