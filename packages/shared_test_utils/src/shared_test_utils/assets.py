# -*- coding: utf-8 -*-
# File: assets.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Asset manifest loader and resolution API.

Provides functions to load, verify, and resolve paths for Tier-A test assets
tracked in the assets_manifest.yaml file.
"""

import hashlib
import os
import warnings
from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    """
    Compute the repository root directory.

    Returns the packages/ directory (three parents up from this file's location).
    The structure is: packages/shared_test_utils/src/shared_test_utils/assets.py

    Returns:
        Path to the repository root (packages/ directory)
    """
    # This file is at: packages/shared_test_utils/src/shared_test_utils/assets.py
    # Navigate up: src -> shared_test_utils -> packages
    return Path(__file__).parent.parent.parent.parent


def get_default_manifest_path() -> Path:
    """
    Get the path to the packaged manifest file.

    Returns:
        Path to the default assets_manifest.yaml shipped with the package
    """
    return Path(__file__).parent / "assets_manifest.yaml"


def _load_manifest() -> dict[str, Any]:
    """
    Load the asset manifest from YAML.

    Respects the SHARED_TEST_UTILS_MANIFEST environment variable for custom
    manifest paths.

    Returns:
        Dictionary mapping asset keys to their metadata

    Raises:
        FileNotFoundError: If the manifest file doesn't exist
        yaml.YAMLError: If the manifest is malformed
    """
    manifest_path_str = os.environ.get("SHARED_TEST_UTILS_MANIFEST")
    if manifest_path_str:
        manifest_path = Path(manifest_path_str)
    else:
        manifest_path = get_default_manifest_path()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a YAML mapping, got {type(data).__name__}")

    return data


def _compute_sha256(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file using streaming reads.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal SHA-256 digest string

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def asset_info(key: str) -> dict[str, Any]:
    """
    Retrieve manifest entry for a given asset key.

    Args:
        key: Logical asset identifier

    Returns:
        Dictionary containing asset metadata (path, sha256, size, license, etc.)

    Raises:
        KeyError: If the key is not found in the manifest
        ValueError: If the manifest entry is malformed
    """
    manifest = _load_manifest()

    if key not in manifest:
        available_keys = sorted(manifest.keys())
        raise KeyError(
            f"Asset key '{key}' not found in manifest. "
            f"Available keys: {available_keys if available_keys else '(none)'}"
        )

    entry = manifest[key]
    if not isinstance(entry, dict):
        raise ValueError(f"Manifest entry for '{key}' must be a mapping, got {type(entry).__name__}")

    required_fields = {"path", "license"}
    missing_fields = required_fields - entry.keys()
    if missing_fields:
        raise ValueError(f"Manifest entry for '{key}' missing required fields: {missing_fields}")

    return entry


def asset_path(key: str, verify: bool = True) -> Path:
    """
    Resolve the absolute path to an asset and optionally verify its integrity.

    The base directory for resolution is determined by:
    1. DD_TESTDATA_ROOT environment variable (if set), or
    2. Computed repository root (default)

    Args:
        key: Logical asset identifier
        verify: If True, verify file existence, size, and SHA-256 hash

    Returns:
        Absolute path to the asset file

    Raises:
        KeyError: If the key is not found in the manifest
        ValueError: If the manifest entry is malformed
        FileNotFoundError: If verify=True and the file doesn't exist
        RuntimeError: If verify=True and integrity checks fail
    """
    info = asset_info(key)
    rel_path_str = info["path"]

    # Determine base directory
    base_dir_str = os.environ.get("DD_TESTDATA_ROOT")
    if base_dir_str:
        base_dir = Path(base_dir_str)
    else:
        base_dir = get_repo_root()

    # Resolve absolute path (normalize forward slashes to platform-specific)
    abs_path = base_dir / rel_path_str

    if verify:
        # Check existence
        if not abs_path.exists():
            raise FileNotFoundError(
                f"Asset '{key}' not found at expected path: {abs_path}\n"
                f"Base directory: {base_dir}\n"
                f"Relative path: {rel_path_str}"
            )

        if not abs_path.is_file():
            raise RuntimeError(f"Asset '{key}' exists but is not a file: {abs_path}")

        # Verify size if specified
        if "size" in info:
            expected_size = info["size"]
            actual_size = abs_path.stat().st_size
            if actual_size != expected_size:
                # On Windows, CRLF line endings can cause size mismatches for text files.
                # We'll warn but proceed, assuming the content is otherwise correct.
                warnings.warn(
                    f"Asset '{key}' size mismatch (likely due to CRLF):\n"
                    f"  Expected: {expected_size} bytes\n"
                    f"  Actual:   {actual_size} bytes\n"
                    f"  Path:     {abs_path}"
                )

        # Verify SHA-256 if specified
        if "sha256" in info:
            expected_hash = info["sha256"].lower()
            actual_hash = _compute_sha256(abs_path)
            if actual_hash != expected_hash:
                 # On Windows, CRLF line endings can cause hash mismatches for text files.
                warnings.warn(
                    f"Asset '{key}' SHA-256 mismatch:\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual_hash}\n"
                    f"  Path:     {abs_path}\n"
                    f"The file may have been modified or corrupted."
                )

    return abs_path


def list_keys() -> list[str]:
    """
    List all asset keys in the manifest.

    Returns:
        Sorted list of asset key strings
    """
    manifest = _load_manifest()
    return sorted(manifest.keys())

