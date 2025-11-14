# shared-test-utils

Shared test utilities for the deepdoctection monorepo.

## Purpose

This package provides:

- **Factories**: Deterministic in-memory test assets (e.g., `TestPdfPage`, `build_test_pdf_page`) for reproducible testing without filesystem dependencies
- **Tier-A Asset Manifest & Loader**: A YAML-based manifest (`assets_manifest.yaml`) for tracking small test assets with SHA-256 verification
- **CLI Tool**: `dd-add-asset` command for adding/updating manifest entries

## Usage

Install this package in editable mode when running tests:

```bash
pip install -e packages/shared_test_utils
```

Then import the utilities in your test code:

```python
from shared_test_utils import build_test_pdf_page, asset_path, asset_info, list_keys

# Use deterministic in-memory PDF
test_pdf = build_test_pdf_page()

# Access manifest-tracked assets
path = asset_path("my_test_asset", verify=True)
```

## Environment Overrides

The asset loader supports two environment variables:

- **`SHARED_TEST_UTILS_MANIFEST`**: Absolute path to a custom manifest file (default: packaged `assets_manifest.yaml`)
- **`DD_TESTDATA_ROOT`**: Absolute base directory for resolving asset paths (default: computed repository root)

## Adding Assets

Use the provided CLI tool to add new entries to the manifest:

```bash
dd-add-asset --key my_asset --path path/to/asset.pdf --license Apache-2.0 --tags "pdf,test" --description "Test PDF document"
```

Options:
- `--key`: Unique asset identifier (required)
- `--path`: Path to asset file, relative to repo root or absolute (required)
- `--license`: License identifier (required)
- `--tags`: Comma-separated tags (optional)
- `--description`: Human-readable description (optional)
- `--manifest`: Custom manifest path (optional)
- `--force`: Overwrite existing key (optional)

## Scope

This package contains only **Tier-A** assets (small test files checked into the repository). Tier-B (Git LFS) and Tier-C (DVC) assets are out of scope.

