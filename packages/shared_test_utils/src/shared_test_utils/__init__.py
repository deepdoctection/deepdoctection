# -*- coding: utf-8 -*-
"""
Public API for shared_test_utils package.

Exports:
    - TestPdfPage: Dataclass for deterministic PDF test assets
    - build_test_pdf_page: Factory function for creating test PDFs
    - asset_path: Resolve and verify asset paths from manifest
    - asset_info: Retrieve asset metadata from manifest
    - list_keys: List all available asset keys
"""

from .assets import asset_info, asset_path, list_keys
from shared_test_utils.collector import collect_datapoint_from_dataflow

__all__ = [
    "asset_path",
    "asset_info",
    "list_keys",
    "collect_datapoint_from_dataflow",
]

__version__ = "1.0.0"

