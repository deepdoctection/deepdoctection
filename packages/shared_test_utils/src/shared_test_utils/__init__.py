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

from shared_test_utils.assets import asset_info, asset_path, list_keys
from shared_test_utils.factories import TestPdfPage, build_test_pdf_page, WhiteImage, build_white_image

__all__ = [
    "TestPdfPage",
    "WhiteImage",
    "build_test_pdf_page",
    "build_white_image",
    "asset_path",
    "asset_info",
    "list_keys",
]

__version__ = "1.0.0"

