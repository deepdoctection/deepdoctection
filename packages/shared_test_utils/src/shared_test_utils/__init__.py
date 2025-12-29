# -*- coding: utf-8 -*-
# File: __init__.py

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

