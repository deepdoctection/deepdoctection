# -*- coding: utf-8 -*-
# File: conftest.py

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
Configuration file for pytest fixtures.

This module contains pytest fixtures to provide reusable components for testing purposes.
These fixtures include setups for handling PDF files, numerical arrays, image arrays,
bounding boxes, and category names.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest import fixture


@fixture(name="pdf_bytes")
def fixture_pdf_bytes(pdf_file_path_two_pages: Path) -> bytes:
    """PDF as bytes"""
    if pdf_file_path_two_pages.exists():
        with open(pdf_file_path_two_pages, "rb") as f:
            return f.read()
    return b""


@fixture(name="coords")
def fixture_coords() -> NDArray[np.float32]:
    """Sample coordinates"""
    return np.array([[10.0, 20.0, 50.0, 80.0], [30.0, 40.0, 90.0, 100.0]], dtype=np.float32)


@fixture(name="np_image")
def fixture_np_image() -> NDArray[np.uint8]:
    """np_array image"""
    return np.ones([100, 150, 3], dtype=np.uint8) * 255


@fixture(name="boxes")
def fixture_boxes() -> NDArray[np.float32]:
    """Sample bounding boxes"""
    return np.array([[10.0, 20.0, 50.0, 80.0], [60.0, 30.0, 100.0, 90.0]], dtype=np.float32)


@fixture(name="category_names")
def fixture_category_names() -> list[tuple[str, str]]:
    """Sample category names"""
    return [("category1", "value1"), ("category2", "value2")]
