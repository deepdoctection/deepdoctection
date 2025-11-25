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
Fixtures for datapoint package testing
"""

from __future__ import annotations

from pytest import fixture

from ..data import PDF_BYTES
from ..conftest import WhiteImage, TestPdfPage



@fixture(name="white_image")
def fixture_image() -> WhiteImage:
    """Provide a white test image"""
    return WhiteImage()


@fixture(name="pdf_page")
def fixture_pdf_page() -> TestPdfPage:
    """Provide a deterministic 1-page PDF for rendering tests."""
    return TestPdfPage(
        pdf_bytes=PDF_BYTES
    )





