# -*- coding: utf-8 -*-
# File: factories.py

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
Deterministic in-memory test asset factories.

Provides factory functions for creating test assets without filesystem dependencies.
All factories produce stable, reproducible outputs suitable for cross-platform testing.
"""

from dataclasses import dataclass

import numpy as np

from ._data import PDF_BYTES


@dataclass(frozen=True)
class TestPdfPage:
    """
    Container for a deterministic single-page PDF test asset.

    Attributes:
        pdf_bytes: Raw PDF file content as bytes
        loc: Logical location identifier
        file_name: Suggested filename for this PDF
        np_array_shape_default: Expected numpy array shape at default DPI (72)
        np_array_shape_300: Expected numpy array shape at 300 DPI
    """

    pdf_bytes: bytes
    loc: str =  "/testlocation/test"
    file_name: str  = "test_image_0.pdf"
    np_array_shape: tuple[int, int, int] = (3300, 2550, 3)
    np_array_shape_default: tuple[int, int, int] = (792, 612, 3)



@dataclass(frozen=True)
class WhiteImage:
    """Test fixture for a white image with deterministic properties"""

    image = np.ones([400, 600, 3], dtype=np.uint8)
    location = "/testlocation/test"
    file_name = "test_image.png"
    external_id = "1234"
    uuid = "90c05f37-0000-0000-0000-b84f9d14ff44"


def build_test_pdf_page() -> TestPdfPage:
    """
    Build a deterministic single-page PDF for testing.
    """

    return TestPdfPage(
        pdf_bytes=PDF_BYTES
    )


def build_white_image() -> WhiteImage:
    """
    Build a deterministic white image for testing.
    """
    return WhiteImage()


