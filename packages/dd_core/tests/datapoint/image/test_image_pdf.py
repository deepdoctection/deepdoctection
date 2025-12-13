# -*- coding: utf-8 -*-
# File: test_image_pdf.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
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
Testing Image PDF operations and special cases
"""

from dd_core.datapoint import Image

from ..conftest import TestPdfPage


class TestImagePDF:
    """Test Image with PDF inputs"""

    def test_image_stores_pdf_bytes(self, pdf_page: TestPdfPage) -> None:
        """Image can store PDF bytes"""
        img = Image(file_name=pdf_page.file_name, location=pdf_page.loc)
        img.image = pdf_page.pdf_bytes

        assert img.height in {pdf_page.np_array_shape[0], pdf_page.np_array_shape[0] + 1}
        assert img.width == pdf_page.np_array_shape[1]

    def test_pdf_bytes_property_can_be_set(self, pdf_page: TestPdfPage) -> None:
        """pdf_bytes property can be set"""
        img = Image(file_name=pdf_page.file_name)
        img.pdf_bytes = pdf_page.pdf_bytes

        assert img.pdf_bytes is not None
        assert isinstance(img.pdf_bytes, bytes)

    def test_pdf_bytes_initially_none(self) -> None:
        """pdf_bytes is initially None"""
        img = Image(file_name="test.pdf")

        assert img.pdf_bytes is None

    def test_image_from_pdf_creates_numpy_array(self, pdf_page: TestPdfPage) -> None:
        """Setting image from PDF bytes creates numpy array"""
        img = Image(file_name=pdf_page.file_name)
        img.image = pdf_page.pdf_bytes

        assert img.image is not None
        assert img.image.shape in (
            (pdf_page.np_array_shape[0], pdf_page.np_array_shape[1], 3),  # RGB channels
            (pdf_page.np_array_shape[0] + 1, pdf_page.np_array_shape[1], 3),  # RGB channels
        )
