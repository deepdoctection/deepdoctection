# -*- coding: utf-8 -*-
# File: test_convert.py

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
Testing module datapoint.convert
"""
from deepdoctection.datapoint import convert_pdf_bytes_to_np_array_v2

from .conftest import TestPdfPage


def test_convert_pdf_bytes_to_np_array_v2(pdf_page: TestPdfPage) -> None:
    """
    testing convert_pdf_bytes_to_np_array_v2 returns a np.array of correct shape
    """

    # Act
    np_array = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes)

    # Assert
    assert np_array.shape == pdf_page.np_array_shape_default


def test_convert_pdf_bytes_to_np_array_v2_with_dpi_300(pdf_page: TestPdfPage) -> None:
    """
    testing convert_pdf_bytes_to_np_array_v2 returns a np.array of correct shape when dpi is set to 300
    """

    # Act
    np_array = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, 300)

    # Assert
    assert np_array.shape == pdf_page.np_array_shape
