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


import pytest

from dd_core.datapoint import convert_pdf_bytes_to_np_array_v2
from dd_core.utils import file_utils as fu

import shared_test_utils as stu

PDFIUM_AVAILABLE = True
try:
    import pypdfium2  # noqa: F401
except ImportError:
    PDFIUM_AVAILABLE = False

POPLER_AVAILABLE = bool(
    fu.pdf_to_ppm_available() or fu.pdf_to_cairo_available() or fu.get_poppler_version()
)


def test_importerror_when_pypdf_missing(monkeypatch: pytest.MonkeyPatch, pdf_page: stu.TestPdfPage) -> None:
    """
    When dpi, width and height are not given, PdfReader is used to infer size.
    If pypdf is not installed, ImportError must be raised.
    """
    monkeypatch.setenv("USE_DD_PDFIUM", "False")
    # Simulate that pypdf was not imported successfully in the module
    monkeypatch.delattr("dd_core.datapoint.convert.PdfReader", raising=False)

    with pytest.raises(ImportError, match="pypdf is not installed"):
        convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=None)


@pytest.mark.skipif(not PDFIUM_AVAILABLE, reason="pypdfium2 is not installed")
def test_pdfium_with_dpi_200_fixed_shape(monkeypatch: pytest.MonkeyPatch, pdf_page: stu.TestPdfPage) -> None:
    """
    With USE_DD_PDFIUM=True and dpi=200 there is a predetermined shape.
    """
    monkeypatch.setenv("USE_DD_PDFIUM", "True")
    np_array = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=200)
    assert np_array.shape == (2200, 1700, 3)


@pytest.mark.skipif(not PDFIUM_AVAILABLE, reason="pypdfium2 is not installed")
def test_pdfium_with_dpi_and_size_falls_back_to_dpi(monkeypatch: pytest.MonkeyPatch, pdf_page: stu.TestPdfPage) -> None:
    """
    If dpi is provided together with width/height under PDFium, width/height are ignored (fallback to dpi).
    """
    monkeypatch.setenv("USE_DD_PDFIUM", "True")
    dpi = 150
    with_size = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=dpi, width=800, height=600)
    without_size = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=dpi)
    assert with_size.shape == without_size.shape


@pytest.mark.skipif(not PDFIUM_AVAILABLE, reason="pypdfium2 is not installed")
def test_pdfium_without_dpi_but_only_size_raises_valueerror(monkeypatch: pytest.MonkeyPatch, pdf_page: stu.TestPdfPage) -> None:
    """
    Under PDFium, providing only width/height must raise ValueError('dpi must be provided.').
    """
    monkeypatch.setenv("USE_DD_PDFIUM", "True")
    with pytest.raises(ValueError, match="dpi must be provided"):
        convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=None, width=800, height=600)


@pytest.mark.skipif(not POPLER_AVAILABLE, reason="Poppler is not available")
def test_poppler_with_dpi_300(monkeypatch: pytest.MonkeyPatch, pdf_page: stu.TestPdfPage) -> None:
    """
    With Poppler (USE_DD_PDFIUM=False) and dpi=300, expect the known shape from fixture.
    """
    monkeypatch.setenv("USE_DD_PDFIUM", "False")
    np_array = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=300)
    assert np_array.shape == pdf_page.np_array_shape


@pytest.mark.skipif(not POPLER_AVAILABLE, reason="Poppler is not available")
def test_poppler_with_explicit_size(monkeypatch: pytest.MonkeyPatch, pdf_page: stu.TestPdfPage) -> None:
    """
    With Poppler and explicit width/height, expect exact (height,width,3).
    Use values less than 1000.
    """
    monkeypatch.setenv("USE_DD_PDFIUM", "False")
    width, height = 800, 600
    np_array = convert_pdf_bytes_to_np_array_v2(pdf_page.pdf_bytes, dpi=None, width=width, height=height)
    assert np_array.shape == (height, width, 3)
