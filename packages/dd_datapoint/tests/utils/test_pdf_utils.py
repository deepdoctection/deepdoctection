# -*- coding: utf-8 -*-
# File: test_pdf_utils.py

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
Testing the module utils.pdf_utils
"""

from pathlib import Path

import pytest
from numpy import uint8

from dd_datapoint.utils import file_utils as fu
from dd_datapoint.utils.pdf_utils import (
    PDFStreamer,
    get_pdf_file_reader,
    load_bytes_from_pdf_file,
    pdf_to_np_array_pdfmium,
    pdf_to_np_array_poppler,
)




POPPLER_AVAILABLE = bool(
    fu.pdf_to_ppm_available() or fu.pdf_to_cairo_available() or fu.get_poppler_version()
)

class TestGetPdfFileReader:
    """Test get_pdf_file_reader"""

    @staticmethod
    def test_get_pdf_file_reader_from_path(pdf_file_path_two_pages: Path) -> None:
        """Test reading PDF from path"""
        reader = get_pdf_file_reader(pdf_file_path_two_pages)
        assert reader is not None
        assert len(reader.pages) == 2

    @staticmethod
    def test_get_pdf_file_reader_from_bytes(pdf_bytes: bytes) -> None:
        """Test reading PDF from bytes"""
        if pdf_bytes:
            reader = get_pdf_file_reader(pdf_bytes)
            assert reader is not None



class TestPDFStreamer:
    """Test PDFStreamer class"""

    @staticmethod
    @pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
    def test_pdf_streamer_length(pdf_file_path_two_pages: Path) -> None:
        """Test PDFStreamer returns correct number of pages"""

        streamer = PDFStreamer(pdf_file_path_two_pages)
        assert len(streamer) > 0
        streamer.close()

    @staticmethod
    @pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
    def test_pdf_streamer_getitem(pdf_file_path_two_pages: Path) -> None:
        """Test PDFStreamer __getitem__ returns bytes"""

        streamer = PDFStreamer(pdf_file_path_two_pages)
        page_bytes = streamer[0]
        assert isinstance(page_bytes, bytes)
        assert len(page_bytes) > 0
        streamer.close()


class TestPdfToNpArrayPoppler:
    """Test pdf_to_np_array_poppler"""

    @staticmethod
    @pytest.mark.skipif(not POPPLER_AVAILABLE, reason="Poppler is not installed")
    @pytest.mark.parametrize("dpi", [72, 150])
    def test_pdf_to_np_array_poppler_with_dpi(pdf_bytes: bytes, dpi: int) -> None:
        """Test converting PDF to numpy array using Poppler with different DPI"""

        np_array = pdf_to_np_array_poppler(pdf_bytes, dpi=dpi)
        assert np_array.dtype == uint8
        assert len(np_array.shape) == 3


class TestPdfToNpArrayPdfmium:
    """Test pdf_to_np_array_pdfmium"""

    @staticmethod
    @pytest.mark.skipif(not fu.pypdfium2_available(), reason="pypdfium2 is not installed")
    @pytest.mark.parametrize("dpi", [72, 150])
    def test_pdf_to_np_array_pdfmium_with_dpi(pdf_bytes: bytes, dpi: int) -> None:
        """Test converting PDF to numpy array using pdfium with different DPI"""

        np_array = pdf_to_np_array_pdfmium(pdf_bytes, dpi=dpi)
        assert np_array.dtype == uint8
        assert len(np_array.shape) == 3


class TestLoadBytesFromPdfFile:
    """Test load_bytes_from_pdf_file"""

    @staticmethod
    @pytest.mark.parametrize("page_number", [0])
    @pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
    def test_load_bytes_from_pdf_file(pdf_file_path_two_pages: Path, page_number: int) -> None:
        """Test loading bytes from PDF file"""

        pdf_bytes = load_bytes_from_pdf_file(pdf_file_path_two_pages, page_number=page_number)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

