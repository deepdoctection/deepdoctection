# -*- coding: utf-8 -*-
# File: test_pdftext.py

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
Unit tests for text detection functionality in PDF processing.

This module contains unit tests to validate the behavior and correctness of
PDF text detection implementations using different backends such as pdfplumber
and pypdfium2. Tests include evaluating text extraction and metadata
retrieval.

"""

import pytest

from deepdoctection.extern.pdftext import (
    Pdfmium2TextDetector,
    PdfPlumberTextDetector,
)


def test_pdfplumber_text_detector_predict_real_pdf(sample_pdf_bytes: bytes) -> None:
    """test pdfplumber text detector predict real pdf"""
    pytest.importorskip("pdfplumber")

    det = PdfPlumberTextDetector()
    results = det.predict(sample_pdf_bytes)

    assert len(results) > 0
    assert results[0].class_name == "word"

    texts = [getattr(r, "text", "") for r in results]
    texts_lower = [t.lower() for t in texts]
    assert any("pdf" in t for t in texts_lower) or any("simple" in t for t in texts_lower)


def test_pdfplumber_text_detector_get_width_height_real_pdf(sample_pdf_bytes: bytes) -> None:
    """test pdfplumber text detector get width height real pdf"""
    pytest.importorskip("pdfplumber")

    det = PdfPlumberTextDetector()
    det.predict(sample_pdf_bytes)

    w, h = det.get_width_height(sample_pdf_bytes)
    assert w > 0
    assert h > 0


def test_pdfmium2_text_detector_predict_real_pdf(sample_pdf_bytes: bytes) -> None:
    """test pdfmium2 text detector predict real pdf"""
    pytest.importorskip("pypdfium2")

    det = Pdfmium2TextDetector()
    results = det.predict(sample_pdf_bytes)

    assert len(results) > 0
    assert results[0].class_name == "line"

    texts = [getattr(r, "text", "") for r in results]
    texts_lower = [t.lower() for t in texts]
    assert any("pdf" in t for t in texts_lower) or any("simple" in t for t in texts_lower)


def test_pdfmium2_text_detector_get_width_height_real_pdf(sample_pdf_bytes: bytes) -> None:
    """test pdfmium2 text detector get width height real pdf"""
    pytest.importorskip("pypdfium2")

    det = Pdfmium2TextDetector()
    w, h = det.get_width_height(sample_pdf_bytes)
    assert w > 0
    assert h > 0
