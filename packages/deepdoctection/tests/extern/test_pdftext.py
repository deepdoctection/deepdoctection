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


import pytest

from deepdoctection.extern.pdftext import (
    PdfPlumberTextDetector,
    Pdfmium2TextDetector,
)


def test_pdfplumber_text_detector_predict_real_pdf(sample_pdf_bytes: bytes) -> None:
    pytest.importorskip("pdfplumber")

    det = PdfPlumberTextDetector()
    results = det.predict(sample_pdf_bytes)

    assert len(results) > 0
    assert results[0].class_name == "word"

    texts = [getattr(r, "text", "") for r in results]
    texts_lower = [t.lower() for t in texts]
    assert any("pdf" in t for t in texts_lower) or any("simple" in t for t in texts_lower)


def test_pdfplumber_text_detector_get_width_height_real_pdf(sample_pdf_bytes: bytes) -> None:
    pytest.importorskip("pdfplumber")

    det = PdfPlumberTextDetector()
    det.predict(sample_pdf_bytes)

    w, h = det.get_width_height(sample_pdf_bytes)
    assert w > 0
    assert h > 0


def test_pdfmium2_text_detector_predict_real_pdf(sample_pdf_bytes: bytes) -> None:
    pytest.importorskip("pypdfium2")

    det = Pdfmium2TextDetector()
    results = det.predict(sample_pdf_bytes)

    assert len(results) > 0
    assert results[0].class_name == "line"

    texts = [getattr(r, "text", "") for r in results]
    texts_lower = [t.lower() for t in texts]
    assert any("pdf" in t for t in texts_lower) or any("simple" in t for t in texts_lower)


def test_pdfmium2_text_detector_get_width_height_real_pdf(sample_pdf_bytes: bytes) -> None:
    pytest.importorskip("pypdfium2")

    det = Pdfmium2TextDetector()
    w, h = det.get_width_height(sample_pdf_bytes)
    assert w > 0
    assert h > 0
