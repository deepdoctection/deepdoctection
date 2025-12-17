# -*- coding: utf-8 -*-
# File: test_tessocr.py

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
Unit tests for validating the `TesseractOcrDetector` functionality.

This file contains a series of test cases that focus on verifying the behavior of the
`TesseractOcrDetector` class, including text prediction from images, handling of different
configuration options, language mappings, and requirements retrieval.

The tests use Python's `pytest` framework and the `monkeypatch` feature for dependency isolation.
"""

from typing import Any

import pytest
from numpy.typing import NDArray

from dd_core.utils.env_info import SETTINGS
from dd_core.utils.object_types import Languages, LayoutType
from dd_core.utils.file_utils import tesseract_available
from deepdoctection.extern.tessocr import TesseractOcrDetector

REQUIRES_TESSERACT = pytest.mark.skipif(
    not tesseract_available(),
    reason="Requires Poppler or pypdfium2 installed",
)

@REQUIRES_TESSERACT
def test_tesseract_ocr_predict_words_basic(monkeypatch: pytest.MonkeyPatch, sample_np_img: NDArray[Any]) -> None:
    """test tesseract ocr predict words basic"""
    # Mock the Tesseract data extraction to avoid any subprocess calls
    fake_rows = {
        "left": [1, 10],
        "top": [2, 20],
        "width": [9, 20],
        "height": [10, 20],
        "conf": [85, 90],
        "text": ["Hello", "PDF"],
        "block_num": [0, 0],
        "line_num": [0, 1],
    }
    monkeypatch.setattr(
        "deepdoctection.extern.tessocr.image_to_dict",
        lambda img, lang, cfg: fake_rows,
        raising=True,
    )

    det = TesseractOcrDetector(SETTINGS.CONF_TESSERACT_SRC)
    results = det.predict(sample_np_img)

    assert len(results) == 2
    assert all(r.class_name == LayoutType.WORD for r in results)
    texts = [r.text.lower() for r in results if r.text is not None]
    assert any("pdf" in t for t in texts) or any("hello" in t for t in texts)


@REQUIRES_TESSERACT
def test_tesseract_ocr_predict_lines_enabled(monkeypatch: pytest.MonkeyPatch, sample_np_img: NDArray[Any]) -> None:
    """test tesseract ocr predict lines enabled"""
    # Provide multiple words on two lines so line grouping can be formed
    fake_rows = {
        "left": [1, 12, 5, 15],
        "top": [2, 2, 22, 22],
        "width": [10, 10, 8, 12],
        "height": [10, 10, 12, 12],
        "conf": [80, 88, 92, 93],
        "text": ["A", "Simple", "PDF", "File"],
        "block_num": [0, 0, 0, 0],
        "line_num": [0, 0, 1, 1],
    }
    monkeypatch.setattr(
        "deepdoctection.extern.tessocr.image_to_dict",
        lambda img, lang, cfg: fake_rows,
        raising=True,
    )

    det = TesseractOcrDetector(SETTINGS.CONF_TESSERACT_SRC)
    # Ensure lines are generated regardless of the YAML default
    det.config.freeze(freezed=False)
    det.config.LINES = True
    det.config.freeze(freezed=True)

    results = det.predict(sample_np_img)

    # Expect original words plus 2 aggregated lines
    assert len(results) >= 4
    assert any(r.class_name == LayoutType.LINE for r in results)
    line_texts = [r.text.lower() for r in results if r.class_name == LayoutType.LINE and r.text is not None]
    assert any("a simple" in lt for lt in line_texts) or any("pdf file" in lt for lt in line_texts)


@REQUIRES_TESSERACT
def test_tesseract_ocr_get_category_names_toggle_lines() -> None:
    """test tesseract ocr get category names toggle lines"""
    det = TesseractOcrDetector(SETTINGS.CONF_TESSERACT_SRC, config_overwrite=["LINES=False"])
    names = det.get_category_names()
    assert names == (LayoutType.WORD,)

    det = TesseractOcrDetector(SETTINGS.CONF_TESSERACT_SRC, config_overwrite=["LINES=True"])
    names2 = det.get_category_names()
    assert names2 == (LayoutType.WORD, LayoutType.LINE)


@REQUIRES_TESSERACT
def test_tesseract_ocr_set_language_mapping() -> None:
    """test tesseract ocr set language mapping"""
    det = TesseractOcrDetector(SETTINGS.CONF_TESSERACT_SRC)
    # Map pseudo language code to tess code via `_LANG_CODE_TO_TESS_LANG_CODE`
    det.set_language(Languages.GERMAN)  # uses `ObjectTypes` enum path; `nn` -> 'eng'
    assert det.config.LANGUAGES == "deu"


@REQUIRES_TESSERACT
def test_tesseract_ocr_get_requirements() -> None:
    """test tesseract ocr get requirements"""
    det = TesseractOcrDetector(SETTINGS.CONF_TESSERACT_SRC)
    reqs = det.get_requirements()
    assert isinstance(reqs, list)
    assert len(reqs) >= 1
