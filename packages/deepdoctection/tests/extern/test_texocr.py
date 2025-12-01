# -*- coding: utf-8 -*-
# File: test_texocr.py

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
from unittest.mock import MagicMock

from dd_core.utils.types import PixelValues, JsonDict
from deepdoctection.extern.texocr import TextractOcrDetector


def test_textract_ocr_predict_words_basic(monkeypatch: pytest.MonkeyPatch,
                                          sample_np_img: PixelValues,
                                          textract_json) -> None:
    pytest.importorskip("boto3")

    monkeypatch.setattr("deepdoctection.extern.texocr.get_boto3_requirement", lambda: ("boto3", True, ""))
    monkeypatch.setattr("deepdoctection.extern.texocr.boto3", MagicMock())
    det = TextractOcrDetector()
    det.client.detect_document_text = MagicMock(return_value=textract_json)

    results = det.predict(sample_np_img)

    assert isinstance(results, list)
    assert len(results) == 12
    assert any("consolidated" in r.text.lower() for r in results if hasattr(r, "text"))

