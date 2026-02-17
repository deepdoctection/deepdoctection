# -*- coding: utf-8 -*-
# File: test_azurediocr.py

# Copyright 2026 Dr. Janis Meyer and Idan Hemed. All rights reserved.
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
A test module for verifying basic functionality of Azure Document Intelligence OCR integration.

This module contains unit tests to validate the behavior of the AzureDocIntelOcrDetector class.
It ensures the detector correctly predicts OCR results and verifies its functionality using mocked
dependencies and predefined test data.
"""

from unittest.mock import MagicMock

import pytest

from dd_core.utils.types import JsonDict, PixelValues
from deepdoctection.extern.azurediocr import AzureDocIntelOcrDetector


def test_azure_di_ocr_predict_words_basic(
    monkeypatch: pytest.MonkeyPatch, sample_np_img: PixelValues, azure_di_json: JsonDict
) -> None:
    """test azure document intelligence ocr predict words basic"""
    pytest.importorskip("azure.ai.documentintelligence")

    monkeypatch.setattr("deepdoctection.extern.azurediocr.get_azure_di_requirement", lambda: ("azure-ai-documentintelligence", True, ""))
    monkeypatch.setattr("deepdoctection.extern.azurediocr.DocumentIntelligenceClient", MagicMock())
    monkeypatch.setattr("deepdoctection.extern.azurediocr.AzureKeyCredential", MagicMock())

    det = AzureDocIntelOcrDetector(endpoint="https://test.cognitiveservices.azure.com/", api_key="fake-key")

    # Mock the poller pattern
    mock_poller = MagicMock()
    mock_poller.result.return_value = azure_di_json
    det.client.begin_analyze_document = MagicMock(return_value=mock_poller)

    results = det.predict(sample_np_img)

    assert isinstance(results, list)
    assert len(results) == 15
    assert any("consolidated" in r.text.lower() for r in results if hasattr(r, "text") and r.text is not None)
