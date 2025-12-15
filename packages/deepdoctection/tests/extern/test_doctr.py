# -*- coding: utf-8 -*-
# File: test_doctr.py

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
Unit tests for verifying deepdoctection's integration with python-doctr.

This module contains test cases to validate the functionality of DoctrTextlineDetector,
DoctrTextRecognizer, and DocTrRotationTransformer components from the deepdoctection package,
mocking their external dependencies to ensure tests run predictably.

Tests in this module:
- `test_doctr_textline_detector_predict_basic`: Tests text line detection using a mock model and
  prediction function.
- `test_doctr_text_recognizer_predict_basic`: Tests text recognition functionality using a mock
  model and custom logic for text recognition outputs.
- `test_doctr_rotation_transformer_predict_and_transform`: Validates image rotation predictions
  and transformations.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dd_core.utils.file_utils import doctr_available, pytorch_available
from dd_core.utils.object_types import LayoutType
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.doctrocr import (
    DocTrRotationTransformer,
    DoctrTextlineDetector,
    DoctrTextRecognizer,
)

REQUIRES_PT_AND_DOCTR = pytest.mark.skipif(
    not (pytorch_available() and doctr_available()),
    reason="Requires PyTorch and python-doctr installed",
)


@REQUIRES_PT_AND_DOCTR
def test_doctr_textline_detector_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """test text line detection using mocked model and prediction function."""
    # Mock model construction (no real weights/model)
    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.DoctrTextlineDetector.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    # Mock prediction helper
    def _fake_predict(np_img, predictor):  # type: ignore # pylint:disable=W0613
        return [
            DetectionResult(box=[0, 0, 10, 10], class_id=1, score=0.9, class_name=LayoutType.WORD),
            DetectionResult(box=[20, 20, 40, 40], class_id=1, score=0.8, class_name=LayoutType.WORD),
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.doctr_predict_text_lines",
        MagicMock(side_effect=_fake_predict),
        raising=True,
    )

    categories = {1: LayoutType.WORD}
    det = DoctrTextlineDetector("db_resnet50", "dummy.pt", categories, "cpu")

    np_image = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    results = det.predict(np_image)

    assert len(results) == 2
    assert results[0].class_name == "word"


@REQUIRES_PT_AND_DOCTR
def test_doctr_text_recognizer_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """test text recognition using mocked model and custom prediction logic."""
    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.DoctrTextRecognizer.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_recognize(inputs, predictor):  # type: ignore
        return [
            DetectionResult(score=0.7, text="Foo", uuid=inputs[0][0]),
            DetectionResult(score=0.8, text="Bar", uuid=inputs[1][0]),
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.doctr_predict_text",
        MagicMock(side_effect=_fake_recognize),
        raising=True,
    )

    rec = DoctrTextRecognizer("crnn_vgg16_bn", "dummy.pt", "cpu")
    batch = [
        ("id1", (np.random.rand(16, 64, 3) * 255).astype("uint8")),
        ("id2", (np.random.rand(16, 64, 3) * 255).astype("uint8")),
    ]

    results = rec.predict(batch)
    assert len(results) == 2
    assert results[0].text == "Foo"
    assert results[1].text == "Bar"


@REQUIRES_PT_AND_DOCTR
def test_doctr_rotation_transformer_predict_and_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """test image rotation predictions and transformations."""
    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.estimate_orientation",
        MagicMock(return_value=90.0),
        raising=True,
    )
    transformer = DocTrRotationTransformer()
    np_image = (np.random.rand(20, 30, 3) * 255).astype("uint8")

    spec = transformer.predict(np_image)
    assert spec.angle == 90.0

    rotated = transformer.transform_image(np_image, spec)
    # 90 degree rotation swaps dimensions
    assert rotated.shape[0] == np_image.shape[1]
    assert rotated.shape[1] == np_image.shape[0]
