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


# file: deepdoctection/packages/deepdoctection/tests/extern/test_doctrocr_integration.py

import numpy as np
import pytest
from unittest.mock import MagicMock

from dd_core.utils.file_utils import pytorch_available, doctr_available
from dd_core.utils.object_types import LayoutType
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.doctrocr import (
    DoctrTextlineDetector,
    DoctrTextRecognizer,
    DocTrRotationTransformer,
)


REQUIRES_PT_AND_DOCTR = pytest.mark.skipif(
    not (pytorch_available() and doctr_available()),
    reason="Requires PyTorch and python-doctr installed",
)


@REQUIRES_PT_AND_DOCTR
def test_doctr_textline_detector_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock model construction (no real weights/model)
    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.DoctrTextlineDetector.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )
    # Mock prediction helper
    def _fake_predict(np_img, predictor):
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
    monkeypatch.setattr(
        "deepdoctection.extern.doctrocr.DoctrTextRecognizer.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_recognize(inputs, predictor):
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
    batch = [("id1", (np.random.rand(16, 64, 3) * 255).astype("uint8")),
             ("id2", (np.random.rand(16, 64, 3) * 255).astype("uint8"))]

    results = rec.predict(batch)
    assert len(results) == 2
    assert results[0].text == "Foo"
    assert results[1].text == "Bar"


@REQUIRES_PT_AND_DOCTR
def test_doctr_rotation_transformer_predict_and_transform(monkeypatch: pytest.MonkeyPatch) -> None:
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

