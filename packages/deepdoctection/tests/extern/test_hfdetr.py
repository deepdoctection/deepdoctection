# -*- coding: utf-8 -*-
# File: test_hfdetr.py

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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from dd_core.utils import get_torch_device
from dd_core.utils.file_utils import pytorch_available, transformers_available
from dd_core.utils.object_types import LayoutType
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.hfdetr import HFDetrDerivedDetector

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available()),
    reason="Requires PyTorch and Transformers installed",
)


@REQUIRES_PT_AND_TR
def test_hfdetr_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:

    dummy_config = SimpleNamespace(
        architectures=["TableTransformerForObjectDetection"],
        threshold=0.1,
        nms_threshold=0.05,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_config",
        MagicMock(return_value=dummy_config),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_pre_processor",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_predict(np_img, predictor, feature_extractor, device, threshold, nms_threshold):
        # Note: class_id returned by DETR is zero-based; mapping in detector will shift +1 internally
        return [
            DetectionResult(box=[0, 0, 10, 10], class_id=0, score=0.95),
            DetectionResult(box=[20, 20, 40, 40], class_id=0, score=0.85),
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.detr_predict_image",
        MagicMock(side_effect=_fake_predict),
        raising=True,
    )

    categories = {1: LayoutType.TABLE}

    det = HFDetrDerivedDetector(
        path_config_json="dummy_config.json",
        path_weights="dummy_weights.bin",
        path_feature_extractor_config_json="dummy_fe.json",
        categories=categories,
        device=get_torch_device("cpu"),
    )

    np_image = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    results = det.predict(np_image)

    # Expect class_id shifted to 1 and class_name set properly
    assert len(results) == 2
    assert results[0].class_id == 1
    assert results[0].class_name == "table"
    assert results[0].score > 0.9


@REQUIRES_PT_AND_TR
def test_hfdetr_category_filtering(monkeypatch: pytest.MonkeyPatch) -> None:

    dummy_config = SimpleNamespace(
        architectures=["TableTransformerForObjectDetection"],
        threshold=0.1,
        nms_threshold=0.05,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_config",
        MagicMock(return_value=dummy_config),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_pre_processor",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    # Return two detections with class_id=0 (will map to id 1) and one default type (None)
    def _fake_predict(np_img, predictor, feature_extractor, device, threshold, nms_threshold):
        return [
            DetectionResult(box=[0, 0, 10, 10], class_id=0, score=0.9),
            DetectionResult(box=[10, 10, 20, 20], class_id=None, score=0.7),
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.detr_predict_image",
        MagicMock(side_effect=_fake_predict),
        raising=True,
    )

    categories = {1: LayoutType.TABLE}
    # Filter out "table" to test category filtering
    det = HFDetrDerivedDetector(
        "dummy_config.json",
        "dummy_weights.bin",
        "dummy_fe.json",
        categories,
        "cpu",
        filter_categories=[LayoutType.TABLE],
    )

    np_image = (np.random.rand(16, 16, 3) * 255).astype("uint8")
    results = det.predict(np_image)

    # All results filtered out due to category filter
    assert len(results) == 0


@REQUIRES_PT_AND_TR
def test_hfdetr_clear_model(monkeypatch: pytest.MonkeyPatch) -> None:

    dummy_config = SimpleNamespace(
        architectures=["TableTransformerForObjectDetection"],
        threshold=0.1,
        nms_threshold=0.05,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_config",
        MagicMock(return_value=dummy_config),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_pre_processor",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )
    # Minimal predictor call
    monkeypatch.setattr(
        "deepdoctection.extern.hfdetr.detr_predict_image",
        MagicMock(return_value=[]),
        raising=True,
    )

    det = HFDetrDerivedDetector("cfg.json", "w.bin", "fe.json", {1: LayoutType.TABLE}, "cpu")

    assert det.hf_detr_predictor is not None
    det.clear_model()
    assert det.hf_detr_predictor is None
