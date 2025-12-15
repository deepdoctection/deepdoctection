# -*- coding: utf-8 -*-
# File: xxx.py

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
This module contains unit tests for object detection models based on Detectron2 frameworks.

The module is responsible for verifying the functionality of object detection models, focusing
on their ability to load configurations, map prediction outputs to categories, and test basic
instances of object mappings.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from dd_core.utils.file_utils import detectron2_available, pytorch_available
from dd_core.utils.object_types import LayoutType, ObjectTypes
from deepdoctection.extern.d2detect import D2FrcnnDetector, D2FrcnnTracingDetector

if pytorch_available():
    import torch

if detectron2_available():
    from detectron2.structures import Instances, Boxes


REQUIRES_PT_AND_D2 = pytest.mark.skipif(
    not (pytorch_available() and detectron2_available()),
    reason="Requires PyTorch and Detectron2 installed",
)


def _stub_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        NMS_THRESH_CLASS_AGNOSTIC=0.3,
        INPUT=SimpleNamespace(MIN_SIZE_TEST=480, MAX_SIZE_TEST=800),
        MODEL=SimpleNamespace(META_ARCHITECTURE="GeneralizedRCNN"),
    )


def _get_mock_instances() -> List[List[Dict[str, Instances]]]:
    pred_boxes = Boxes(torch.tensor(  # pylint:disable=E0606
        [[1.0, 1.0, 5.0, 6.0], [10.0, 10.0, 12.0, 12.0]], dtype=torch.float32))
    scores = torch.tensor([0.93, 0.54], dtype=torch.float32)
    pred_classes = torch.tensor([0, 1], dtype=torch.uint8)
    inst = Instances((400, 600))
    inst.pred_boxes = pred_boxes
    inst.scores = scores
    inst.pred_classes = pred_classes
    return [[{"instances": inst}]]


@REQUIRES_PT_AND_D2
def test_d2_frcnn_predict_basic_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Unit: mock model build and load, verify predict() and category ID remapping.
    """
    # Monkeypatch constructor steps
    monkeypatch.setattr(
        "deepdoctection.extern.d2detect.D2FrcnnDetector._set_model",
        MagicMock(return_value=MagicMock),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.d2detect.D2FrcnnDetector._instantiate_d2_predictor",
        MagicMock(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.d2detect.D2FrcnnDetector._set_config",
        lambda *_, **__: _stub_cfg(),
        raising=True,
    )

    categories: Dict[int, ObjectTypes] = {1: LayoutType.FIGURE, 2: LayoutType.LIST, 3: LayoutType.TABLE}

    det = D2FrcnnDetector(
        path_yaml="dummy.yaml",
        path_weights="dummy.pt",
        categories=categories,
        device="cpu",
    )
    det.d2_predictor = MagicMock(side_effect=_get_mock_instances())

    np_image = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    results = det.predict(np_image)

    assert len(results) == 2

    assert results[0].class_id == 1
    assert results[1].class_id == 2
    assert results[0].class_name == "figure"
    assert results[1].class_name == "list"


@REQUIRES_PT_AND_D2
def test_d2_frcnn_tracing_predict_basic_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Unit: mock TorchScript loader, verify predict() and category ID remapping for tracing detector.
    """
    # Patch config loader
    monkeypatch.setattr(
        "deepdoctection.extern.d2detect.D2FrcnnTracingDetector._set_config",
        lambda *_, **__: _stub_cfg(),
        raising=True,
    )

    # Fake TorchScript callable returning (boxes, classes, scores, _)
    def _fake_ts_forward(_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        boxes = torch.tensor([[2.0, 2.0, 6.0, 6.0], [12.0, 12.0, 16.0, 16.0]], dtype=torch.float32)
        classes = torch.tensor([0, 1], dtype=torch.int64)
        scores = torch.tensor([0.95, 0.7], dtype=torch.float32)
        return boxes, classes, scores, None

    monkeypatch.setattr(
        "deepdoctection.extern.d2detect.D2FrcnnTracingDetector.get_wrapped_model",
        MagicMock(return_value=MagicMock(side_effect=_fake_ts_forward)),
        raising=True,
    )

    categories: Dict[int, ObjectTypes] = {1: LayoutType.FIGURE, 2: LayoutType.LIST, 3: LayoutType.TABLE}

    det = D2FrcnnTracingDetector(
        path_yaml="dummy.yaml",
        path_weights="dummy.ts",
        categories=categories,
        filter_categories=None,
    )

    np_image = (np.random.rand(40, 40, 3) * 255).astype("uint8")
    results = det.predict(np_image)

    assert len(results) == 2
    # Class IDs must be shifted by +1 compared to raw TS outputs
    assert results[0].class_id == 1
    assert results[1].class_id == 2
    assert results[0].class_name == "figure"
    assert results[1].class_name == "list"
