# -*- coding: utf-8 -*-
# File: test_nms.py

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

from dd_core.mapper import nms, pt_nms_image_annotations
from dd_core.utils.file_utils import pytorch_available

if pytorch_available():
    import torch


@pytest.mark.skipif(not pytorch_available(), reason="torch is not installed")
def test_batched_nms_uses_box_ops():

    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8, 0.75], dtype=torch.float32)
    idxs = torch.tensor([0, 0, 1], dtype=torch.int64)
    threshold = 0.5
    expected = nms.box_ops.batched_nms(boxes.float(), scores, idxs, threshold)
    kept = nms.batched_nms(boxes, scores, idxs, threshold)
    assert torch.equal(kept, expected)


@pytest.mark.skipif(not pytorch_available(), reason="torch is not installed")
def test_pt_nms_image_annotations_returns_expected_subset(monkeypatch, annotations):
    dp_image = annotations(True, True)
    anns = dp_image.get_annotation()
    output = pt_nms_image_annotations(anns, threshold=0.01)
    assert "51fca38d-b181-3ea2-9c97-7e265febcc86" not in output


@pytest.mark.skipif(not pytorch_available(), reason="torch is not installed")
def test_pt_nms_image_annotations_returns_expected_subset(monkeypatch, annotations):
    dp_image = annotations(True, True)
    anns = dp_image.get_annotation()
    output = pt_nms_image_annotations(anns, threshold=0.01, prio="title")
    assert "773eb5ea-1757-3f18-88f3-fdffebe771cc" not in output
