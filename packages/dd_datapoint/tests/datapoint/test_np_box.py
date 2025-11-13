# -*- coding: utf-8 -*-
# File: test_np_box.py

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
Testing functions from datapoint.box that operate on numpy boxes
"""


import numpy as np
import pytest

from dd_datapoint.datapoint.box import area, intersection, np_iou, ioa


@pytest.mark.parametrize(
    "boxes, expected",
    [
        # two boxes: areas 6 and 12
        (
            np.array([[0, 0, 2, 3], [1, 1, 4, 5]], dtype=np.float32),
            np.array([6.0, 12.0], dtype=np.float32),
        ),
        # absolute: (4-1)*(5-1)=12
        (np.array([[1, 1, 4, 5]], dtype=np.float32), np.array([12.0], dtype=np.float32)),
        # relative: (0.4-0.1)*(0.9-0.2)=0.21
        (np.array([[0.1, 0.2, 0.4, 0.9]], dtype=np.float32), np.array([0.21], dtype=np.float32)),
    ],
)
def test_area(boxes, expected):
    out = area(boxes)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "boxes1, boxes2, expected",
    [
        # boxes1 has two boxes, boxes2 has one -> expected is 2x1
        (
            np.array([[0, 0, 2, 3], [0, 0, 1, 1]], dtype=np.float32),
            np.array([[1, 1, 4, 5]], dtype=np.float32),
            np.array([[2.0], [0.0]], dtype=np.float32),
        ),
        # absolute: disjoint -> 0
        (
            np.array([[0, 0, 1, 1]], dtype=np.float32),
            np.array([[2, 2, 3, 3]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
        ),
        # relative overlap: [0.1,0.1,0.4,0.5] vs [0.3,0.2,0.8,0.6] -> 0.03
        (
            np.array([[0.1, 0.1, 0.4, 0.5]], dtype=np.float32),
            np.array([[0.3, 0.2, 0.8, 0.6]], dtype=np.float32),
            np.array([[0.03]], dtype=np.float32),
        ),
    ],
)
def test_intersection(boxes1, boxes2, expected):
    out = intersection(boxes1, boxes2)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "boxes1, boxes2, expected",
    [
        # absolute: IoU = 2 / (6+12-2) = 2/16 = 0.125
        (
            np.array([[0, 0, 2, 3]], dtype=np.float32),
            np.array([[1, 1, 4, 5]], dtype=np.float32),
            np.array([[0.125]], dtype=np.float32),
        ),
        # absolute: contained -> IoU = 1 / (4) = 0.25
        (
            np.array([[0, 0, 2, 2]], dtype=np.float32),
            np.array([[0, 0, 1, 1]], dtype=np.float32),
            np.array([[0.25]], dtype=np.float32),
        ),
        # relative: IoU = 0.03 / (0.12+0.20-0.03) ≈ 0.10344828
        (
            np.array([[0.1, 0.1, 0.4, 0.5]], dtype=np.float32),
            np.array([[0.3, 0.2, 0.8, 0.6]], dtype=np.float32),
            np.array([[0.10344828]], dtype=np.float32),
        ),
    ],
)
def test_np_iou(boxes1, boxes2, expected):
    out = np_iou(boxes1, boxes2)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "boxes1, boxes2, expected",
    [
        # absolute: IOA = inter/area(box2) = 2/12 = 1/6
        (
            np.array([[0, 0, 2, 3]], dtype=np.float32),
            np.array([[1, 1, 4, 5]], dtype=np.float32),
            np.array([[1.0 / 6.0]], dtype=np.float32),
        ),
        # absolute: contained -> IOA = 1/1 = 1
        (
            np.array([[0, 0, 2, 2]], dtype=np.float32),
            np.array([[0, 0, 1, 1]], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
        ),
        # relative: IOA = 0.03 / 0.20 = 0.15
        (
            np.array([[0.1, 0.1, 0.4, 0.5]], dtype=np.float32),
            np.array([[0.3, 0.2, 0.8, 0.6]], dtype=np.float32),
            np.array([[0.15]], dtype=np.float32),
        ),
    ],
)
def test_ioa(boxes1, boxes2, expected):
    out = ioa(boxes1, boxes2)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6)