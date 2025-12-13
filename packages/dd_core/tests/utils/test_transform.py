# -*- coding: utf-8 -*-
# File: test_transform.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Testing the module utils.transform
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pytest

from dd_core.utils.transform import (
    InferenceResize,
    PadTransform,
    ResizeTransform,
    RotationTransform,
    box_to_point4,
    normalize_image,
    pad_image,
    point4_to_box,
)


class TestBoxToPoint4:
    """Test box_to_point4"""

    @staticmethod
    @pytest.mark.parametrize(
        "boxes,expected_shape",
        [
            (np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32), (4, 2)),
            (np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]], dtype=np.float32), (8, 2)),
        ],
    )
    def test_box_to_point4_shape(boxes: NDArray[np.float32], expected_shape: tuple[int,int]) -> None:
        """Test box_to_point4 returns correct shape"""
        result = box_to_point4(boxes)
        assert result.shape == expected_shape


class TestPoint4ToBox:
    """Test point4_to_box"""

    @staticmethod
    @pytest.mark.parametrize(
        "points,expected_shape",
        [
            (np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32), (1, 4)),
            (
                np.array(
                    [
                        [0.0, 0.0],
                        [10.0, 0.0],
                        [10.0, 10.0],
                        [0.0, 10.0],
                        [5.0, 5.0],
                        [15.0, 5.0],
                        [15.0, 15.0],
                        [5.0, 15.0],
                    ],
                    dtype=np.float32,
                ),
                (2, 4),
            ),
        ],
    )
    def test_point4_to_box_shape(points: NDArray[np.float32], expected_shape: tuple[int,int]) -> None:
        """Test point4_to_box returns correct shape"""
        result = point4_to_box(points)
        assert result.shape == expected_shape


class TestResizeTransform:
    """Test ResizeTransform"""

    @staticmethod
    @pytest.mark.parametrize("h,w,new_h,new_w", [(100, 150, 50, 75), (100, 150, 200, 300)])
    def test_resize_transform_apply_image(np_image: NDArray[np.uint8], h: int, w: int,
                                          new_h: int, new_w: int) -> None:
        """Test ResizeTransform.apply_image"""
        transform = ResizeTransform(h, w, new_h, new_w, "VIZ")
        result = transform.apply_image(np_image)
        assert result.shape[:2] == (new_h, new_w)

    @staticmethod
    @pytest.mark.parametrize("h,w,new_h,new_w", [(100, 150, 50, 75)])
    def test_resize_transform_apply_coords(coords: NDArray[np.float32], h: int, w: int, new_h: int, new_w: int) -> None:
        """Test ResizeTransform.apply_coords"""
        transform = ResizeTransform(h, w, new_h, new_w, "VIZ")
        result = transform.apply_coords(coords.copy())
        assert result.shape == coords.shape
        # Check scaling is applied
        assert not np.array_equal(result, coords)


class TestInferenceResize:
    """Test InferenceResize"""

    @staticmethod
    @pytest.mark.parametrize("short_edge,max_size", [(600, 1000), (800, 1333)])
    def test_inference_resize_get_transform(np_image: NDArray[np.uint8], short_edge: int, max_size: int) -> None:
        """Test InferenceResize.get_transform"""
        resize = InferenceResize(short_edge, max_size, "VIZ")
        transform = resize.get_transform(np_image)
        assert isinstance(transform, ResizeTransform)


class TestNormalizeImage:
    """Test normalize_image"""

    @staticmethod
    @pytest.mark.parametrize(
        "pixel_mean,pixel_std",
        [
            (np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)),
        ],
    )
    def test_normalize_image(np_image: NDArray[np.uint8], pixel_mean: NDArray[np.float32],
                             pixel_std: NDArray[np.float32]) -> None:
        """Test normalize_image"""
        result = normalize_image(np_image, pixel_mean, pixel_std)
        assert result.shape == np_image.shape
        assert not np.array_equal(result, np_image)


class TestPadImage:
    """Test pad_image"""

    @staticmethod
    @pytest.mark.parametrize("top,right,bottom,left", [(10, 20, 10, 20), (5, 5, 5, 5)])
    def test_pad_image(np_image: NDArray[np.uint8], top: int, right: int, bottom: int, left: int) -> None:
        """Test pad_image"""
        result = pad_image(np_image, top, right, bottom, left)
        expected_h = np_image.shape[0] + top + bottom
        expected_w = np_image.shape[1] + left + right
        assert result.shape == (expected_h, expected_w, np_image.shape[2])


class TestPadTransform:
    """Test PadTransform"""

    @staticmethod
    @pytest.mark.parametrize("top,right,bottom,left", [(10, 20, 10, 20)])
    def test_pad_transform_apply_image(np_image: NDArray[np.uint8], top: int, right: int, bottom: int, left: int) -> None:
        """Test PadTransform.apply_image"""
        transform = PadTransform(top, right, bottom, left)
        result = transform.apply_image(np_image)
        expected_h = np_image.shape[0] + top + bottom
        expected_w = np_image.shape[1] + left + right
        assert result.shape == (expected_h, expected_w, np_image.shape[2])

    @staticmethod
    @pytest.mark.parametrize("top,right,bottom,left", [(10, 20, 10, 20)])
    def test_pad_transform_apply_coords(coords: NDArray[np.float32], top: int, right: int, bottom: int, left: int) -> None:
        """Test PadTransform.apply_coords"""
        transform = PadTransform(top, right, bottom, left)
        result = transform.apply_coords(coords.copy())
        assert result.shape == coords.shape

        assert result[0, 0] == coords[0, 0] + left
        assert result[0, 1] == coords[0, 1] + top


class TestRotationTransform:
    """Test RotationTransform"""

    @staticmethod
    @pytest.mark.parametrize("angle", [90, 180, 270])
    def test_rotation_transform_apply_image(np_image: NDArray[np.uint8], angle: Literal[90, 180, 270]) -> None:
        """Test RotationTransform.apply_image"""
        transform = RotationTransform(angle)
        result = transform.apply_image(np_image)
        assert result is not None
        if angle in [90, 270]:
            # Dimensions should swap
            assert result.shape[0] == np_image.shape[1]
            assert result.shape[1] == np_image.shape[0]
        else:
            # Dimensions should remain the same
            assert result.shape[:2] == np_image.shape[:2]

    @staticmethod
    @pytest.mark.parametrize("angle", [90, 180])
    def test_rotation_transform_apply_coords(coords: NDArray[np.float32], angle: Literal[90, 180]) -> None:
        """Test RotationTransform.apply_coords"""
        transform = RotationTransform(angle)
        transform.set_image_width(150)
        transform.set_image_height(100)
        result = transform.apply_coords(coords.copy())
        assert result.shape == coords.shape
        assert not np.array_equal(result, coords)
