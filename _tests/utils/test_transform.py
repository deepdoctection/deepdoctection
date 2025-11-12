# -*- coding: utf-8 -*-
# File: test_transform.py

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
Testing module utils.transform
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
from numpy import float32

from deepdoctection.utils.transform import PadTransform, ResizeTransform, RotationTransform
from deepdoctection.utils.types import PixelValues


class TestResizeTransform:
    """Test class for ResizeTransform"""

    def test_apply_image(self, np_image_large: PixelValues) -> None:
        """Test ResizeTransform.apply_image with large image"""
        # Setup
        h, w = np_image_large.shape[:2]
        transform = ResizeTransform(h=h, w=w, new_h=200, new_w=300, interp="VIZ")

        # Execute
        resized_image = transform.apply_image(np_image_large)

        # Assert
        assert resized_image.shape == (200, 300, 3)
        assert np.all(resized_image == 1.0)

    @pytest.mark.parametrize(
        "coords,h,w,new_h,new_w,expected",
        [
            (
                np.array(
                    [[100, 50], [200, 50], [200, 150], [100, 150], [300, 200], [500, 200], [500, 300], [300, 300]],
                    dtype=float32,
                ),
                400,
                600,
                200,
                300,
                np.array(
                    [[50, 25], [100, 25], [100, 75], [50, 75], [150, 100], [250, 100], [250, 150], [150, 150]],
                    dtype=float32,
                ),
            ),
            (
                np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float32),
                200,
                200,
                100,
                100,
                np.array([[25, 25], [75, 25], [75, 75], [25, 75]], dtype=float32),
            ),
        ],
    )
    def test_apply_coords(
        self, coords: npt.NDArray[np.float32], h: int, w: int, new_h: int, new_w: int, expected: npt.NDArray[np.float32]
    ) -> None:
        """Test ResizeTransform.apply_coords with parametrized bounding boxes"""
        # Setup
        transform = ResizeTransform(h=h, w=w, new_h=new_h, new_w=new_w, interp="VIZ")

        # Execute
        transformed_coords = transform.apply_coords(coords)

        # Assert
        np.testing.assert_almost_equal(transformed_coords, expected)

    @pytest.mark.parametrize(
        "coords,h,w,new_h,new_w,expected",
        [
            (
                np.array(
                    [[50, 25], [100, 25], [100, 75], [50, 75], [150, 100], [250, 100], [250, 150], [150, 150]],
                    dtype=float32,
                ),
                400,
                600,
                200,
                300,
                np.array(
                    [[100, 50], [200, 50], [200, 150], [100, 150], [300, 200], [500, 200], [500, 300], [300, 300]],
                    dtype=float32,
                ),
            ),
            (
                np.array([[25, 25], [75, 25], [75, 75], [25, 75]], dtype=float32),
                200,
                200,
                100,
                100,
                np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float32),
            ),
        ],
    )
    def test_inverse_apply_coords(
        self, coords: npt.NDArray[np.float32], h: int, w: int, new_h: int, new_w: int, expected: npt.NDArray[np.float32]
    ) -> None:
        """Test ResizeTransform.inverse_apply_coords with parametrized bounding boxes"""
        # Setup
        transform = ResizeTransform(h=h, w=w, new_h=new_h, new_w=new_w, interp="VIZ")

        # Execute
        restored_coords = transform.inverse_apply_coords(coords)

        # Assert
        np.testing.assert_almost_equal(restored_coords, expected)

    @pytest.mark.parametrize(
        "coords,h,w,new_h,new_w",
        [
            (
                np.array(
                    [[100, 50], [200, 50], [200, 150], [100, 150], [300, 200], [500, 200], [500, 300], [300, 300]],
                    dtype=float32,
                ),
                400,
                600,
                200,
                300,
            ),
            (np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float32), 200, 200, 100, 100),
        ],
    )
    def test_transform_inverse_transform_cycle(
        self, coords: npt.NDArray[np.float32], h: int, w: int, new_h: int, new_w: int
    ) -> None:
        """Test that applying transform followed by inverse_transform returns original coordinates"""
        # Setup
        transform = ResizeTransform(h=h, w=w, new_h=new_h, new_w=new_w, interp="VIZ")

        # Execute
        transformed = transform.apply_coords(coords)
        restored = transform.inverse_apply_coords(transformed)

        # Assert
        np.testing.assert_almost_equal(restored, coords)


class TestPadTransform:
    """Test class for PadTransform"""

    def test_apply_image(self, np_image_large: PixelValues) -> None:
        """Test PadTransform.apply_image with large image"""
        # Setup
        h, w = np_image_large.shape[:2]
        transform = PadTransform(pad_top=10, pad_right=20, pad_bottom=30, pad_left=40)

        # Execute
        padded_image = transform.apply_image(np_image_large)

        # Assert
        assert padded_image.shape == (h + 40, w + 60, 3)  # top+bottom=40, left+right=60
        assert np.all(padded_image[0:10, :, :] == 255)  # top padding
        assert np.all(padded_image[:, 0:40, :] == 255)  # left padding
        assert np.all(padded_image[h + 10 :, :, :] == 255)  # bottom padding
        assert np.all(padded_image[:, w + 40 :, :] == 255)  # right padding

    @pytest.mark.parametrize(
        "coords,top,right,bottom,left,expected",
        [
            (
                np.array([[100, 50, 200, 150]], dtype=float32),
                10,
                20,
                30,
                40,
                np.array([[140, 60, 240, 160]], dtype=float32),
            ),
            (
                np.array([[50, 50, 150, 150]], dtype=float32),
                5,
                15,
                25,
                10,
                np.array([[60, 55, 160, 155]], dtype=float32),
            ),
        ],
    )
    def test_apply_coords(
        self,
        coords: npt.NDArray[np.float32],
        top: int,
        right: int,
        bottom: int,
        left: int,
        expected: npt.NDArray[np.float32],
    ) -> None:
        """Test PadTransform.apply_coords with parametrized bounding boxes in xyxy format"""
        # Setup
        transform = PadTransform(pad_top=top, pad_right=right, pad_bottom=bottom, pad_left=left)

        # Execute
        transformed_coords = transform.apply_coords(coords)

        # Assert
        np.testing.assert_almost_equal(transformed_coords, expected)

    @pytest.mark.parametrize(
        "coords,top,right,bottom,left,image_width,image_height,expected",
        [
            (
                np.array([[140, 60, 240, 160]], dtype=float32),
                10,
                20,
                30,
                40,
                400,
                300,
                np.array([[100, 50, 200, 150]], dtype=float32),
            ),
            (
                np.array([[0, 0, 300, 200]], dtype=float32),
                10,
                20,
                30,
                40,
                200,
                150,
                np.array([[0, 0, 200, 150]], dtype=float32),  # Clipped to image boundaries
            ),
        ],
    )
    def test_inverse_apply_coords(
        self,
        coords: npt.NDArray[np.float32],
        top: int,
        right: int,
        bottom: int,
        left: int,
        image_width: int,
        image_height: int,
        expected: npt.NDArray[np.float32],
    ) -> None:
        """Test PadTransform.inverse_apply_coords with parametrized bounding boxes"""

        # Setup
        transform = PadTransform(pad_top=top, pad_right=right, pad_bottom=bottom, pad_left=left)
        transform.image_width = image_width
        transform.image_height = image_height

        # Execute
        restored_coords = transform.inverse_apply_coords(coords)

        # Assert
        np.testing.assert_almost_equal(restored_coords, expected)

    @pytest.mark.parametrize(
        "coords,top,right,bottom,left,image_width,image_height",
        [
            (np.array([[140, 60, 240, 160]], dtype=float32), 10, 20, 30, 40, 400, 300),
            (np.array([[60, 55, 160, 155]], dtype=float32), 5, 15, 25, 10, 200, 150),
        ],
    )
    def test_transform_inverse_transform_cycle(
        self,
        coords: npt.NDArray[np.float32],
        top: int,
        right: int,
        bottom: int,
        left: int,
        image_width: int,
        image_height: int,
    ) -> None:
        """Test that applying transform followed by inverse_transform returns original coordinates"""
        # Setup
        transform = PadTransform(pad_top=top, pad_right=right, pad_bottom=bottom, pad_left=left)
        transform.image_width = image_width
        transform.image_height = image_height

        # Get the original coordinates by inverse transformation
        original = transform.inverse_apply_coords(coords.copy())

        # Apply transform to get back to padded coordinates
        transform.image_width = image_width
        transform.image_height = image_height
        transformed = transform.apply_coords(original.copy())

        # Assert
        np.testing.assert_almost_equal(transformed, coords)


class TestRotationTransform:
    """Test class for RotationTransform"""

    @pytest.mark.parametrize("angle", [90, 180, 270, 360])
    def test_apply_image(self, np_image_large: PixelValues, angle: Literal[90, 180, 270, 360]) -> None:
        """Test RotationTransform.apply_image with large image for different angles"""
        # Setup
        h, w = np_image_large.shape[:2]
        transform = RotationTransform(angle=angle)

        # Execute
        rotated_image = transform.apply_image(np_image_large)

        # Assert
        if angle in (90, 270):
            assert rotated_image.shape == (w, h, 3)  # Height and width are swapped
        else:
            assert rotated_image.shape == (h, w, 3)  # Same dimensions

    @pytest.mark.parametrize(
        "angle,coords,expected",
        [
            (90, np.array([[100, 50, 200, 150]], dtype=float32), np.array([[50, 200, 150, 300]], dtype=float32)),
            (180, np.array([[100, 50, 200, 150]], dtype=float32), np.array([[200, 150, 300, 250]], dtype=float32)),
            (270, np.array([[100, 50, 200, 150]], dtype=float32), np.array([[150, 100, 250, 200]], dtype=float32)),
            (360, np.array([[100, 50, 200, 150]], dtype=float32), np.array([[100, 50, 200, 150]], dtype=float32)),
        ],
    )
    def test_apply_coords(
        self, angle: Literal[90, 180, 270, 360], coords: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]
    ) -> None:
        """Test RotationTransform.apply_coords with parametrized angles and bounding boxes"""
        # Setup
        transform = RotationTransform(angle=angle)
        transform.image_width = 400
        transform.image_height = 300

        # Execute
        transformed_coords = transform.apply_coords(coords.copy())

        # Assert
        np.testing.assert_almost_equal(transformed_coords, expected)

    @pytest.mark.parametrize(
        "angle,coords,expected",
        [
            (90, np.array([[50, 200, 150, 300]], dtype=float32), np.array([[100, 50, 200, 150]], dtype=float32)),
            (180, np.array([[200, 150, 300, 250]], dtype=float32), np.array([[100, 50, 200, 150]], dtype=float32)),
            (270, np.array([[150, 100, 250, 200]], dtype=float32), np.array([[100, 50, 200, 150]], dtype=float32)),
            (360, np.array([[150, 100, 250, 200]], dtype=float32), np.array([[150, 100, 250, 200]], dtype=float32)),
        ],
    )
    def test_inverse_apply_coords(
        self, angle: Literal[90, 180, 270, 360], coords: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]
    ) -> None:
        """Test RotationTransform.inverse_apply_coords with parametrized angles and bounding boxes"""

        # Setup
        transform = RotationTransform(angle=angle)
        transform.image_width = 400
        transform.image_height = 300

        # Execute
        restored_coords = transform.inverse_apply_coords(coords)

        # Assert
        np.testing.assert_almost_equal(restored_coords, expected)

    @pytest.mark.parametrize("angle", [90, 180, 270, 360])
    def test_transform_inverse_transform_cycle(self, angle: Literal[90, 180, 270, 360]) -> None:
        """Test that applying transform followed by inverse_transform returns original coordinates"""
        # Setup
        transform = RotationTransform(angle=angle)
        transform.image_width = 400
        transform.image_height = 300

        original_coords = np.array([[100, 50, 200, 150]], dtype=float32)

        # Execute
        transformed = transform.apply_coords(original_coords.copy())
        restored = transform.inverse_apply_coords(transformed.copy())

        # Assert
        np.testing.assert_almost_equal(restored, original_coords)
