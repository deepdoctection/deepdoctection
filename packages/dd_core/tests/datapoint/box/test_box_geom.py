# -*- coding: utf-8 -*-
# File: test_bbox_geom.py

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
Focused pytest coverage for BoundingBox geometry helper functions:
- intersection_box
- crop_box_from_image
- local_to_global_coords
- global_to_local_coords
- merge_boxes
- rescale_coords
- intersection_boxes
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from dd_core.datapoint import (
    BoundingBox,
    crop_box_from_image,
    global_to_local_coords,
    intersection_box,
    intersection_boxes,
    local_to_global_coords,
    merge_boxes,
    rescale_coords,
)
from dd_core.utils import BoundingBoxError


class TestIntersectionBox:
    """Test suite for intersection_box function."""

    def test_case_a_same_mode_absolute_overlapping(self) -> None:
        """
        Case A: Same mode (absolute), overlapping boxes.
        Assert that ulx/uly are floored and lrx/lry are ceiled.
        """
        # Box 1: [10.3, 20.7] to [50.2, 60.8]
        # Box 2: [30.1, 40.5] to [70.9, 80.3]
        # Expected intersection: [30, 40] to [51, 61]
        box1 = BoundingBox(absolute_coords=True, ulx=10.3, uly=20.7, lrx=50.2, lry=60.8)
        box2 = BoundingBox(absolute_coords=True, ulx=30.1, uly=40.5, lrx=70.9, lry=80.3)

        result = intersection_box(box1, box2)

        assert result.absolute_coords is True
        assert result.ulx == 30
        assert result.uly == 40
        assert result.lrx == 50
        assert result.lry == 61

    def test_case_a_exact_overlap(self) -> None:
        """Same absolute boxes produce identical intersection."""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=30, lry=40)
        result = intersection_box(box, box)

        assert result == box

    def test_case_b_no_overlap_raises(self) -> None:
        """Case B: No overlap should raise BoundingBoxError or ValueError."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20)
        box2 = BoundingBox(absolute_coords=True, ulx=30, uly=30, lrx=40, lry=40)

        with pytest.raises(BoundingBoxError):
            intersection_box(box1, box2)

    def test_case_b_touching_boxes_zero_width(self) -> None:
        """Case B: Boxes just touching (zero width) should raise exception."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=30)
        box2 = BoundingBox(absolute_coords=True, ulx=20, uly=15, lrx=30, lry=25)

        # They touch at x=20, so intersection would have zero width
        with pytest.raises(BoundingBoxError):
            intersection_box(box1, box2)

    def test_case_b_touching_boxes_zero_height(self) -> None:
        """Case B: Boxes just touching (zero height) should raise exception."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=30, lry=20)
        box2 = BoundingBox(absolute_coords=True, ulx=15, uly=20, lrx=25, lry=30)

        # They touch at y=20, so intersection would have zero height
        with pytest.raises(BoundingBoxError):
            intersection_box(box1, box2)

    def test_case_c_mixed_modes_without_dimensions_raises(self) -> None:
        """Case C: Mixed modes without width/height raises AssertionError."""
        box1 = BoundingBox(absolute_coords=True, ulx=100, uly=100, lrx=200, lry=200)
        box2 = BoundingBox(absolute_coords=False, ulx=0.3, uly=0.4, lrx=0.7, lry=0.8)

        with pytest.raises(AssertionError):
            intersection_box(box1, box2)

    def test_case_c_mixed_modes_with_dimensions(self) -> None:
        """
        Case C: Mixed modes with width/height provided.
        Result has coordinate mode of box_2.
        """
        # Image: 1000x500
        # Box1 absolute: [100, 50] to [300, 150]
        # Box2 relative: [0.2, 0.2] to [0.4, 0.4] => absolute [200, 100] to [400, 200]
        # Intersection (absolute): [200, 100] to [300, 150]
        # In relative coords: [0.2, 0.2] to [0.3, 0.3]

        box1 = BoundingBox(absolute_coords=True, ulx=100, uly=50, lrx=300, lry=150)
        box2 = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.2, lrx=0.4, lry=0.4)

        result = intersection_box(box1, box2, width=1000.0, height=500.0)

        assert result.absolute_coords is False  # Same as box2
        assert_allclose(result.ulx, 0.2, rtol=1e-6)
        assert_allclose(result.uly, 0.2, rtol=1e-6)
        assert_allclose(result.lrx, 0.3, rtol=1e-6)
        assert_allclose(result.lry, 0.3, rtol=1e-6)

    def test_case_c_mixed_modes_reverse_order(self) -> None:
        """Mixed modes with box1 relative, box2 absolute."""
        # Image: 800x600
        # Box1 relative: [0.25, 0.5] to [0.75, 1.0] => absolute [200, 300] to [600, 600]
        # Box2 absolute: [300, 400] to [500, 550]
        # Intersection (absolute): [300, 400] to [500, 550]

        box1 = BoundingBox(absolute_coords=False, ulx=0.25, uly=0.5, lrx=0.75, lry=1.0)
        box2 = BoundingBox(absolute_coords=True, ulx=300, uly=400, lrx=500, lry=550)

        result = intersection_box(box1, box2, width=800.0, height=600.0)

        assert result.absolute_coords is True  # Same as box2
        assert result.ulx == 300
        assert result.uly == 400
        assert result.lrx == 500
        assert result.lry == 550

    def test_relative_coords_intersection(self) -> None:
        """Intersection of relative coordinate boxes."""
        box1 = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.5, lry=0.6)
        box2 = BoundingBox(absolute_coords=False, ulx=0.3, uly=0.4, lrx=0.7, lry=0.8)

        result = intersection_box(box1, box2)

        assert result.absolute_coords is False
        assert_allclose(result.ulx, 0.3, rtol=1e-6)
        assert_allclose(result.uly, 0.4, rtol=1e-6)
        assert_allclose(result.lrx, 0.5, rtol=1e-6)
        assert_allclose(result.lry, 0.6, rtol=1e-6)


def get_np_array_for_cropping() -> NDArray[np.int32]:
    """
    numpy array for cropping
    """
    return np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]], dtype=np.uint8)


class TestCropBoxFromImage:
    """Test suite for crop_box_from_image function."""

    def test_absolute_crop_shape(self) -> None:
        """
        Absolute crop: shape equals (ceil(lry)-floor(uly), ceil(lrx)-floor(ulx)).
        """
        # Create a 23x17 test image
        image = np.arange(23 * 17).reshape(23, 17).astype(np.uint8)

        crop_box = BoundingBox(absolute_coords=True, ulx=5.3, uly=7.2, lrx=12.8, lry=18.9)

        result = crop_box_from_image(image, crop_box)

        assert result.shape == (12, 8)  # 19 - 7 = 12, 13 - 5 = 8

    @pytest.mark.parametrize(
        "np_image,crop_box,width,height,expected_np_array",
        [
            (
                get_np_array_for_cropping(),
                BoundingBox(absolute_coords=True, ulx=1, uly=1, lrx=3, lry=3),
                None,
                None,
                np.array([[[12, 13, 14], [15, 16, 17]]], dtype=np.uint8),
            ),
            (
                get_np_array_for_cropping(),
                BoundingBox(absolute_coords=True, ulx=0.5, uly=1.0, lrx=1.5, lry=2.3),
                None,
                None,
                np.array([[[9, 10, 11], [12, 13, 14]]], dtype=np.uint8),
            ),
            (
                get_np_array_for_cropping(),
                BoundingBox(absolute_coords=True, ulx=0, uly=0, lrx=1, lry=1),
                None,
                None,
                np.array([[[0, 1, 2]]], dtype=np.uint8),
            ),
            (
                get_np_array_for_cropping(),
                BoundingBox(absolute_coords=False, ulx=0, uly=0, lrx=0.5, lry=0.5),
                2,
                3,
                np.array([[[0, 1, 2]], [[9, 10, 11]]], dtype=np.uint8),
            ),
        ],
    )
    def test_crop_image_and_check_pixelwise(
        self,
        np_image: NDArray[np.uint8],
        crop_box: BoundingBox,
        width: int | None,
        height: int | None,
        expected_np_array: NDArray[np.uint8],
    ) -> None:
        """
        Testing func: crop_image returns np_image correctly
        """

        # Act
        cropped_image = crop_box_from_image(np_image, crop_box, width, height)

        # Assert
        assert_array_equal(cropped_image, expected_np_array)

    def test_absolute_crop_clamping_at_borders(self) -> None:
        """Cropping with out-of-bounds box clamps to image extents."""
        image = np.ones((10, 15), dtype=np.uint8) * 42

        # Box extends beyond image
        crop_box = BoundingBox(absolute_coords=True, ulx=10, uly=5, lrx=20, lry=15)
        result = crop_box_from_image(image, crop_box)

        # Image is 10 rows (0-9), 15 cols (0-14)
        # Crop uly=5 to lry=15, but max row is 9
        # Crop ulx=10 to lrx=20, but max col is 14
        # Result: rows 5-9 (5 rows), cols 10-14 (5 cols)
        assert result.shape == (5, 5)
        assert np.all(result == 42)

    def test_relative_crop_without_dimensions_raises(self) -> None:
        """Relative crop without width/height raises AssertionError."""
        image = np.zeros((20, 30), dtype=np.uint8)
        crop_box = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.6, lry=0.7)

        with pytest.raises(AssertionError):
            crop_box_from_image(image, crop_box)

    def test_relative_crop_with_dimensions(self) -> None:
        """
        Relative crop with width/height produces same region as absolute.
        """
        # 20x30 image
        image = np.arange(20 * 30).reshape(20, 30).astype(np.uint8)

        # Relative box: [0.2, 0.3] to [0.6, 0.7]
        # For 30x20 image: absolute [6, 6] to [18, 14]
        crop_box_rel = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.6, lry=0.7)
        result_rel = crop_box_from_image(image, crop_box_rel, width=30.0, height=20.0)

        # Equivalent absolute crop
        crop_box_abs = BoundingBox(absolute_coords=True, ulx=6, uly=6, lrx=18, lry=14)
        result_abs = crop_box_from_image(image, crop_box_abs)

        assert_array_equal(result_rel, result_abs)

    def test_out_of_bounds_crop_clamped(self) -> None:
        """Out-of-bounds box is clamped to image size."""
        image = np.arange(100).reshape(10, 10).astype(np.uint8)

        # Box completely outside
        crop_box = BoundingBox(absolute_coords=True, ulx=15, uly=15, lrx=25, lry=25)
        result = crop_box_from_image(image, crop_box)

        # Should get empty array or minimal slice
        # Image max is (10, 10), so cropping [15:, 15:] gives empty
        assert result.shape[0] == 0 or result.shape[1] == 0


class TestLocalGlobalCoords:
    """Test suite for local/global coordinate transformations."""

    def test_local_to_global_happy_path(self) -> None:
        """
        local_to_global: both absolute, global = local + embedding offset.
        """
        embedding = BoundingBox(absolute_coords=True, ulx=50, uly=100, lrx=200, lry=300)
        local_box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=30, lry=40)

        result = local_to_global_coords(local_box, embedding)

        # Expected: offset by embedding's (ulx, uly)
        assert result.absolute_coords is True
        assert result.ulx == 50 + 10  # 60
        assert result.uly == 100 + 20  # 120
        assert result.lrx == 50 + 30  # 80
        assert result.lry == 100 + 40  # 140

    def test_local_to_global_non_absolute_raises(self) -> None:
        """local_to_global with non-absolute coords raises AssertionError."""
        embedding = BoundingBox(absolute_coords=False, ulx=0.5, uly=0.5, lrx=0.9, lry=0.9)
        local_box = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20)

        with pytest.raises(AssertionError):
            local_to_global_coords(local_box, embedding)

    def test_local_to_global_local_non_absolute_raises(self) -> None:
        """local_to_global with local box non-absolute raises AssertionError."""
        embedding = BoundingBox(absolute_coords=True, ulx=50, uly=100, lrx=200, lry=300)
        local_box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.1, lrx=0.3, lry=0.3)

        with pytest.raises(AssertionError):
            local_to_global_coords(local_box, embedding)

    def test_global_to_local_fully_inside(self) -> None:
        """
        global_to_local: fully inside embedding, local = global - embedding.ul.
        """
        embedding = BoundingBox(absolute_coords=True, ulx=100, uly=200, lrx=400, lry=500)
        global_box = BoundingBox(absolute_coords=True, ulx=150, uly=250, lrx=300, lry=400)

        result = global_to_local_coords(global_box, embedding)

        assert result.absolute_coords is True
        assert result.ulx == 150 - 100  # 50
        assert result.uly == 250 - 200  # 50
        assert result.lrx == 300 - 100  # 200
        assert result.lry == 400 - 200  # 200

    def test_global_to_local_partially_outside_clipped(self) -> None:
        """
        global_to_local: partially outside, verify clipping.
        Values clamped to [0, width] x [0, height].
        """
        # Embedding: [100, 200] to [400, 500], size 300x300
        embedding = BoundingBox(absolute_coords=True, ulx=100, uly=200, lrx=400, lry=500)

        # Global box extends beyond: [50, 150] to [450, 550]
        global_box = BoundingBox(absolute_coords=True, ulx=50, uly=150, lrx=450, lry=550)

        result = global_to_local_coords(global_box, embedding)

        # ulx: max(50 - 100, 0) = 0
        # uly: max(150 - 200, 0) = 0
        # lrx: min(450 - 100, 300) = 300
        # lry: min(550 - 200, 300) = 300
        assert result.ulx == 0
        assert result.uly == 0
        assert result.lrx == 300
        assert result.lry == 300

    def test_global_to_local_non_absolute_raises(self) -> None:
        """global_to_local with non-absolute coords raises AssertionError."""
        embedding = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.8, lry=0.9)
        global_box = BoundingBox(absolute_coords=True, ulx=100, uly=100, lrx=200, lry=200)

        with pytest.raises(AssertionError):
            global_to_local_coords(global_box, embedding)

    def test_roundtrip_local_global_local(self) -> None:
        """Roundtrip: local -> global -> local should preserve box."""
        embedding = BoundingBox(absolute_coords=True, ulx=50, uly=80, lrx=350, lry=480)
        local_original = BoundingBox(absolute_coords=True, ulx=20, uly=30, lrx=100, lry=150)

        global_box = local_to_global_coords(local_original, embedding)
        local_back = global_to_local_coords(global_box, embedding)

        assert local_back == local_original


class TestMergeBoxes:
    """Test suite for merge_boxes function."""

    def test_merge_multiple_absolute_boxes(self) -> None:
        """Multiple absolute boxes -> min ulx/uly, max lrx/lry."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=50, lry=60)
        box2 = BoundingBox(absolute_coords=True, ulx=30, uly=10, lrx=70, lry=40)
        box3 = BoundingBox(absolute_coords=True, ulx=5, uly=50, lrx=40, lry=80)

        result = merge_boxes(box1, box2, box3)

        assert result.absolute_coords is True
        assert result.ulx == 5  # min of 10, 30, 5
        assert result.uly == 10  # min of 20, 10, 50
        assert result.lrx == 70  # max of 50, 70, 40
        assert result.lry == 80  # max of 60, 40, 80

    def test_merge_multiple_relative_boxes(self) -> None:
        """Multiple relative boxes -> min/max coords."""
        box1 = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.5, lry=0.6)
        box2 = BoundingBox(absolute_coords=False, ulx=0.4, uly=0.1, lrx=0.7, lry=0.4)
        box3 = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.5, lrx=0.3, lry=0.9)

        result = merge_boxes(box1, box2, box3)

        assert result.absolute_coords is False
        assert_allclose(result.ulx, 0.1, rtol=1e-6)
        assert_allclose(result.uly, 0.1, rtol=1e-6)
        assert_allclose(result.lrx, 0.7, rtol=1e-6)
        assert_allclose(result.lry, 0.9, rtol=1e-6)

    def test_merge_single_box(self) -> None:
        """Merging a single box returns same box."""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=30, lry=40)
        result = merge_boxes(box)

        assert result == box

    def test_merge_mixed_modes_raises(self) -> None:
        """Mixed modes raise AssertionError."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20)
        box2 = BoundingBox(absolute_coords=False, ulx=0.3, uly=0.3, lrx=0.5, lry=0.5)

        with pytest.raises(AssertionError):
            merge_boxes(box1, box2)


class TestRescaleCoords:
    """Test suite for rescale_coords function."""

    def test_rescale_absolute_scale_up(self) -> None:
        """Absolute mode: scale up by factor of 2."""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=30, lry=40)

        # Current: 100x100, Scaled: 200x200 (2x scale)
        result = rescale_coords(box, 100.0, 100.0, 200.0, 200.0)

        assert result.absolute_coords is True
        assert result.ulx == 20  # 10 * 2
        assert result.uly == 40  # 20 * 2
        assert result.lrx == 60  # 30 * 2
        assert result.lry == 80  # 40 * 2

    def test_rescale_absolute_scale_down(self) -> None:
        """Absolute mode: scale down by factor of 0.5."""
        box = BoundingBox(absolute_coords=True, ulx=100, uly=200, lrx=300, lry=400)

        # Current: 1000x1000, Scaled: 500x500 (0.5x scale)
        result = rescale_coords(box, 1000.0, 1000.0, 500.0, 500.0)

        assert result.absolute_coords is True
        assert result.ulx == 50  # 100 * 0.5
        assert result.uly == 100  # 200 * 0.5
        assert result.lrx == 150  # 300 * 0.5
        assert result.lry == 200  # 400 * 0.5

    def test_rescale_absolute_non_uniform(self) -> None:
        """Absolute mode: non-uniform scaling (different x and y factors)."""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=50, lry=80)

        # Width: 100 -> 200 (2x), Height: 100 -> 50 (0.5x)
        result = rescale_coords(box, 100.0, 100.0, 200.0, 50.0)

        assert result.absolute_coords is True
        assert result.ulx == 20  # 10 * 2
        assert result.uly == 10  # 20 * 0.5
        assert result.lrx == 100  # 50 * 2
        assert result.lry == 40  # 80 * 0.5

    def test_rescale_relative_unchanged(self) -> None:
        """Relative mode: function returns same box unchanged."""
        box = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.6, lry=0.8)

        result = rescale_coords(box, 100.0, 100.0, 200.0, 200.0)

        assert result.absolute_coords is False
        # Should be the same box (identity)
        assert result == box
        # Check if it's actually the same object
        assert result is box

    def test_rescale_no_change(self) -> None:
        """Scaling with same dimensions returns box with same coords."""
        box = BoundingBox(absolute_coords=True, ulx=15, uly=25, lrx=45, lry=65)

        result = rescale_coords(box, 100.0, 100.0, 100.0, 100.0)

        assert result.ulx == box.ulx
        assert result.uly == box.uly
        assert result.lrx == box.lrx
        assert result.lry == box.lry


class TestIntersectionBoxes:
    """Test suite for intersection_boxes function."""

    def test_empty_inputs_both_empty(self) -> None:
        """([], []) returns []."""
        result = intersection_boxes([], [])
        assert not result

    def test_empty_inputs_first_empty(self) -> None:
        """([], B) returns B."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20)
        box2 = BoundingBox(absolute_coords=True, ulx=30, uly=30, lrx=40, lry=40)

        result = intersection_boxes([], [box1, box2])
        assert result == [box1, box2]

    def test_empty_inputs_second_empty(self) -> None:
        """(A, []) returns A."""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20)
        box2 = BoundingBox(absolute_coords=True, ulx=30, uly=30, lrx=40, lry=40)

        result = intersection_boxes([box1, box2], [])
        assert result == [box1, box2]

    def test_mode_mismatch_raises(self) -> None:
        """Mode mismatch between the two lists raises ValueError."""
        boxes1 = [BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20)]
        boxes2 = [BoundingBox(absolute_coords=False, ulx=0.3, uly=0.3, lrx=0.5, lry=0.5)]

        with pytest.raises(ValueError):
            intersection_boxes(boxes1, boxes2)

    def test_pairwise_intersections_mixed_overlap(self) -> None:
        """
        Pairwise intersections with mixed overlap.
        2x2 grid: some pairs intersect, some don't.
        """
        # Boxes1: two boxes
        box1a = BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=50, lry=50)
        box1b = BoundingBox(absolute_coords=True, ulx=100, uly=100, lrx=150, lry=150)

        # Boxes2: two boxes
        box2a = BoundingBox(absolute_coords=True, ulx=30, uly=30, lrx=70, lry=70)
        box2b = BoundingBox(absolute_coords=True, ulx=120, uly=120, lrx=180, lry=180)

        # Overlaps:
        # - box1a & box2a: YES [30,30] to [50,50]
        # - box1a & box2b: NO
        # - box1b & box2a: NO
        # - box1b & box2b: YES [120,120] to [150,150]
        # Expected: 2 valid intersections out of 4 pairs

        result = intersection_boxes([box1a, box1b], [box2a, box2b])

        # Should have 2 results (the two valid intersections)
        assert len(result) == 2

        # Check the first intersection (box1a & box2a)
        # Expected: [30, 30] to [50, 50]
        intersect_1 = [r for r in result if r.ulx == 30 and r.uly == 30]
        assert len(intersect_1) == 1
        assert intersect_1[0].lrx == 50
        assert intersect_1[0].lry == 50

        # Check the second intersection (box1b & box2b)
        # Expected: [120, 120] to [150, 150]
        intersect_2 = [r for r in result if r.ulx == 120 and r.uly == 120]
        assert len(intersect_2) == 1
        assert intersect_2[0].lrx == 150
        assert intersect_2[0].lry == 150

    def test_pairwise_intersections_all_overlap(self) -> None:
        """All pairs overlap."""
        boxes1 = [
            BoundingBox(absolute_coords=True, ulx=0, uly=0, lrx=100, lry=100),
            BoundingBox(absolute_coords=True, ulx=50, uly=50, lrx=150, lry=150),
        ]
        boxes2 = [
            BoundingBox(absolute_coords=True, ulx=25, uly=25, lrx=75, lry=75),
        ]

        # 2x1 = 2 pairs, both should overlap
        result = intersection_boxes(boxes1, boxes2)
        assert len(result) == 2

    def test_pairwise_intersections_no_overlap(self) -> None:
        """No pairs overlap."""
        boxes1 = [
            BoundingBox(absolute_coords=True, ulx=10, uly=10, lrx=20, lry=20),
        ]
        boxes2 = [
            BoundingBox(absolute_coords=True, ulx=100, uly=100, lrx=110, lry=110),
        ]

        # 1x1 = 1 pair, no overlap -> constructor will fail, so 0 results
        result = intersection_boxes(boxes1, boxes2)
        assert len(result) == 0

    def test_pairwise_exact_intersection_values(self) -> None:
        """
        Verify exact intersection values for representative pairs.
        Including floor/ceil behavior for absolute mode.
        """
        # Box1: [10, 21] to [50, 61]
        # Box2: [30, 40] to [71, 80]
        # Intersection: [30, 40] to [50, 61]

        boxes1 = [BoundingBox(absolute_coords=True, ulx=10.3, uly=20.7, lrx=50.2, lry=60.8)]
        boxes2 = [BoundingBox(absolute_coords=True, ulx=30.1, uly=40.5, lrx=70.9, lry=80.3)]

        result = intersection_boxes(boxes1, boxes2)

        assert len(result) == 1
        assert result[0].ulx == 30
        assert result[0].uly == 40
        assert result[0].lrx == 50
        assert result[0].lry == 61

    def test_relative_coords_pairwise(self) -> None:
        """Pairwise intersections with relative coordinates."""
        boxes1 = [
            BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.5, lry=0.6),
            BoundingBox(absolute_coords=False, ulx=0.6, uly=0.6, lrx=0.9, lry=0.9),
        ]
        boxes2 = [
            BoundingBox(absolute_coords=False, ulx=0.3, uly=0.4, lrx=0.7, lry=0.8),
        ]

        # First box overlaps with boxes2[0]: [0.3, 0.4] to [0.5, 0.6]
        # Second box overlaps with boxes2[0]: [0.6, 0.6] to [0.7, 0.8]
        result = intersection_boxes(boxes1, boxes2)

        assert len(result) == 2
        assert all(r.absolute_coords is False for r in result)
