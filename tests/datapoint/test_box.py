# -*- coding: utf-8 -*-
# File: test_box.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
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
Testing the module datapoint.box
"""

from typing import List, Optional

from numpy import asarray
from numpy.testing import assert_almost_equal, assert_array_equal
from pytest import mark, raises

from deepdoctection.datapoint import (
    BoundingBox,
    crop_box_from_image,
    global_to_local_coords,
    intersection_box,
    local_to_global_coords,
    merge_boxes,
    rescale_coords,
)
from deepdoctection.utils.detection_types import ImageType

from .conftest import Box


class TestBoundingBox:
    """
    Testing BoundingBox methods
    """

    @staticmethod
    def test_incomplete_data_for_bounding_box(box: Box) -> None:
        """
        Testing assertion errors when box constructor receives incomplete
        data.
        :param box: Box dataclass from fixtures
        """

        # Act and Assert
        with raises(AssertionError):
            BoundingBox(ulx=box.ulx, uly=box.uly, lrx=box.lrx, absolute_coords=box.absolute_coords)

        with raises(AssertionError):
            BoundingBox(ulx=box.ulx, uly=box.uly, height=box.h, width=0.0, absolute_coords=box.absolute_coords)

    @staticmethod
    def test_center_area_to_list(box: Box) -> None:
        """
        Testing internal box coordinates and get_export methods
        :param box: Box dataclass from fixtures
        """

        # Arrange
        bounding_box = BoundingBox(
            ulx=box.ulx, uly=box.uly, lrx=box.lrx, lry=box.lry, absolute_coords=box.absolute_coords
        )

        # Assert
        assert bounding_box.area == box.area
        assert bounding_box.cx == box.cx
        assert bounding_box.cy == box.cy
        assert bounding_box.to_list(mode="xyxy") == [box.ulx, box.uly, box.lrx, box.lry]
        assert bounding_box.to_list(mode="xywh") == [box.ulx, box.uly, box.w, box.h]

    @staticmethod
    def test_transform(box: Box) -> None:
        """
        Testing relative <-> absolute coordinate transformation
        :param box: Box dataclass from fixtures
        """

        # Arrange
        bounding_box_absolute = BoundingBox(
            ulx=box.ulx, uly=box.uly, lrx=box.lrx, lry=box.lry, absolute_coords=box.absolute_coords
        )

        # Act
        box_relative_list = bounding_box_absolute.transform(
            image_width=box.image_width,
            image_height=box.image_height,
            absolute_coords=False,
            output="list",
            mode="xyxy",
        )
        box_copy = bounding_box_absolute.transform(
            image_width=box.image_width, image_height=box.image_height, absolute_coords=False, output="box", mode="xyxy"
        )

        # Assert
        assert box_relative_list == [box.ulx_relative, box.uly_relative, box.lrx_relative, box.lry_relative]
        assert isinstance(box_copy, BoundingBox)
        assert box_copy.absolute_coords is False


@mark.parametrize(
    "box_1,box_2,expected_box",
    [
        (
            BoundingBox(absolute_coords=True, ulx=1, uly=1.5, lrx=3.5, lry=3),
            BoundingBox(absolute_coords=True, ulx=2, uly=2.5, lrx=4.5, lry=4.0),
            BoundingBox(absolute_coords=True, ulx=2, uly=2, lrx=4, lry=3),
        ),
        (
            BoundingBox(absolute_coords=True, ulx=0, uly=0, lrx=100, lry=100),
            BoundingBox(absolute_coords=True, ulx=10, uly=15, lrx=35.5, lry=30),
            BoundingBox(absolute_coords=True, ulx=10, uly=15, lrx=36, lry=30),
        ),
        (
            BoundingBox(absolute_coords=False, ulx=0.25, uly=0.4, lrx=0.6, lry=0.6),
            BoundingBox(absolute_coords=False, ulx=0.3, uly=0.55, lrx=0.4, lry=0.8),
            BoundingBox(absolute_coords=False, ulx=0.3, uly=0.55, lrx=0.4, lry=0.6),
        ),
    ],
)
def test_intersection_box(box_1: BoundingBox, box_2: BoundingBox, expected_box: BoundingBox) -> None:
    """
    Testing intersection box
    """

    # Act
    output_box = intersection_box(box_1, box_2)

    # Assert
    assert output_box == expected_box


@mark.parametrize(
    "box_1,box_2,width,height,expected_box",
    [
        (
            BoundingBox(absolute_coords=True, ulx=1, uly=5.0, lrx=3.0, lry=13.0),
            BoundingBox(absolute_coords=False, ulx=0.2, uly=0.5, lrx=0.5, lry=0.7),
            10,
            20,
            BoundingBox(absolute_coords=False, ulx=0.2, uly=0.5, lrx=0.3, lry=0.65),
        ),
        (
            BoundingBox(absolute_coords=False, ulx=0, uly=0, lrx=1.0, lry=1.0),
            BoundingBox(absolute_coords=True, ulx=10, uly=15, lrx=35.5, lry=30),
            100,
            200,
            BoundingBox(absolute_coords=True, ulx=10, uly=15, lrx=36, lry=30),
        ),
    ],
)
def test_intersection_box_with_diff_abs_coords(
    box_1: BoundingBox, box_2: BoundingBox, width: float, height: float, expected_box: BoundingBox
) -> None:
    """
    Testing intersection box with different absolute coords format
    """

    # Act
    output_box = intersection_box(box_1, box_2, width, height)

    # Assert
    assert output_box == expected_box


def get_np_array_for_cropping() -> ImageType:
    """
    numpy array for cropping
    """
    return asarray([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]])


@mark.parametrize(
    "np_image,crop_box,width,height,expected_np_array",
    [
        (
            get_np_array_for_cropping(),
            BoundingBox(absolute_coords=True, ulx=1, uly=1, lrx=3, lry=3),
            None,
            None,
            asarray([[[12, 13, 14], [15, 16, 17]]]),
        ),
        (
            get_np_array_for_cropping(),
            BoundingBox(absolute_coords=True, ulx=0.5, uly=1.0, lrx=1.5, lry=2.3),
            None,
            None,
            asarray([[[9, 10, 11], [12, 13, 14]]]),
        ),
        (
            get_np_array_for_cropping(),
            BoundingBox(absolute_coords=True, ulx=0, uly=0, lrx=1, lry=1),
            None,
            None,
            asarray([[[0, 1, 2]]]),
        ),
        (
            get_np_array_for_cropping(),
            BoundingBox(absolute_coords=False, ulx=0, uly=0, lrx=0.5, lry=0.5),
            2,
            3,
            asarray([[[0, 1, 2]], [[9, 10, 11]]]),
        ),
    ],
)
def test_crop_image(
    np_image: ImageType,
    crop_box: BoundingBox,
    width: Optional[float],
    height: Optional[float],
    expected_np_array: ImageType,
) -> None:
    """
    Testing func: crop_image returns np_image coorectly
    """

    # Act
    cropped_image = crop_box_from_image(np_image, crop_box, width, height)

    # Assert
    assert_array_equal(cropped_image, expected_np_array)


@mark.parametrize(
    "local_box,embedding_box,expected_embedded_box",
    [
        (
            BoundingBox(absolute_coords=True, ulx=10, uly=15.5, lrx=20.0, lry=22.5),
            BoundingBox(absolute_coords=True, ulx=100, uly=150, lrx=200.0, lry=225),
            BoundingBox(absolute_coords=True, ulx=110, uly=165.5, lrx=120.0, lry=172.5),
        )
    ],
)
def test_local_to_global_coords(
    local_box: BoundingBox, embedding_box: BoundingBox, expected_embedded_box: BoundingBox
) -> None:
    """
    Testing func:  local_to_global_coords returns BoundingBox with global coords correctly
    """

    # Act
    embedded_box = local_to_global_coords(local_box, embedding_box)

    # Assert
    assert embedded_box == expected_embedded_box


@mark.parametrize(
    "global_box,embedding_box,expected_local_box",
    [
        (
            BoundingBox(absolute_coords=True, ulx=100, uly=150, lrx=200, lry=200),
            BoundingBox(absolute_coords=True, ulx=50, uly=50, lrx=250, lry=225),
            BoundingBox(absolute_coords=True, ulx=50, uly=100, lrx=150, lry=150),
        ),
        (
            BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=20, lry=30),
            BoundingBox(absolute_coords=True, ulx=10, uly=25, lrx=200, lry=200),
            BoundingBox(absolute_coords=True, ulx=0, uly=0, lrx=10, lry=5),
        ),
    ],
)
def test_global_to_local_coords(
    global_box: BoundingBox, embedding_box: BoundingBox, expected_local_box: BoundingBox
) -> None:
    """
    Testing func: global_to_local_coords returns BoundingBox with local coords correclty
    """

    # Act
    local_box = global_to_local_coords(global_box, embedding_box)

    # Assert
    assert local_box == expected_local_box


@mark.parametrize(
    "box_list,expected_box",
    [
        (
            [
                BoundingBox(absolute_coords=True, ulx=100, uly=150, lrx=200, lry=200),
                BoundingBox(absolute_coords=True, ulx=50, uly=50, lrx=250, lry=225),
                BoundingBox(absolute_coords=True, ulx=50, uly=100, lrx=150, lry=150),
            ],
            BoundingBox(absolute_coords=True, ulx=50, uly=50, lrx=250, lry=225),
        ),
        (
            [
                BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.4, lry=0.4),
                BoundingBox(absolute_coords=False, ulx=0.6, uly=0.6, lrx=0.7, lry=0.8),
            ],
            BoundingBox(absolute_coords=False, ulx=0.2, uly=0.3, lrx=0.7, lry=0.8),
        ),
    ],
)
def test_merge_boxes(box_list: List[BoundingBox], expected_box: BoundingBox) -> None:
    """
    Test func: merge_boxes returns smallest BoundingBox containing all input boxes
    """

    # Act
    merged_box = merge_boxes(*box_list)

    # Assert
    assert merged_box == expected_box


@mark.parametrize(
    "box,current_total_width,current_total_height,scaled_total_width,scaled_total_height,expected_box",
    [
        (
            BoundingBox(absolute_coords=True, ulx=100, uly=150, lrx=200, lry=200),
            500,
            700,
            1000,
            2100,
            BoundingBox(absolute_coords=True, ulx=200, uly=450, lrx=400, lry=600),
        ),
        (
            BoundingBox(absolute_coords=False, ulx=0.7, uly=0.3, lrx=0.85, lry=0.4),
            500,
            700,
            1000,
            2100,
            BoundingBox(absolute_coords=False, ulx=0.7, uly=0.3, lrx=0.85, lry=0.4),
        ),
        (
            BoundingBox(absolute_coords=True, ulx=35.5, uly=47, lrx=92, lry=81),
            500,
            700,
            100,
            210,
            BoundingBox(absolute_coords=True, ulx=7.1, uly=14.1, lrx=18.4, lry=24.3),
        ),
    ],
)
def test_rescale_coords(
    box: BoundingBox,
    current_total_width: float,
    current_total_height: float,
    scaled_total_width: float,
    scaled_total_height: float,
    expected_box: BoundingBox,
) -> None:
    """
    Test func: rescale_coords returns rescaled BoundingBox, if coordinates are in absolute coords terms
    """

    # Act
    rescaled_box = rescale_coords(
        box, current_total_width, current_total_height, scaled_total_width, scaled_total_height
    )

    # Assert
    np_box = rescaled_box.to_np_array(mode="xyxy")
    np_expected_box = expected_box.to_np_array(mode="xyxy")
    assert_almost_equal(np_box, np_expected_box, decimal=3)
