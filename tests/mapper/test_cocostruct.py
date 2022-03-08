# -*- coding: utf-8 -*-
# File: test_cocostruct.py

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
Testing the module mapper.cocostruct
"""

from math import isclose
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_allclose

from deepdoctection.datapoint import Image
from deepdoctection.mapper import coco_to_image, image_to_coco
from deepdoctection.utils.detection_types import JsonDict

from .conftest import get_coco_white_image
from .data import DatapointCoco, DatapointImage


@patch("deepdoctection.mapper.cocostruct.load_image_from_file", MagicMock(side_effect=get_coco_white_image))
def test_coco_to_image(datapoint_coco: JsonDict, categories_coco: Dict[str, str], coco_results: DatapointCoco) -> None:
    """
    testing coco_to_image is mapping correctly
    """
    load_image = True
    # Act
    coco_to_image_mapper = coco_to_image(categories_coco, load_image, True, True)  # type: ignore # pylint: disable=E1120  # 259
    dp = coco_to_image_mapper(datapoint_coco)

    # Assert
    datapoint = coco_results
    assert dp is not None
    test_anns = dp.get_annotation()
    assert len(test_anns) == datapoint.get_number_anns()
    assert dp.width == datapoint.get_width(load_image)
    assert dp.height == datapoint.get_height(load_image)
    assert test_anns[0].category_name == datapoint.get_first_ann_category(False)
    assert test_anns[0].category_id == datapoint.get_first_ann_category(True)
    assert isclose(test_anns[0].bounding_box.ulx, datapoint.get_first_ann_box().ulx, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.uly, datapoint.get_first_ann_box().uly, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.width, datapoint.get_first_ann_box().w, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.height, datapoint.get_first_ann_box().h, rel_tol=1e-15)

    load_image = False
    coco_to_image_mapper = coco_to_image(categories_coco, load_image, True, True)  # type: ignore # pylint: disable=E1120  # 259
    dp = coco_to_image_mapper(datapoint_coco)
    assert isclose(dp.width, datapoint.get_width(load_image), rel_tol=1e-15)
    assert isclose(dp.height, datapoint.get_height(load_image), rel_tol=1e-15)


def test_image_to_coco(datapoint_image: Image, image_results: DatapointImage) -> None:
    """
    testing image_to_coco is mapping correctly
    """

    # Act
    img, anns = image_to_coco(datapoint_image)

    # Assert
    expected_image = image_results
    assert img == expected_image.get_coco_image()
    assert len(anns) == 2

    first_ann = anns[0]
    expected_ann = expected_image.get_coco_anns()[0]
    assert isclose(first_ann["area"], expected_ann["area"], abs_tol=1e-10)
    assert_allclose(np.asarray(first_ann["bbox"]), np.asarray(expected_ann["bbox"]), atol=1e-10)  # type: ignore
    assert isclose(first_ann["category_id"], expected_ann["category_id"], abs_tol=1e-10)
    assert isclose(first_ann["score"], expected_ann["score"], abs_tol=1e-10)
