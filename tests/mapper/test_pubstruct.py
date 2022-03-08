# -*- coding: utf-8 -*-
# File: test_pubstruct.py

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
Testing the module mapper.pubstruct
"""

from math import isclose
from typing import Dict
from unittest.mock import MagicMock, patch

from deepdoctection.datapoint.annotation import SummaryAnnotation
from deepdoctection.datapoint.box import BoundingBox
from deepdoctection.mapper import pub_to_image
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.settings import names

from .conftest import get_pubtabnet_white_image
from .data import DatapointPubtabnet


@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_pub_to_image(
    datapoint_pubtabnet: JsonDict,
    categories_name_as_key_pubtabnet: Dict[str, str],
    pubtabnet_results: DatapointPubtabnet,
) -> None:
    """
    testing pub_to_image is mapping correctly
    """

    load_image = True

    # Act
    pub_to_image_mapper = pub_to_image(  # type: ignore # pylint: disable=E1120  # 259
        categories_name_as_key_pubtabnet, load_image, True, False
    )
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    datapoint = pubtabnet_results
    assert dp is not None
    test_anns = dp.get_annotation(category_names=names.C.CELL)

    # Assert
    assert isinstance(test_anns, list) and len(test_anns) >= 1
    first_ann = test_anns[0]
    last_ann = test_anns[-1]

    assert len(test_anns) == datapoint.get_number_cell_anns()
    assert dp.width == datapoint.get_width()
    assert dp.height == datapoint.get_height()

    assert first_ann.category_name == datapoint.get_first_ann_category(False)
    assert first_ann.category_id == datapoint.get_first_ann_category(True)
    assert isinstance(first_ann.bounding_box, BoundingBox)
    assert isclose(first_ann.bounding_box.ulx, datapoint.get_first_ann_box().ulx, rel_tol=1e-15)
    assert isclose(first_ann.bounding_box.uly, datapoint.get_first_ann_box().uly, rel_tol=1e-15)
    assert isclose(first_ann.bounding_box.width, datapoint.get_first_ann_box().w, rel_tol=1e-15)
    assert isclose(first_ann.bounding_box.height, datapoint.get_first_ann_box().h, rel_tol=1e-15)
    assert first_ann.get_sub_category(names.C.HEAD).category_name == datapoint.get_first_ann_sub_category_header_name()
    assert first_ann.get_sub_category(names.C.RN).category_id == datapoint.get_first_ann_sub_category_row_number_id()
    assert first_ann.get_sub_category(names.C.CN).category_id == datapoint.get_first_ann_sub_category_col_number_id()
    assert first_ann.get_sub_category(names.C.RS).category_id == datapoint.get_first_ann_sub_category_row_span_id()
    assert first_ann.get_sub_category(names.C.CS).category_id == datapoint.get_first_ann_sub_category_col_span_id()

    assert last_ann.category_name == datapoint.get_last_ann_category_name()
    assert last_ann.get_sub_category(names.C.HEAD).category_name == datapoint.get_last_ann_sub_category_header_name()
    assert last_ann.get_sub_category(names.C.RN).category_id == datapoint.get_last_ann_sub_category_row_number_id()
    assert last_ann.get_sub_category(names.C.CN).category_id == datapoint.get_last_ann_sub_category_col_number_id()
    assert last_ann.get_sub_category(names.C.RS).category_id == datapoint.get_last_ann_sub_category_row_span_id()
    assert last_ann.get_sub_category(names.C.CS).category_id == datapoint.get_last_ann_sub_category_col_span_id()

    summary_ann = dp.summary
    assert isinstance(summary_ann, SummaryAnnotation)
    assert summary_ann.get_sub_category(names.C.NR).category_id == datapoint.get_summary_ann_sub_category_rows_id()
    assert summary_ann.get_sub_category(names.C.NC).category_id == datapoint.get_summary_ann_sub_category_col_id()
    assert summary_ann.get_sub_category(names.C.NRS).category_id == datapoint.get_summary_ann_sub_category_row_span_id()
    assert summary_ann.get_sub_category(names.C.NCS).category_id == datapoint.get_summary_ann_sub_category_col_span_id()


@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_pub_to_image_when_items_are_added(
    datapoint_pubtabnet: JsonDict, categories_name_as_key_pubtabnet: Dict[str, str]
) -> None:
    """
    testing pub_to_image is adding items "ROW" and "COL" correctly
    """

    load_image = False
    # Act
    pub_to_image_mapper = pub_to_image(  # type: ignore # pylint: disable=E1120  # 259
        categories_name_as_key_pubtabnet, load_image, True, True
    )
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    assert dp is not None

    test_anns = dp.get_annotation(category_names=names.C.ITEM)
    summary_ann = dp.summary
    assert isinstance(summary_ann, SummaryAnnotation)
    assert isinstance(test_anns, list)

    assert len(test_anns) == int(summary_ann.get_sub_category(names.C.NR).category_id) + int(
        summary_ann.get_sub_category(names.C.NC).category_id
    )
