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

from pytest import mark

from deepdoctection.datapoint.box import BoundingBox
from deepdoctection.mapper import pub_to_image
from deepdoctection.utils.settings import CellType, LayoutType, TableType
from deepdoctection.utils.types import JsonDict

from .conftest import get_pubtabnet_white_image
from .data import DatapointPubtabnet


@mark.basic
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
    pub_to_image_mapper = pub_to_image(categories_name_as_key_pubtabnet, load_image, True, False, False, False, False)
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    datapoint = pubtabnet_results
    assert dp is not None
    test_anns = dp.get_annotation(category_names=LayoutType.CELL)

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
    assert (
        first_ann.get_sub_category(CellType.HEADER).category_name == datapoint.get_first_ann_sub_category_header_name()
    )
    assert (
        first_ann.get_sub_category(CellType.ROW_NUMBER).category_id
        == datapoint.get_first_ann_sub_category_row_number_id()
    )
    assert (
        first_ann.get_sub_category(CellType.COLUMN_NUMBER).category_id
        == datapoint.get_first_ann_sub_category_col_number_id()
    )
    assert (
        first_ann.get_sub_category(CellType.ROW_SPAN).category_id == datapoint.get_first_ann_sub_category_row_span_id()
    )
    assert (
        first_ann.get_sub_category(CellType.COLUMN_SPAN).category_id
        == datapoint.get_first_ann_sub_category_col_span_id()
    )

    assert last_ann.category_name == datapoint.get_last_ann_category_name()
    assert last_ann.get_sub_category(CellType.HEADER).category_name == datapoint.get_last_ann_sub_category_header_name()
    assert (
        last_ann.get_sub_category(CellType.ROW_NUMBER).category_id
        == datapoint.get_last_ann_sub_category_row_number_id()
    )
    assert (
        last_ann.get_sub_category(CellType.COLUMN_NUMBER).category_id
        == datapoint.get_last_ann_sub_category_col_number_id()
    )
    assert last_ann.get_sub_category(CellType.ROW_SPAN).category_id == datapoint.get_last_ann_sub_category_row_span_id()
    assert (
        last_ann.get_sub_category(CellType.COLUMN_SPAN).category_id == datapoint.get_last_ann_sub_category_col_span_id()
    )

    summary_ann = dp.summary
    assert (
        summary_ann.get_sub_category(TableType.NUMBER_OF_ROWS).category_id
        == datapoint.get_summary_ann_sub_category_rows_id()
    )
    assert (
        summary_ann.get_sub_category(TableType.NUMBER_OF_COLUMNS).category_id
        == datapoint.get_summary_ann_sub_category_col_id()
    )
    assert (
        summary_ann.get_sub_category(TableType.MAX_ROW_SPAN).category_id
        == datapoint.get_summary_ann_sub_category_row_span_id()
    )
    assert (
        summary_ann.get_sub_category(TableType.MAX_COL_SPAN).category_id
        == datapoint.get_summary_ann_sub_category_col_span_id()
    )


@mark.basic
@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_pub_to_image_when_items_are_added(
    datapoint_pubtabnet: JsonDict, categories_name_as_key_pubtabnet: Dict[str, str]
) -> None:
    """
    testing pub_to_image is adding items "ROW" and "COL" correctly
    """

    load_image = False
    # Act
    pub_to_image_mapper = pub_to_image(categories_name_as_key_pubtabnet, load_image, True, True, False, False, False)
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    assert dp is not None

    test_anns = dp.get_annotation(category_names=TableType.ITEM)
    summary_ann = dp.summary

    assert (
        len(test_anns)
        == summary_ann.get_sub_category(TableType.NUMBER_OF_ROWS).category_id
        + summary_ann.get_sub_category(TableType.NUMBER_OF_COLUMNS).category_id
    )


@mark.basic
@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_pub_to_image_when_dd_pipe_like(
    datapoint_pubtabnet: JsonDict, categories_name_as_key_pubtabnet: Dict[str, str]
) -> None:
    """
    testing pub_to_image generates Image like dd-analyzer
    """

    # Act
    pub_to_image_mapper = pub_to_image(categories_name_as_key_pubtabnet, True, True, True, True, False, False)
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    assert dp is not None
    table_list = dp.get_annotation(category_names=LayoutType.TABLE)
    assert len(table_list) == 1
    table = table_list[0]
    assert table.image is not None
