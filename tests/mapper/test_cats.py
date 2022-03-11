# -*- coding: utf-8 -*-
# File: test_cats.py

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
Testing the module mapper.cats
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from deepdoctection.datapoint import CategoryAnnotation, Image
from deepdoctection.mapper import cat_to_sub_cat, filter_cat, image_to_cat_id, pub_to_image, remove_cats
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.settings import names

from .conftest import get_pubtabnet_white_image
from .data import DatapointPubtabnet


@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_cat_to_sub_cat(datapoint_pubtabnet: JsonDict, pubtabnet_results: DatapointPubtabnet) -> None:
    """
    test func: cat_to_sub_cat replaces categories with sub categories correctly
    """
    # Arrange
    categories_name_as_key_init = {names.C.CELL: "1", names.C.ITEM: "2"}
    pub_to_image_mapper = pub_to_image(categories_name_as_key_init, False, False, True)  # type: ignore # pylint: disable=E1120  # 259
    dp = pub_to_image_mapper(datapoint_pubtabnet)

    categories = MagicMock()
    categories._cat_to_sub_cat = {names.C.CELL: names.C.HEAD, names.C.ITEM: "row_col"}  # pylint: disable=W0212
    categories.get_categories = Mock(
        return_value={names.C.HEAD: "1", names.C.BODY: "2", names.C.ROW: "3", names.C.COL: "4"}
    )

    datapoint = pubtabnet_results

    # Act
    if dp is not None:
        cat_to_sub_cat_mapper = cat_to_sub_cat(  # type: ignore
            categories.get_categories(), categories._cat_to_sub_cat  # pylint: disable=W0212
        )  # pylint: disable=E1120  # 259
        dp = cat_to_sub_cat_mapper(dp)

    if dp is not None:
        heads = dp.get_annotation(category_names=names.C.HEAD)
        bodies = dp.get_annotation(category_names=names.C.BODY)
        rows = dp.get_annotation(category_names=names.C.ROW)
        cols = dp.get_annotation(category_names=names.C.COL)

        # Assert
        assert len(heads) == datapoint.get_number_of_heads()
        assert len(bodies) == datapoint.get_number_of_bodies()
        assert str(len(rows)) == datapoint.get_summary_ann_sub_category_rows_id()
        assert str(len(cols)) == datapoint.get_summary_ann_sub_category_col_id()


@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_filter_categories(datapoint_pubtabnet: JsonDict, pubtabnet_results: DatapointPubtabnet) -> None:
    """
    test func:`filter_categories` removes categories correctly. Also tests that category ids are re assigned correctly
    """

    # Arrange
    categories_name_as_key_init = {names.C.CELL: "1", names.C.ITEM: "2"}
    pub_to_image_mapper = pub_to_image(categories_name_as_key_init, False, False, True)  # type: ignore # pylint: disable=E1120  # 259
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    assert dp is not None

    categories = MagicMock()
    categories.get_categories = Mock(return_value=[names.C.ITEM])

    datapoint = pubtabnet_results

    # Act
    filter_cat_mapper = filter_cat(  # type: ignore  # pylint: disable=E1120
        categories.get_categories(as_dict=False, filtered=True),
        list(categories_name_as_key_init.keys()),
    )  # pylint: disable=E1120  # 259
    dp = filter_cat_mapper(dp)

    items = dp.get_annotation(category_names=names.C.ITEM)
    cells = dp.get_annotation(category_names=names.C.CELL)

    # Assert
    assert len(items) == int(datapoint.get_summary_ann_sub_category_rows_id()) + int(
        datapoint.get_summary_ann_sub_category_col_id()
    )
    assert items[0].category_id == "1"
    assert len(cells) == 0


def test_image_to_cat_id_1(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    category_names = names.C.TAB
    expected_output = [2]

    # Act
    image_to_cat_id_mapper = image_to_cat_id(category_names)
    output = image_to_cat_id_mapper(dp_image_fully_segmented)  # pylint: disable=E1102  # 259

    # Assert
    assert output[names.C.TAB] == expected_output


def test_image_to_cat_id_2(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    category_names = [names.C.TAB, names.C.ROW, names.C.COL]
    expected_output = {names.C.TAB: [2], names.C.ROW: [6, 6], names.C.COL: [7, 7]}

    # Act
    image_to_cat_id_mapper = image_to_cat_id(category_names)
    output = image_to_cat_id_mapper(dp_image_fully_segmented)  # pylint: disable=E1102  # 259

    # Assert
    assert output[names.C.TAB] == expected_output[names.C.TAB]
    assert output[names.C.ROW] == expected_output[names.C.ROW]
    assert output[names.C.COL] == expected_output[names.C.COL]


def test_image_to_cat_id_3(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    sub_category_names = {names.C.CELL: names.C.RS}
    expected_output = {names.C.RS: [1, 1, 1, 1, 0]}

    # Act
    image_to_cat_id_mapper = image_to_cat_id(sub_category_names=sub_category_names)  # type: ignore # pylint: disable=E1120  # 259
    output = image_to_cat_id_mapper(dp_image_fully_segmented)  # pylint: disable=E1102  # 259

    # Assert
    assert output[names.C.RS] == expected_output[names.C.RS]


def test_image_to_cat_id_4(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    sub_category_names = {names.C.CELL: [names.C.RN, names.C.RS, names.C.CN, names.C.CS]}
    expected_output = {
        names.C.RN: [1, 2, 1, 2, 0],
        names.C.RS: [1, 1, 1, 1, 0],
        names.C.CN: [1, 1, 2, 2, 0],
        names.C.CS: [1, 1, 1, 1, 0],
    }

    # Act
    image_to_cat_id_mapper = image_to_cat_id(sub_category_names=sub_category_names)  # type: ignore # pylint: disable=E1120  # 259
    output = image_to_cat_id_mapper(dp_image_fully_segmented)  # pylint: disable=E1102  # 259

    # Assert
    assert output[names.C.RN] == expected_output[names.C.RN]
    assert output[names.C.RS] == expected_output[names.C.RS]
    assert output[names.C.CN] == expected_output[names.C.CN]
    assert output[names.C.CS] == expected_output[names.C.CS]


def test_remove_cats(dp_image_fully_segmented: Image) -> None:
    """
    test func: remove_cats returns datapoint with removed categories
    """

    # Arrange
    categories = names.C.ROW

    # Act
    remove_cats_mapper = remove_cats(category_names=categories)  # type: ignore # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp_image_fully_segmented)

    # Assert
    anns = dp.get_annotation(category_names=names.C.ROW)
    assert len(anns) == 0


def test_remove_cats_2(dp_image_fully_segmented: Image) -> None:
    """
    test func: remove_cats returns datapoint with removed sub categories
    """

    # Arrange
    sub_categories = {names.C.CELL: [names.C.RN, names.C.RS], names.C.ROW: names.C.RN}

    # Act
    remove_cats_mapper = remove_cats(sub_categories=sub_categories)  # type: ignore # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp_image_fully_segmented)

    # Assert
    anns_row = dp.get_annotation(category_names=names.C.ROW)
    anns_cell = dp.get_annotation(category_names=names.C.CELL)

    assert len(anns_row) == 2
    assert len(anns_cell) == 5

    first_ann_row = anns_row[0]

    with pytest.raises(Exception):
        first_ann_row.get_sub_category(names.C.RN)

    first_ann_cell = anns_cell[0]
    scd_ann_cell = anns_cell[1]

    with pytest.raises(Exception):
        first_ann_cell.get_sub_category(names.C.RN)

    with pytest.raises(Exception):
        first_ann_cell.get_sub_category(names.C.RS)

    assert isinstance(first_ann_cell.get_sub_category(names.C.CN), CategoryAnnotation)
    assert isinstance(first_ann_cell.get_sub_category(names.C.CS), CategoryAnnotation)

    with pytest.raises(Exception):
        scd_ann_cell.get_sub_category(names.C.RN)

    with pytest.raises(Exception):
        scd_ann_cell.get_sub_category(names.C.RS)

    assert isinstance(scd_ann_cell.get_sub_category(names.C.CN), CategoryAnnotation)
    assert isinstance(scd_ann_cell.get_sub_category(names.C.CS), CategoryAnnotation)
