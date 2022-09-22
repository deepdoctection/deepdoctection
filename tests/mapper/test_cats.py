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

from deepdoctection.datapoint import CategoryAnnotation, Image, SummaryAnnotation
from deepdoctection.mapper import cat_to_sub_cat, filter_cat, filter_summary, image_to_cat_id, pub_to_image, remove_cats
from deepdoctection.utils.detection_types import JsonDict

# from deepdoctection.utils.settings import names
from deepdoctection.utils.settings import CellType, LayoutType, TableType, get_type

from .conftest import get_pubtabnet_white_image
from .data import DatapointPubtabnet


@pytest.mark.basic
@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_cat_to_sub_cat(datapoint_pubtabnet: JsonDict, pubtabnet_results: DatapointPubtabnet) -> None:
    """
    test func: cat_to_sub_cat replaces categories with sub categories correctly
    """
    # Arrange
    categories_name_as_key_init = {LayoutType.cell: "1", TableType.item: "2"}
    pub_to_image_mapper = pub_to_image(categories_name_as_key_init, False, False, True, False, False)
    dp = pub_to_image_mapper(datapoint_pubtabnet)

    categories = MagicMock()
    categories._cat_to_sub_cat = {  # pylint: disable=W0212
        LayoutType.cell: CellType.header,
        TableType.item: TableType.item,
    }
    categories.get_categories = Mock(
        return_value={CellType.header: "1", CellType.body: "2", LayoutType.row: "3", LayoutType.column: "4"}
    )

    datapoint = pubtabnet_results

    # Act
    if dp is not None:
        cat_to_sub_cat_mapper = cat_to_sub_cat(
            categories.get_categories(), categories._cat_to_sub_cat  # pylint: disable=W0212
        )
        dp = cat_to_sub_cat_mapper(dp)

    if dp is not None:
        heads = dp.get_annotation(category_names=CellType.header)
        bodies = dp.get_annotation(category_names=CellType.body)
        rows = dp.get_annotation(category_names=LayoutType.row)
        cols = dp.get_annotation(category_names=LayoutType.column)

        # Assert
        assert len(heads) == datapoint.get_number_of_heads()
        assert len(bodies) == datapoint.get_number_of_bodies()
        assert str(len(rows)) == datapoint.get_summary_ann_sub_category_rows_id()
        assert str(len(cols)) == datapoint.get_summary_ann_sub_category_col_id()


@pytest.mark.basic
@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_pubtabnet_white_image))
def test_filter_categories(datapoint_pubtabnet: JsonDict, pubtabnet_results: DatapointPubtabnet) -> None:
    """
    test func:`filter_categories` removes categories correctly. Also tests that category ids are re assigned correctly
    """

    # Arrange
    categories_name_as_key_init = {LayoutType.cell: "1", TableType.item: "2"}
    pub_to_image_mapper = pub_to_image(categories_name_as_key_init, False, False, True, False, False)
    dp = pub_to_image_mapper(datapoint_pubtabnet)
    assert dp is not None

    categories = MagicMock()
    categories.get_categories = Mock(return_value=[TableType.item])

    datapoint = pubtabnet_results

    # Act
    filter_cat_mapper = filter_cat(  # pylint: disable=E1120
        categories.get_categories(as_dict=False, filtered=True),
        list(categories_name_as_key_init.keys()),
    )
    dp = filter_cat_mapper(dp)

    items = dp.get_annotation(category_names=TableType.item)
    cells = dp.get_annotation(category_names=LayoutType.cell)

    # Assert
    assert len(items) == int(datapoint.get_summary_ann_sub_category_rows_id()) + int(
        datapoint.get_summary_ann_sub_category_col_id()
    )
    assert items[0].category_id == "1"
    assert len(cells) == 0


@pytest.mark.basic
def test_filter_summary_1(datapoint_image_with_summary: Image) -> None:
    """
    test func:`filter_summary` does not filter dataset, if condition is satisfied.
    """

    # Arrange
    output = filter_summary({"BAK": "FOO"}, True)(datapoint_image_with_summary)

    # Assert
    assert output is not None


@pytest.mark.basic
def test_filter_summary_2(datapoint_image_with_summary: Image) -> None:
    """
    test func:`filter_summary`  does filter dataset, if condition is not satisfied.
    """

    # Arrange
    output = filter_summary({"BAK": ["BAZ"]}, True)(datapoint_image_with_summary)

    # Assert
    assert output is None


@pytest.mark.basic
def test_image_to_cat_id_1(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    category_names = LayoutType.table
    expected_output = [2]

    # Act
    output, _ = image_to_cat_id(category_names)(dp_image_fully_segmented)  # pylint: disable = E1102

    # Assert
    assert output[LayoutType.table] == expected_output


@pytest.mark.basic
def test_image_to_cat_id_2(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    category_names = [LayoutType.table, LayoutType.row, LayoutType.column]
    expected_output = {LayoutType.table: [2], LayoutType.row: [6, 6], LayoutType.column: [7, 7]}

    # Act
    output, _ = image_to_cat_id(category_names)(dp_image_fully_segmented)  # pylint: disable = E1102

    # Assert
    assert output[LayoutType.table] == expected_output[LayoutType.table]
    assert output[LayoutType.row] == expected_output[LayoutType.row]
    assert output[LayoutType.column] == expected_output[LayoutType.column]


@pytest.mark.basic
def test_image_to_cat_id_3(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    sub_category_names = {LayoutType.cell: CellType.row_span}
    expected_output = {CellType.row_span: [1, 1, 1, 1, 0]}

    # Act
    output, _ = image_to_cat_id(sub_categories=sub_category_names)(dp_image_fully_segmented)  # pylint: disable = E1102

    # Assert
    assert output[CellType.row_span] == expected_output[CellType.row_span]


@pytest.mark.basic
def test_image_to_cat_id_4(dp_image_fully_segmented: Image) -> None:
    """
    test func: image_to_cat_id returns extraction of category_ids
    """

    # Arrange
    sub_category_names = {
        LayoutType.cell: [CellType.row_number, CellType.row_span, CellType.column_number, CellType.column_span]
    }
    expected_output = {
        CellType.row_number: [1, 2, 1, 2, 0],
        CellType.row_span: [1, 1, 1, 1, 0],
        CellType.column_number: [1, 1, 2, 2, 0],
        CellType.column_span: [1, 1, 1, 1, 0],
    }

    # Act
    output, _ = image_to_cat_id(sub_categories=sub_category_names)(dp_image_fully_segmented)  # pylint: disable = E1102

    # Assert
    assert output[CellType.row_number] == expected_output[CellType.row_number]
    assert output[CellType.row_span] == expected_output[CellType.row_span]
    assert output[CellType.column_number] == expected_output[CellType.column_number]
    assert output[CellType.column_span] == expected_output[CellType.column_span]


@pytest.mark.basic
def test_remove_cats(dp_image_fully_segmented: Image) -> None:
    """
    test func: remove_cats returns datapoint with removed categories
    """

    # Arrange
    categories = LayoutType.row

    # Act
    remove_cats_mapper = remove_cats(category_names=categories)  # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp_image_fully_segmented)

    # Assert
    anns = dp.get_annotation(category_names=LayoutType.row)
    assert len(anns) == 0


@pytest.mark.basic
def test_remove_cats_2(dp_image_fully_segmented: Image) -> None:
    """
    test func: remove_cats returns datapoint with removed sub categories
    """

    # Arrange
    sub_categories = {LayoutType.cell: [CellType.row_number, CellType.row_span], LayoutType.row: CellType.row_number}

    # Act
    remove_cats_mapper = remove_cats(sub_categories=sub_categories)  # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp_image_fully_segmented)

    # Assert
    anns_row = dp.get_annotation(category_names=LayoutType.row)
    anns_cell = dp.get_annotation(category_names=LayoutType.cell)

    assert len(anns_row) == 2
    assert len(anns_cell) == 5

    first_ann_row = anns_row[0]

    with pytest.raises(Exception):
        first_ann_row.get_sub_category(CellType.row_number)

    first_ann_cell = anns_cell[0]
    scd_ann_cell = anns_cell[1]

    with pytest.raises(Exception):
        first_ann_cell.get_sub_category(CellType.row_number)

    with pytest.raises(Exception):
        first_ann_cell.get_sub_category(CellType.row_span)

    assert isinstance(first_ann_cell.get_sub_category(CellType.column_number), CategoryAnnotation)
    assert isinstance(first_ann_cell.get_sub_category(CellType.column_span), CategoryAnnotation)

    with pytest.raises(Exception):
        scd_ann_cell.get_sub_category(CellType.row_number)

    with pytest.raises(Exception):
        scd_ann_cell.get_sub_category(CellType.row_span)

    assert isinstance(scd_ann_cell.get_sub_category(CellType.column_number), CategoryAnnotation)
    assert isinstance(scd_ann_cell.get_sub_category(CellType.column_span), CategoryAnnotation)


@pytest.mark.basic
def test_remove_cats_3(dp_image_fully_segmented: Image) -> None:
    """
    test func: remove_cats removes summary sub category
    """

    # Arrange
    sub_category_ann = CategoryAnnotation(category_name="TEST")
    summary = SummaryAnnotation()
    summary.dump_sub_category(get_type("TEST_SUMMARY"), sub_category_ann)
    dp_image_fully_segmented.summary = summary

    # Act
    remove_cats_mapper = remove_cats(summary_sub_categories="TEST_SUMMARY")  # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp_image_fully_segmented)
    assert dp.summary is not None

    # Assert
    with pytest.raises(Exception):
        dp.summary.get_sub_category(get_type("TEST_SUMMARY"))
