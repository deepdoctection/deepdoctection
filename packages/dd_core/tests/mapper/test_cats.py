# -*- coding: utf-8 -*-
# File: test_cats.py

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
A set of test functions for validating category and subcategory mapping, filtering,
and id reassignment for images and their annotations in a data processing pipeline.

These tests validate the correctness of the following core operations:
- Mapping categories to subcategories using `cat_to_sub_cat`.
- Reassigning or filtering category IDs with `re_assign_cat_ids` and `filter_cat`.
- Extracting and filtering summary data using `filter_summary`.
- Mapping image annotations to category IDs with `image_to_cat_id`.
- Removing specific categories from annotations with `remove_cats`.
"""


from dd_core.datapoint.image import Image
from dd_core.mapper.cats import (
    cat_to_sub_cat,
    filter_cat,
    filter_summary,
    image_to_cat_id,
    re_assign_cat_ids,
    remove_cats,
)


def test_cat_to_sub_word_to_characters_with_id_mapping(image: Image) -> None:
    """
    test func: cat_to_sub_cat replaces categories with sub categories correctly
    """
    categories_dict_names_as_key = {"characters": 2}
    cat_to_sub_cat_dict = {"word": "characters"}
    dp = cat_to_sub_cat(
        categories_dict_names_as_key=categories_dict_names_as_key, cat_to_sub_cat_dict=cat_to_sub_cat_dict
    )(image)
    anns = dp.get_annotation(category_names=["characters"])
    assert len(anns) > 0
    for ann in anns:
        assert ann.category_id == 2

    anns = dp.get_annotation(category_names=["text", "section_header"])
    assert len(anns) > 0
    for ann in anns:
        if ann.category_name == "text":
            assert ann.category_id == 11
        elif ann.category_name == "section_header":
            assert ann.category_id == 9


def test_cat_to_sub_word_to_characters_no_category_id_change(image: Image) -> None:
    """
    test func: cat_to_sub_cat replaces categories with sub categories correctly
    """
    cat_to_sub_cat_dict = {"word": "characters"}

    dp = cat_to_sub_cat(cat_to_sub_cat_dict=cat_to_sub_cat_dict)(image)

    anns = dp.get_annotation(category_names=["characters"])
    assert len(anns) > 0
    for ann in anns:
        assert ann.category_name == "characters"
        assert ann.category_id == -1  # default value


def test_re_assign_cat_ids(image: Image) -> None:
    """Test re_assign_cat_ids keeps only word and text with correct ids."""
    categories_dict_name_as_key = {"word": 1, "text": 2}

    dp = re_assign_cat_ids(categories_dict_name_as_key=categories_dict_name_as_key)(image)

    anns = dp.get_annotation()
    assert len(anns) > 0

    allowed = {"word", "text"}

    for ann in anns:
        assert ann.category_name in allowed
        assert ann.category_id == categories_dict_name_as_key[ann.category_name]

    assert {ann.category_name for ann in anns}.issubset(allowed)


def test_filter_cat(image: Image) -> None:
    """Test filter_cat keeps only figure,row,column and assigns ids 1,2,3."""
    categories_as_list_filtered = ["figure", "row", "column"]
    categories_as_list_unfiltered = [
        "text",
        "page_footer",
        "caption",
        "cell",
        "table",
        "line",
        "column",
        "word",
        "row",
        "figure",
        "section_header",
        "page_header",
        "column_header",
    ]

    dp = filter_cat(
        categories_as_list_filtered=categories_as_list_filtered,
        categories_as_list_unfiltered=categories_as_list_unfiltered,
    )(image)

    anns = dp.get_annotation()

    allowed = {"figure", "row", "column"}
    id_map = {"figure": 1, "row": 2, "column": 3}

    # Only allowed categories remain
    assert {ann.category_name for ann in anns}.issubset(allowed)

    # Each remaining category has the correct reassigned id
    for ann in anns:
        assert ann.category_id == id_map[ann.category_name]


def test_filter_summary_rows_ids_none(table_image: Image) -> None:
    """Should return None when number_of_rows id not in provided list."""
    result = filter_summary(
        sub_cat_to_sub_cat_names_or_ids={"number_of_rows": [1, 2, 3, 4]},
        mode="id",
    )(table_image)
    assert result is None


def test_filter_summary_rows_ids_return(table_image: Image) -> None:
    """Should return image when number_of_rows id matches one of provided ids."""
    result = filter_summary(
        sub_cat_to_sub_cat_names_or_ids={"number_of_rows": [5, 6]},
        mode="id",
    )(table_image)
    assert result is not None
    assert result.image_id == table_image.image_id


def test_filter_summary_columns_name_return(table_image: Image) -> None:
    """Should return image when number_of_columns name matches."""
    result = filter_summary(
        sub_cat_to_sub_cat_names_or_ids={"number_of_columns": ["number_of_columns"]},
        mode="name",
    )(table_image)
    assert result is not None
    assert result.image_id == table_image.image_id


def test_filter_summary_columns_name_none(table_image: Image) -> None:
    """Should return None when number_of_columns name does not match."""
    result = filter_summary(
        sub_cat_to_sub_cat_names_or_ids={"number_of_columns": ["non_existent"]},
        mode="name",
    )(table_image)
    assert result is None


def test_image_to_cat_id_basic_categories(table_image: Image) -> None:
    """Extract ids for column, row, cell categories."""
    result, img_id = image_to_cat_id(category_names=["column", "row", "cell"])(table_image)  # pylint:disable=E1102
    assert result == {
        "column": [2] * 3,
        "row": [3] * 5,
        "cell": [-1] * 15,
    }
    assert img_id == table_image.image_id


def test_image_to_cat_id_subcategory_ids(table_image: Image) -> None:
    """Extract sub-category ids for column_number."""
    result, img_id = image_to_cat_id( # pylint:disable=E1102
        category_names=["column"],
        sub_categories={"column": "column_number"},
    )(table_image)
    assert result == {
        "column": [2, 2, 2],
        "column_number": [3, 1, 2],
    }
    assert img_id == table_image.image_id


def test_image_to_cat_id_subcategory_names(table_image: Image) -> None:
    """Extract sub-category names for column_number."""
    result, img_id = image_to_cat_id( # pylint:disable=E1102
        category_names=["column"],
        sub_categories={"column": "column_number"},
        id_name_or_value="name",
    )(table_image)
    assert result == {
        "column": [2] * 3,
        "column_number": ["column_number"] * 3,
    }
    assert img_id == table_image.image_id


def test_image_to_cat_id_summary_ids(table_image: Image) -> None:
    """Extract summary sub-category ids."""
    result, img_id = image_to_cat_id( # pylint:disable=E1102
        summary_sub_category_names=[
            "number_of_rows",
            "number_of_columns",
            "max_row_span",
            "max_col_span",
        ]
    )(table_image)
    assert result == {
        "number_of_rows": [5],
        "number_of_columns": [3],
        "max_row_span": [1],
        "max_col_span": [1],
    }
    assert img_id == table_image.image_id


def test_image_to_cat_id_summary_names(table_image: Image) -> None:
    """Extract summary sub-category names."""
    result, img_id = image_to_cat_id( # pylint:disable=E1102
        summary_sub_category_names=[
            "number_of_rows",
            "number_of_columns",
            "max_row_span",
            "max_col_span",
        ],
        id_name_or_value="name",
    )(table_image)
    assert result == {
        "number_of_rows": ["number_of_rows"],
        "number_of_columns": ["number_of_columns"],
        "max_row_span": ["max_row_span"],
        "max_col_span": ["max_col_span"],
    }
    assert img_id == table_image.image_id


def test_remove_cats_category_names(image: Image) -> None:
    """Remove text category annotations."""
    dp = remove_cats(category_names=["text"])(image)
    anns = dp.get_annotation()
    assert all(ann.category_name != "text" for ann in anns)


def test_remove_cats_sub_categories(image: Image) -> None:
    """Remove characters subcategory from text annotations."""
    dp = remove_cats(sub_categories={"text": "characters"})(image)
    for ann in dp.get_annotation(category_names=["text"]):
        assert "characters" not in ann.sub_categories


def test_remove_cats_relationships(image: Image) -> None:
    """Remove child relationship from text annotations."""
    dp = remove_cats(relationships={"text": "child"})(image)
    for ann in dp.get_annotation(category_names=["text"]):
        assert len(ann.relationships["child"]) == 0  # type: ignore


def test_remove_cats_category_and_sub_category(image: Image) -> None:
    """Remove text category and its characters subcategory."""
    dp = remove_cats(category_names=["text"], sub_categories={"text": "characters"})(image)
    assert all(ann.category_name != "text" for ann in dp.get_annotation())


def test_remove_cats_category_and_relationship(image: Image) -> None:
    """Remove text category and its child relationship."""
    dp = remove_cats(category_names=["text"], relationships={"text": "child"})(image)
    assert all(ann.category_name != "text" for ann in dp.get_annotation())


def test_remove_cats_sub_category_and_relationship(image: Image) -> None:
    """Remove characters subcategory and child relationship from text."""
    dp = remove_cats(sub_categories={"text": "characters"}, relationships={"text": "child"})(image)
    for ann in dp.get_annotation(category_names=["text"]):
        assert "characters" not in ann.sub_categories
        assert len(ann.relationships["child"]) == 0  # type: ignore


def test_remove_cats_all(image: Image) -> None:
    """Remove text category, characters subcategory and child relationship."""
    dp = remove_cats(
        category_names=["text"],
        sub_categories={"text": "characters"},
        relationships={"text": "child"},
    )(image)
    assert all(ann.category_name != "text" for ann in dp.get_annotation())


def test_remove_cats_summary_sub_categories(table_image: Image) -> None:
    """Remove number_of_rows and number_of_columns from summary."""
    dp = remove_cats(summary_sub_categories=["number_of_rows", "number_of_columns"])(table_image)
    assert "number_of_rows" not in dp.summary.sub_categories
    assert "number_of_columns" not in dp.summary.sub_categories
    assert "max_row_span" in dp.summary.sub_categories
    assert "max_col_span" in dp.summary.sub_categories
