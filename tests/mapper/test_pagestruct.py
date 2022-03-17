# -*- coding: utf-8 -*-
# File: test_pagestruct.py

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
Testing the module mapper.pagestruct
"""

from deepdoctection.datapoint import CategoryAnnotation, Image
from deepdoctection.mapper.pagestruct import page_dict_to_page, to_page
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.settings import names

from .data import DatapointPageDict


def test_to_page(dp_image_with_layout_and_word_annotations: Image) -> None:
    """
    test to_page
    """

    # Arrange
    dp_image = dp_image_with_layout_and_word_annotations
    title_ann = dp_image.get_annotation(category_names=[names.C.TITLE])[0]
    title_ann.dump_sub_category(names.C.RO, CategoryAnnotation(category_name=names.C.RO, category_id=1))
    text_ann = dp_image.get_annotation(category_names=[names.C.TEXT])[0]
    text_ann.dump_sub_category(names.C.RO, CategoryAnnotation(category_name=names.C.RO, category_id=2))

    word_anns = dp_image.get_annotation(category_names=names.C.WORD)

    word_anns[0].dump_sub_category(names.C.RO, CategoryAnnotation(category_name=names.C.RO, category_id=1))
    word_anns[1].dump_sub_category(names.C.RO, CategoryAnnotation(category_name=names.C.RO, category_id=2))
    word_anns[2].dump_sub_category(names.C.RO, CategoryAnnotation(category_name=names.C.RO, category_id=1))
    word_anns[3].dump_sub_category(names.C.RO, CategoryAnnotation(category_name=names.C.RO, category_id=2))

    # Act
    page = to_page(
        dp_image,
        names.C.WORD,
        [names.C.TEXT, names.C.TITLE, names.C.LIST],
        [names.C.TEXT, names.C.TITLE, names.C.LIST, names.C.CELL],
    )

    # Assert
    assert page.get_text() == "\nhello world\nbye world"


def test_page_dict_to_page(page_dict: JsonDict) -> None:
    """
    test page_dict_to_page
    """

    # Act
    page = page_dict_to_page(page_dict)

    # Assert
    expected_results = DatapointPageDict()
    assert page.file_name == expected_results.file_name
    assert len(page.tables) == expected_results.number_tables

    table = page.tables[0]
    assert len(table.cells) == expected_results.number_cells

    first_cell = table.cells[0]
    assert first_cell.text == expected_results.first_cell_text
