# -*- coding: utf-8 -*-
# File: test_annotation.py

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
Testing the module datapoint.annotation
"""
import pytest
from pytest import mark

from deepdoctection.datapoint import CategoryAnnotation
from deepdoctection.utils import get_uuid

from .conftest import CatAnn


class TestCategoryAnnotation:
    """
    Testing Category Annotation methods
    """

    @staticmethod
    @mark.parametrize(
        "external_id,expected",
        [
            (CatAnn.external_id, CatAnn.get_annotation_id("n")),
            (CatAnn.uuid, CatAnn.get_annotation_id("u")),
        ],
    )
    def test_annotation_id_is_correctly_assigned(external_id: str, expected: str) -> None:
        """
        Annotation id is assigned as expected, provided that it is passed as external id
        :param external_id: external_id
        :param expected: annotation_id
        """

        # Arrange
        test_cat = CategoryAnnotation(category_name="FOO", category_id="1", external_id=external_id)

        # Assert
        assert test_cat.annotation_id == expected

    @staticmethod
    def test_dump_sup_cat_and_check_ann_id() -> None:
        """
        Sub categories are dumped to category instance and annotation ids are correctly assigned
        """

        # Arrange
        cat = CategoryAnnotation(category_name="FOO", category_id="1")
        sub_cat_1 = CategoryAnnotation(category_name="BAK", category_id="2")
        sub_cat_2 = CategoryAnnotation(category_name="BAZ", category_id="3")

        # Act
        cat.dump_sub_category("bak", sub_cat_1)
        cat.dump_sub_category("baz", sub_cat_2, "c822f8c3-1148-30c4-90eb-cb4896b1ebe5")

        export_sub_cat_1 = cat.get_sub_category("bak")
        export_sub_cat_2 = cat.get_sub_category("baz")

        # Assert
        if export_sub_cat_1 is not None and export_sub_cat_2 is not None:
            tmp_annotation_id_cat = "6251d1c6-856f-3eac-b73e-db1d300852a3"
            assert export_sub_cat_1.annotation_id == get_uuid("BAK2" + tmp_annotation_id_cat)
            assert export_sub_cat_2.annotation_id == get_uuid(
                "BAZ3" + tmp_annotation_id_cat, "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
            )

    @staticmethod
    def test_remove_sub_cat() -> None:
        """
        Existing sub categories are correctly removed
        """

        # Arrange
        cat = CategoryAnnotation(category_name="FOO", category_id="1")
        sub_cat_1 = CategoryAnnotation(category_name="BAK", category_id="2")
        cat.dump_sub_category("bak", sub_cat_1)

        # Act
        cat.remove_sub_category("bak")

        # Assert
        with pytest.raises(Exception):
            cat.get_sub_category("bak")
