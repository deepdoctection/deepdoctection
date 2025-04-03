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

from deepdoctection.datapoint import BoundingBox, CategoryAnnotation, Image, ImageAnnotation
from deepdoctection.utils import get_uuid
from deepdoctection.utils.settings import get_type

from .conftest import CatAnn, WhiteImage


class TestCategoryAnnotation:
    """
    Testing Category Annotation methods
    """

    @staticmethod
    @mark.basic
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
        test_cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id=external_id)

        # Assert
        assert test_cat.annotation_id == expected

    @staticmethod
    @mark.basic
    def test_dump_sup_cat_and_check_ann_id() -> None:
        """
        Sub categories are dumped to category instance and annotation ids are correctly assigned
        """

        # Arrange
        cat = CategoryAnnotation(category_name="FOO", category_id=1)
        sub_cat_1 = CategoryAnnotation(category_name="BAK", category_id=2)
        sub_cat_2 = CategoryAnnotation(category_name="BAZ", category_id=3)

        # Act
        cat.dump_sub_category(get_type("bak"), sub_cat_1)
        cat.dump_sub_category(get_type("baz"), sub_cat_2, "c822f8c3-1148-30c4-90eb-cb4896b1ebe5")

        export_sub_cat_1 = cat.get_sub_category(get_type("bak"))
        export_sub_cat_2 = cat.get_sub_category(get_type("baz"))

        # Assert
        if export_sub_cat_1 is not None and export_sub_cat_2 is not None:
            tmp_annotation_id_cat = "8295618a-978d-3aad-bea9-b4faa3061ab6"
            assert export_sub_cat_1.annotation_id == get_uuid("TestType.BAK2" + tmp_annotation_id_cat)
            assert export_sub_cat_2.annotation_id == get_uuid(
                "TestType.BAZ3" + tmp_annotation_id_cat, "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
            )

    @staticmethod
    @mark.basic
    def test_dump_sub_cat_with_external_id() -> None:
        """
        Sub categories are dumped to category instance and external category ids will be annotation ids
        """

        # Arrange
        cat = CategoryAnnotation(category_name="FOO", category_id=1)
        sub_cat_1 = CategoryAnnotation(
            category_name="FOOBAK", category_id=2, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        # Act
        cat.dump_sub_category(get_type("bak"), sub_cat_1)

        export_sub_cat_1 = cat.get_sub_category(get_type("bak"))

        # Assert
        if export_sub_cat_1 is not None:
            assert export_sub_cat_1.annotation_id == "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"

    @staticmethod
    @mark.basic
    def test_remove_sub_cat() -> None:
        """
        Existing sub categories are correctly removed
        """

        # Arrange
        cat = CategoryAnnotation(category_name="FOO", category_id=1)
        sub_cat_1 = CategoryAnnotation(category_name="BAK", category_id=2)
        cat.dump_sub_category(get_type("bak"), sub_cat_1)

        # Act
        cat.remove_sub_category(get_type("bak"))

        # Assert
        with pytest.raises(Exception):
            cat.get_sub_category(get_type("bak"))

    @staticmethod
    @mark.basic
    def test_state_id() -> None:
        """
        state_id is correctly determined based by given state attributes
        """

        # Arrange
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="c822f8c3-1148-30c4-90eb-cb4896b1e222")
        sub_cat_1 = CategoryAnnotation(
            category_name="FOOBAK", category_id=2, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        # Act
        cat.dump_sub_category(get_type("bak"), sub_cat_1)

        # Assert
        assert cat.state_id == "8301eef5-bd4a-3c10-9dad-7cae94dd150b"

    @staticmethod
    @mark.basic
    def test_get_bounding_box(image: WhiteImage) -> None:
        """
        Bounding boxes are getting returned correctly
        """

        # Arrange
        cat = ImageAnnotation(
            category_name="FOO",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )
        cat.image = Image(location=image.loc, file_name=image.file_name)
        cat.image.set_embedding(
            cat.image.image_id, BoundingBox(ulx=4.0, uly=5.0, width=1.0, height=2.0, absolute_coords=True)
        )

        # Act
        box = cat.get_bounding_box()
        image_box = cat.get_bounding_box(cat.image.image_id)

        # Assert
        assert box == BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True)
        assert image_box == BoundingBox(ulx=4.0, uly=5.0, width=1.0, height=2.0, absolute_coords=True)
