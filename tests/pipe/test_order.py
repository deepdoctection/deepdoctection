# -*- coding: utf-8 -*-
# File: test_order.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Testing module pipe.order
"""

from pytest import mark

from deepdoctection.datapoint import BoundingBox, ContainerAnnotation, Image, ImageAnnotation
from deepdoctection.pipe.order import TextOrderService
from deepdoctection.utils.settings import LayoutType, Relationships, WordType


class TestTextOrderService:
    """
    Test TextOrderService
    """

    @staticmethod
    @mark.basic
    def test_integration_pipeline_component(dp_image_with_layout_and_word_annotations: Image) -> None:
        """
        test integration_pipeline_component
        """

        # Arrange
        text_order_service = TextOrderService(
            text_container=LayoutType.WORD,
            text_block_categories=[
                LayoutType.TITLE,
                LayoutType.TEXT,
                LayoutType.LIST,
                LayoutType.CELL,
            ],
            floating_text_block_categories=[LayoutType.TITLE, LayoutType.TEXT, LayoutType.LIST],
        )
        dp_image = dp_image_with_layout_and_word_annotations

        # Act
        text_order_service.pass_datapoint(dp_image)

        # Assert
        layout_anns = dp_image.get_annotation(category_names=[LayoutType.TITLE, LayoutType.TEXT])
        word_anns = dp_image.get_annotation(category_names=LayoutType.WORD)

        # only need to check on layout_anns and word_anns, if sub cats have been added
        # and numbers are correctly assigned

        sub_cat = layout_anns[0].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = layout_anns[1].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"

        sub_cat = word_anns[0].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[1].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"
        sub_cat = word_anns[2].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[3].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"

    @staticmethod
    @mark.basic
    def test_integration_pipeline_component_wit_include_residual_text_container_to_false(
        dp_image_with_layout_and_word_annotations: Image,
    ) -> None:
        """
        test integration_pipeline_component_wit_include_residual_text_container_to_false
        """

        # Arrange
        text_order_service = TextOrderService(
            text_container=LayoutType.WORD,
            text_block_categories=[
                LayoutType.TITLE,
                LayoutType.TEXT,
                LayoutType.LIST,
                LayoutType.CELL,
            ],
            floating_text_block_categories=[LayoutType.TITLE, LayoutType.TEXT, LayoutType.LIST],
            include_residual_text_container=False,
        )

        dp_image = dp_image_with_layout_and_word_annotations

        residual_word_ann = ImageAnnotation(
            bounding_box=BoundingBox(ulx=350.0, uly=390.0, lrx=355.0, lry=395.0, absolute_coords=True),
            score=0.6,
            category_name=LayoutType.WORD,
            category_id="8",
        )
        dp_image.dump(residual_word_ann)
        dp_image.image_ann_to_image(residual_word_ann.annotation_id)
        residual_word_ann.dump_sub_category(
            WordType.CHARACTERS, ContainerAnnotation(category_name=WordType.CHARACTERS, value="residual")
        )

        # Act
        text_order_service.pass_datapoint(dp_image)

        # Assert
        layout_anns = dp_image.get_annotation(category_names=[LayoutType.TITLE, LayoutType.TEXT])
        word_anns = dp_image.get_annotation(category_names=LayoutType.WORD)

        sub_cat = layout_anns[0].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = layout_anns[1].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"

        sub_cat = word_anns[0].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[1].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"
        sub_cat = word_anns[2].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[3].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"

        assert Relationships.READING_ORDER not in word_anns[4].sub_categories

    @staticmethod
    @mark.basic
    def test_integration_pipeline_component_wit_include_residual_text_container_to_true(
        dp_image_with_layout_and_word_annotations: Image,
    ) -> None:
        """
        test integration_pipeline_component_wit_include_residual_text_container_to_true
        """

        # Arrange
        text_order_service = TextOrderService(
            text_container=LayoutType.WORD,
            text_block_categories=[
                LayoutType.TITLE,
                LayoutType.TEXT,
                LayoutType.LIST,
                LayoutType.CELL,
            ],
            floating_text_block_categories=[LayoutType.TITLE, LayoutType.TEXT, LayoutType.LIST],
        )

        dp_image = dp_image_with_layout_and_word_annotations

        residual_word_ann = ImageAnnotation(
            bounding_box=BoundingBox(ulx=350.0, uly=390.0, lrx=355.0, lry=395.0, absolute_coords=True),
            score=0.6,
            category_name=LayoutType.WORD,
            category_id="8",
        )
        dp_image.dump(residual_word_ann)
        dp_image.image_ann_to_image(residual_word_ann.annotation_id)
        residual_word_ann.dump_sub_category(
            WordType.CHARACTERS, ContainerAnnotation(category_name=WordType.CHARACTERS, value="residual")
        )

        # Act
        text_order_service.pass_datapoint(dp_image)

        # Assert
        layout_anns = dp_image.get_annotation(category_names=[LayoutType.TITLE, LayoutType.TEXT, LayoutType.LINE])
        word_anns = dp_image.get_annotation(category_names=LayoutType.WORD)

        sub_cat = layout_anns[0].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = layout_anns[1].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"
        sub_cat = layout_anns[2].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "3"
        relation = layout_anns[2].get_relationship(Relationships.CHILD)
        assert residual_word_ann.annotation_id in relation

        sub_cat = word_anns[0].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[1].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"
        sub_cat = word_anns[2].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[3].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "2"
        sub_cat = word_anns[4].get_sub_category(Relationships.READING_ORDER)
        assert sub_cat.category_id == "1"
