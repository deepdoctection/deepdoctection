# -*- coding: utf-8 -*-
# File: test_order.py

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


from pytest import mark

from dd_core.datapoint import BoundingBox, ContainerAnnotation, Image, ImageAnnotation, get_type
from deepdoctection.pipe.order import TextOrderService


def test_integration_pipeline_component(dp_image_with_layout_and_word_annotations: Image) -> None:
    """
    test integration_pipeline_component
    """

    text_order_service = TextOrderService(
        text_container=get_type("word"),
        text_block_categories=[
            get_type("title"),
            get_type("text"),
            get_type("list"),
            get_type("cell"),
        ],
        floating_text_block_categories=[get_type("title"), get_type("text"), get_type("list")],
    )
    dp_image = dp_image_with_layout_and_word_annotations

    text_order_service.pass_datapoint(dp_image)

    layout_anns = dp_image.get_annotation(category_names=[get_type("title"), get_type("text")])
    word_anns = dp_image.get_annotation(category_names=get_type("word"))

    sub_cat = layout_anns[0].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = layout_anns[1].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = word_anns[0].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = word_anns[1].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = word_anns[2].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = word_anns[3].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2


def test_integration_pipeline_component_wit_include_residual_text_container_to_false(
    dp_image_with_layout_and_word_annotations: Image,
) -> None:
    """
    test integration_pipeline_component_wit_include_residual_text_container_to_false
    """

    text_order_service = TextOrderService(
        text_container=get_type("word"),
        text_block_categories=[
            get_type("title"),
            get_type("text"),
            get_type("list"),
            get_type("cell"),
        ],
        floating_text_block_categories=[get_type("title"), get_type("text"), get_type("list")],
        include_residual_text_container=False,
    )

    dp_image = dp_image_with_layout_and_word_annotations

    residual_word_ann = ImageAnnotation(
        bounding_box=BoundingBox(ulx=350.0, uly=390.0, lrx=355.0, lry=395.0, absolute_coords=True),
        score=0.6,
        category_name=get_type("word"),
        category_id=8,
    )
    dp_image.dump(residual_word_ann)
    dp_image.image_ann_to_image(residual_word_ann.annotation_id)
    residual_word_ann.dump_sub_category(
        get_type("characters"), ContainerAnnotation(category_name=get_type("characters"), value="residual")
    )

    text_order_service.pass_datapoint(dp_image)

    layout_anns = dp_image.get_annotation(category_names=[get_type("title"), get_type("text")])
    word_anns = dp_image.get_annotation(category_names=get_type("word"))

    sub_cat = layout_anns[0].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = layout_anns[1].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = word_anns[0].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = word_anns[1].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = word_anns[2].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = word_anns[3].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2

    assert get_type("reading_order") not in word_anns[4].sub_categories


def test_integration_pipeline_component_with_include_residual_text_container_to_true(
    dp_image_with_layout_and_word_annotations: Image,
) -> None:
    """
    test integration_pipeline_component_wit_include_residual_text_container_to_true
    """

    text_order_service = TextOrderService(
        text_container=get_type("word"),
        text_block_categories=[
            get_type("title"),
            get_type("text"),
            get_type("cell"),
        ],
        floating_text_block_categories=[get_type("title"), get_type("text")],
    )

    dp_image = dp_image_with_layout_and_word_annotations

    residual_word_ann = ImageAnnotation(
        bounding_box=BoundingBox(ulx=350.0, uly=390.0, lrx=355.0, lry=395.0, absolute_coords=True),
        score=0.6,
        category_name=get_type("word"),
        category_id=8,
    )
    dp_image.dump(residual_word_ann)
    dp_image.image_ann_to_image(residual_word_ann.annotation_id)
    residual_word_ann.dump_sub_category(
        get_type("characters"), ContainerAnnotation(category_name=get_type("characters"), value="residual")
    )

    text_order_service.pass_datapoint(dp_image)

    layout_anns = dp_image.get_annotation(category_names=[get_type("title"), get_type("text"), get_type("line")])
    word_anns = dp_image.get_annotation(category_names=get_type("word"))

    sub_cat = layout_anns[0].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = layout_anns[1].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = layout_anns[2].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 3
    relation = layout_anns[2].get_relationship(get_type("child"))
    assert residual_word_ann.annotation_id in relation

    sub_cat = word_anns[0].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = word_anns[1].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = word_anns[2].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
    sub_cat = word_anns[3].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 2
    sub_cat = word_anns[4].get_sub_category(get_type("reading_order"))
    assert sub_cat.category_id == 1
