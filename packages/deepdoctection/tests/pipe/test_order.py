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

"""
Test cases for the TextOrderService component integration.

This module contains various test cases to validate the integration of the
TextOrderService component with different configurations. The tests ensure
that the reading order of text and word annotations is correctly assigned
to the processed datapoint.

"""

from copy import deepcopy

import pytest

from dd_core.datapoint import BoundingBox, ContainerAnnotation, Image, ImageAnnotation, get_type
from deepdoctection.analyzer.config import cfg
from deepdoctection.analyzer.factory import ServiceFactory
from deepdoctection.pipe.order import OrderGenerator, TextOrderService


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


def _two_column_page(dp_image: Image) -> dict[str, ImageAnnotation]:
    """
    Builds a page with two text blocks lying next to each other and a full-width text block below them.

    Each upper block has two words on its first line and one word on its second line. The lower block has one word.
    """
    blocks = {
        "left": BoundingBox(ulx=50.0, uly=50.0, lrx=270.0, lry=150.0, absolute_coords=True),
        "right": BoundingBox(ulx=330.0, uly=50.0, lrx=550.0, lry=150.0, absolute_coords=True),
        "bottom": BoundingBox(ulx=50.0, uly=200.0, lrx=550.0, lry=250.0, absolute_coords=True),
    }
    words = {
        "left_line_1_left": ("left", BoundingBox(ulx=60.0, uly=60.0, lrx=120.0, lry=80.0, absolute_coords=True)),
        "left_line_1_right": ("left", BoundingBox(ulx=140.0, uly=60.0, lrx=200.0, lry=80.0, absolute_coords=True)),
        "left_line_2": ("left", BoundingBox(ulx=60.0, uly=100.0, lrx=120.0, lry=120.0, absolute_coords=True)),
        "right_line_1_left": ("right", BoundingBox(ulx=340.0, uly=60.0, lrx=400.0, lry=80.0, absolute_coords=True)),
        "right_line_1_right": ("right", BoundingBox(ulx=420.0, uly=60.0, lrx=480.0, lry=80.0, absolute_coords=True)),
        "right_line_2": ("right", BoundingBox(ulx=420.0, uly=100.0, lrx=480.0, lry=120.0, absolute_coords=True)),
        "bottom_line_1": ("bottom", BoundingBox(ulx=60.0, uly=210.0, lrx=120.0, lry=230.0, absolute_coords=True)),
    }

    anns: dict[str, ImageAnnotation] = {}
    for name, box in blocks.items():
        anns[name] = ImageAnnotation(bounding_box=box, score=0.9, category_name=get_type("text"), category_id=1)
        dp_image.dump(anns[name])
    for name, (block_name, box) in words.items():
        anns[name] = ImageAnnotation(bounding_box=box, score=0.9, category_name=get_type("word"), category_id=8)
        dp_image.dump(anns[name])
        anns[block_name].dump_relationship(get_type("child"), anns[name].annotation_id)
    return anns


def _reading_order(ann: ImageAnnotation) -> int:
    return ann.get_sub_category(get_type("reading_order")).category_id


def _text_order_service(right_to_left: bool) -> TextOrderService:
    return TextOrderService(
        text_container=get_type("word"),
        text_block_categories=[get_type("text")],
        floating_text_block_categories=[get_type("text")],
        right_to_left=right_to_left,
    )


def test_text_order_service_left_to_right_is_default(dp_image: Image) -> None:
    """
    By default, words in a line and columns next to each other are read from left to right
    """
    anns = _two_column_page(dp_image)

    _text_order_service(right_to_left=False).pass_datapoint(dp_image)

    assert _reading_order(anns["left"]) == 1
    assert _reading_order(anns["right"]) == 2
    assert _reading_order(anns["bottom"]) == 3
    assert _reading_order(anns["left_line_1_left"]) == 1
    assert _reading_order(anns["left_line_1_right"]) == 2
    assert _reading_order(anns["left_line_2"]) == 3
    assert _reading_order(anns["right_line_1_left"]) == 1
    assert _reading_order(anns["right_line_1_right"]) == 2
    assert _reading_order(anns["right_line_2"]) == 3


def test_text_order_service_right_to_left(dp_image: Image) -> None:
    """
    With right_to_left=True, words in a line and columns next to each other are read from right to left, while lines
    and blocks lying over each other are still read from top to bottom
    """
    anns = _two_column_page(dp_image)

    _text_order_service(right_to_left=True).pass_datapoint(dp_image)

    assert _reading_order(anns["right"]) == 1
    assert _reading_order(anns["left"]) == 2
    assert _reading_order(anns["bottom"]) == 3
    assert _reading_order(anns["left_line_1_right"]) == 1
    assert _reading_order(anns["left_line_1_left"]) == 2
    assert _reading_order(anns["left_line_2"]) == 3
    assert _reading_order(anns["right_line_1_right"]) == 1
    assert _reading_order(anns["right_line_1_left"]) == 2
    assert _reading_order(anns["right_line_2"]) == 3
    assert _reading_order(anns["bottom_line_1"]) == 1


def test_text_order_service_right_to_left_orders_residual_lines(dp_image: Image) -> None:
    """
    Words without a parent text block are grouped into synthetic lines whose words are ordered right to left
    """
    words = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=ulx, uly=60.0, lrx=ulx + 40.0, lry=80.0, absolute_coords=True),
            score=0.9,
            category_name=get_type("word"),
            category_id=8,
        )
        for ulx in (100.0, 150.0, 200.0)
    ]
    for word in words:
        dp_image.dump(word)

    _text_order_service(right_to_left=True).pass_datapoint(dp_image)

    assert [_reading_order(word) for word in words] == [3, 2, 1]


def test_text_order_service_clone_keeps_right_to_left() -> None:
    """
    clone keeps the reading direction
    """
    assert _text_order_service(right_to_left=True).clone().order_generator.right_to_left is True
    assert _text_order_service(right_to_left=False).clone().order_generator.right_to_left is False


@pytest.mark.parametrize(
    "left_box, right_box",
    [
        # columns of equal height lying next to each other
        (
            BoundingBox(ulx=30.0, uly=40.0, lrx=270.0, lry=120.0, absolute_coords=True),
            BoundingBox(ulx=330.0, uly=40.0, lrx=570.0, lry=120.0, absolute_coords=True),
        ),
        # columns of different height lying next to each other
        (
            BoundingBox(ulx=30.0, uly=40.0, lrx=270.0, lry=200.0, absolute_coords=True),
            BoundingBox(ulx=330.0, uly=60.0, lrx=570.0, lry=120.0, absolute_coords=True),
        ),
    ],
)
@pytest.mark.parametrize("right_to_left", [False, True])
def test_order_blocks_side_by_side_columns(left_box: BoundingBox, right_box: BoundingBox, right_to_left: bool) -> None:
    """
    Columns lying next to each other are read from left to right or, with right_to_left=True, from right to left
    """
    left = ImageAnnotation(bounding_box=left_box, category_name=get_type("text"), category_id=1, external_id="left")
    right = ImageAnnotation(bounding_box=right_box, category_name=get_type("text"), category_id=1, external_id="right")
    order_generator = OrderGenerator(
        starting_point_tolerance=0.005, broken_line_tolerance=0.003, height_tolerance=2.0, right_to_left=right_to_left
    )

    reading_order = order_generator.order_blocks([left, right], image_width=600, image_height=400)

    expected = [right, left] if right_to_left else [left, right]
    assert reading_order == [(1, expected[0].annotation_id), (2, expected[1].annotation_id)]


def test_build_text_order_service_reads_right_to_left_from_config() -> None:
    """
    TEXT_ORDERING.RIGHT_TO_LEFT is passed to the text order service and defaults to left to right
    """
    assert ServiceFactory.build_text_order_service(cfg).order_generator.right_to_left is False

    config = deepcopy(cfg)
    config.freeze(False)
    config.update_args(["TEXT_ORDERING.RIGHT_TO_LEFT=True"])
    config.freeze()

    assert ServiceFactory.build_text_order_service(config).order_generator.right_to_left is True
    assert cfg.TEXT_ORDERING.RIGHT_TO_LEFT is False
