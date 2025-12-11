# -*- coding: utf-8 -*-
# File: conftest.py

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


import pytest

from dd_core.utils.object_types import ObjectTypes, object_types_registry, update_black_list
from dd_core.utils.viz import viz_handler


import os
import numpy as np
from copy import deepcopy

import pytest

from dd_core.utils.object_types import get_type

from dd_core.datapoint.box import BoundingBox, local_to_global_coords
from dd_core.datapoint.annotation import ImageAnnotation, CategoryAnnotation, ContainerAnnotation
from dd_core.datapoint.image import Image
from deepdoctection.extern.base import TokenClassResult, DetectionResult

import shared_test_utils as stu



@pytest.fixture
def dp_image() -> Image:
    """fixture Image datapoint"""
    img = Image(location="/test/to/path", file_name="test_name")
    img.image = np.ones([400, 600, 3], dtype=np.float32)
    return img


@pytest.fixture
def layout_annotations():

    def layout_ann(segmentation: bool =False) -> list[ImageAnnotation]:
        if segmentation:
            table_layout_ann = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=100.0, lrx=200.0, lry=400.0, absolute_coords=True),
            score=0.97,
            category_name=get_type("table"),
            category_id=2,
            model_id="test_model",
            service_id="test_service",
        )
    ]
            cell_layout_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=10.0, uly=100.0, lrx=20.0, lry=150.0, absolute_coords=True),
            score=0.8,
            category_name=get_type("cell"),
            category_id=3,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=10.0, uly=200.0, lrx=20.0, lry=250.0, absolute_coords=True),
            score=0.7,
            category_name=get_type("cell"),
            category_id=3,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=40.0, uly=100.0, lrx=50.0, lry=150.0, absolute_coords=True),
            score=0.6,
            category_name=get_type("cell"),
            category_id=3,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=40.0, uly=200.0, lrx=50.0, lry=250.0, absolute_coords=True),
            score=0.5,
            category_name=get_type("cell"),
            category_id=3,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=80.0, uly=260.0, lrx=90.0, lry=280.0, absolute_coords=True),
            score=0.4,
            category_name=get_type("cell"),
            category_id=3,
            model_id="test_model",
            service_id="test_service",
        ),
    ]
            row_layout_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=15.0, uly=100.0, lrx=60.0, lry=150.0, absolute_coords=True),
            score=0.8,
            category_name=get_type("row"),
            category_id=6,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=15.0, uly=200.0, lrx=70.0, lry=240.0, absolute_coords=True),
            score=0.7,
            category_name=get_type("row"),
            category_id=6,
            model_id="test_model",
            service_id="test_service",
        ),
    ]
            col_layout_anns = [
                    ImageAnnotation(
                        bounding_box=BoundingBox(ulx=10.0, uly=50.0, lrx=20.0, lry=250.0, absolute_coords=True),
                        score=0.3,
                        category_name=get_type("column"),
                        category_id=7,
                        model_id="test_model",
                        service_id="test_service",
                    ),
                    ImageAnnotation(
                        bounding_box=BoundingBox(ulx=40.0, uly=20.0, lrx=50.0, lry=240.0, absolute_coords=True),
                        score=0.2,
                        category_name=get_type("column"),
                        category_id=7,
                        model_id="test_model",
                        service_id="test_service",
                    ),
                ]

            return table_layout_ann + cell_layout_anns + row_layout_anns + col_layout_anns

        return [
        ImageAnnotation(
            bounding_box=BoundingBox(absolute_coords=True, ulx=100, uly=160, lrx=200, lry=260),
            score=0.63,
            category_name=get_type("title"),
            category_id=2,
            model_id="test_model",
            service_id="d0b8e9f3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(absolute_coords=True, ulx=50, uly=50, lrx=150, lry=200),
            score=0.97,
            category_name=get_type("table"),
            category_id=4,
            model_id="test_model",
            service_id="d0b8e9f3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(absolute_coords=True, ulx=100, uly=320, lrx=150, lry=350),
            score=0.53,
            category_name=get_type("text"),
            category_id=1,
            model_id="test_model",
            service_id="d0b8e9f3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(absolute_coords=True, ulx=200, uly=50, lrx=250, lry=200),
            score=0.83,
            category_name=get_type("table"),
            category_id=4,
            model_id="test_model",
            service_id="d0b8e9f3",
        ),
    ]

    return layout_ann

@pytest.fixture
def dp_image_tab_cell_item(dp_image: Image, layout_annotations) -> Image:
    """fixture dp_image_tab_cell_item"""
    dp_image = deepcopy(dp_image)
    for ann in layout_annotations(segmentation=True):
        dp_image.dump(ann)

    table = dp_image.get_annotation(category_names=get_type("table"))[0]
    dp_image.image_ann_to_image(table.annotation_id, True)
    table_anns = dp_image.get_annotation(category_names=[get_type("cell"), get_type("row"), get_type("column")])
    for ann in table_anns:
        table.image.dump(ann)
        table.image.image_ann_to_image(ann.annotation_id)
        ann_global_box = local_to_global_coords(
            ann.bounding_box, table.get_bounding_box(dp_image.image_id).transform(image_width=dp_image.width,
                                                                                  image_height=dp_image.height,
                                                                                  absolute_coords=True)  # type: ignore
        )

        ann.image.set_embedding(table.annotation_id, ann.bounding_box.transform(image_width=table.image.width,
                                                                            image_height=table.image.height,
                                                                            absolute_coords=False))  # type: ignore
        ann.image.set_embedding(dp_image.image_id, ann_global_box.transform(image_width=dp_image.width,
                                                                            image_height=dp_image.height,
                                                                            absolute_coords=False))
        table.dump_relationship(get_type("child"), ann.annotation_id)
    return dp_image

@pytest.fixture
def row_sub_cats() -> list[CategoryAnnotation]:
    return [
        CategoryAnnotation(category_name=get_type("row_number"), category_id=1, service_id="dbf4f87c"),
        CategoryAnnotation(category_name=get_type("row_number"), category_id=2, service_id="dbf4f87c"),
    ]


@pytest.fixture
def column_sub_cats() -> list[CategoryAnnotation]:
    return [
        CategoryAnnotation(category_name=get_type("column_number"), category_id=1, service_id="dbf4f87c"),
        CategoryAnnotation(category_name=get_type("column_number"), category_id=2, service_id="dbf4f87c"),
    ]


@pytest.fixture
def dp_image_fully_segmented(
    dp_image_tab_cell_item: Image,
    row_sub_cats: list[CategoryAnnotation],
    column_sub_cats:  list[CategoryAnnotation],
) -> Image:
    """fixture dp_image_fully_segmented"""
    dp = deepcopy(dp_image_tab_cell_item)
    table = dp.get_annotation(category_names=get_type("table"))[0]
    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))
    table_embedding_box = table.get_bounding_box(dp.image_id)
    for row in rows:
        embedding_box = row.get_bounding_box(dp.image_id)
        embedding_box.ulx = table_embedding_box.ulx + 1.0/dp.width
        embedding_box.lrx = table_embedding_box.lrx - 1.0/dp.height

    for col in cols:
        embedding_box = col.get_bounding_box(dp.image_id)
        embedding_box.uly = table_embedding_box.uly + 1.0/dp.width
        embedding_box.lry = table_embedding_box.lry - 1.0/dp.height


    cell_sub_cats = [
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=0, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=0, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=0, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=0, service_id="dbf4f87c"),
        ),
    ]
    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))
    for row, col, row_sub_cat, col_sub_cat in zip(rows, cols, row_sub_cats, column_sub_cats):
        row.dump_sub_category(get_type("row_number"), row_sub_cat)
        col.dump_sub_category(get_type("column_number"), col_sub_cat)

    cells = dp.get_annotation(category_names=[get_type("cell"), get_type("column_header"), get_type("body")])

    for cell, sub_cats in zip(cells, cell_sub_cats):
        cell.dump_sub_category(get_type("row_number"), sub_cats[0])
        cell.dump_sub_category(get_type("column_number"), sub_cats[1])
        cell.dump_sub_category(get_type("row_span"), sub_cats[2])
        cell.dump_sub_category(get_type("column_span"), sub_cats[3])

    return dp

class ObjectTestType(ObjectTypes):
    """Object type members for testing purposes"""

    REPORT_DATE = "report_date"
    UMBRELLA = "umbrella"
    TEST_CAT_1 = "test_cat_1"
    TEST_CAT_2 = "test_cat_2"
    TEST_CAT_3 = "test_cat_3"
    TEST_CAT_4 = "test_cat_4"
    SUB_CAT_1 = "sub_cat_1"
    SUB_CAT_2 = "sub_cat_2"
    SUB_CAT_3 = "sub_cat_3"
    RELATIONSHIP_1 = "relationship_1"
    RELATIONSHIP_2 = "relationship_2"
    NON_EXISTENT = "non_existent"


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("ObjectTestType")(ObjectTestType)
    for item in ObjectTestType:
        update_black_list(item.value)
    viz_handler.refresh()