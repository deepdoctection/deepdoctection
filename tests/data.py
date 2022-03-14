# -*- coding: utf-8 -*-
# File: data.py

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
Dataclasses for generating fixtures in conftest
"""
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from deepdoctection.datapoint import BoundingBox, CategoryAnnotation, ContainerAnnotation, ImageAnnotation
from deepdoctection.extern.base import DetectionResult, TokenClassResult
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.settings import names


@dataclass
class Annotations:  # pylint: disable=R0904
    """
    dataclass Annotations for building fixtures
    """

    layout_detect_results = [
        DetectionResult(box=[100.0, 160.0, 200.0, 260.0], score=0.63, class_id=2, class_name=names.C.TITLE),
        DetectionResult(box=[120.0, 120.0, 140.0, 120.0], score=0.03, class_id=5, class_name=names.C.FIG),
        DetectionResult(box=[50.0, 50.0, 150.0, 200.0], score=0.97, class_id=4, class_name=names.C.TAB),
        DetectionResult(box=[100.0, 320.0, 150.0, 350.0], score=0.53, class_id=1, class_name=names.C.TEXT),
        DetectionResult(box=[200.0, 50.0, 250.0, 200.0], score=0.83, class_id=4, class_name=names.C.TAB),
    ]

    layout_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=160.0, lrx=200.0, lry=260.0, absolute_coords=True),
            score=0.63,
            category_name=names.C.TITLE,
            category_id="2",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=50.0, uly=50.0, lrx=150.0, lry=200.0, absolute_coords=True),
            score=0.97,
            category_name=names.C.TAB,
            category_id="4",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=320.0, lrx=150.0, lry=350.0, absolute_coords=True),
            score=0.53,
            category_name=names.C.TEXT,
            category_id="1",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=200.0, uly=50.0, lrx=250.0, lry=200.0, absolute_coords=True),
            score=0.83,
            category_name=names.C.TAB,
            category_id="4",
        ),
    ]

    layout_ann_for_ordering = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=160.0, lrx=200.0, lry=260.0, absolute_coords=True),
            score=0.9,
            category_name=names.C.TITLE,
            category_id="2",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=300.0, lrx=250.0, lry=350.0, absolute_coords=True),
            score=0.8,
            category_name=names.C.TEXT,
            category_id="1",
        ),
    ]

    cell_detect_results = [
        [
            DetectionResult(box=[20.0, 20.0, 25.0, 30.0], score=0.8, class_id=1, class_name=names.C.HEAD),
            DetectionResult(box=[40.0, 40.0, 50.0, 50.0], score=0.53, class_id=2, class_name=names.C.BODY),
        ],
        [DetectionResult(box=[15.0, 20.0, 20.0, 30.0], score=0.4, class_id=1, class_name=names.C.BODY)],
    ]

    cell_layout_anns = [
        [
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=20.0, uly=20.0, lrx=25.0, lry=30.0, absolute_coords=True),
                category_id="1",
                category_name=names.C.HEAD,
                score=0.8,
            ),
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=40.0, uly=40.0, lrx=50.0, lry=50.0, absolute_coords=True),
                category_id="2",
                category_name=names.C.BODY,
                score=0.53,
            ),
        ],
        [
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=15.0, uly=20.0, lrx=20.0, lry=30.0, absolute_coords=True),
                category_id="1",
                category_name=names.C.BODY,
                score=0.4,
            )
        ],
    ]

    global_cell_boxes = [
        [
            BoundingBox(absolute_coords=True, ulx=70.0, uly=70.0, lrx=75.0, lry=80.0),
            BoundingBox(absolute_coords=True, ulx=90.0, uly=90.0, lrx=100.0, lry=100.0),
        ],
        [BoundingBox(absolute_coords=True, ulx=215.0, uly=70.0, lrx=220.0, lry=80.0)],
    ]

    table_layout_ann = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=100.0, lrx=200.0, lry=400.0, absolute_coords=True),
            score=0.97,
            category_name=names.C.TAB,
            category_id="2",
        )
    ]

    cell_layout_anns_one_table = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=10.0, uly=100.0, lrx=20.0, lry=150.0, absolute_coords=True),
            score=0.8,
            category_name=names.C.CELL,
            category_id="3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=10.0, uly=200.0, lrx=20.0, lry=250.0, absolute_coords=True),
            score=0.7,
            category_name=names.C.CELL,
            category_id="3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=40.0, uly=100.0, lrx=50.0, lry=150.0, absolute_coords=True),
            score=0.6,
            category_name=names.C.CELL,
            category_id="3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=40.0, uly=200.0, lrx=50.0, lry=250.0, absolute_coords=True),
            score=0.5,
            category_name=names.C.CELL,
            category_id="3",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=80.0, uly=260.0, lrx=90.0, lry=280.0, absolute_coords=True),
            score=0.4,
            category_name=names.C.CELL,
            category_id="3",
        ),
    ]

    row_layout_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=15.0, uly=100.0, lrx=60.0, lry=150.0, absolute_coords=True),
            score=0.8,
            category_name=names.C.ROW,
            category_id="6",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=15.0, uly=200.0, lrx=70.0, lry=240.0, absolute_coords=True),
            score=0.7,
            category_name=names.C.ROW,
            category_id="6",
        ),
    ]

    row_box_tiling_table = [
        BoundingBox(ulx=101.0, uly=101.0, lrx=199.0, lry=250.0, absolute_coords=True),
        BoundingBox(ulx=101.0, uly=250.0, lrx=199.0, lry=399.0, absolute_coords=True),
    ]

    col_layout_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=10.0, uly=50.0, lrx=20.0, lry=250.0, absolute_coords=True),
            score=0.3,
            category_name=names.C.COL,
            category_id="7",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=40.0, uly=20.0, lrx=50.0, lry=240.0, absolute_coords=True),
            score=0.2,
            category_name=names.C.COL,
            category_id="7",
        ),
    ]

    col_box_tiling_table = [
        BoundingBox(ulx=101.0, uly=101.0, lrx=120.0, lry=399.0, absolute_coords=True),
        BoundingBox(ulx=120.0, uly=101.0, lrx=199.0, lry=399.0, absolute_coords=True),
    ]

    row_sub_cats = [
        CategoryAnnotation(category_name=names.C.RN, category_id=1),
        CategoryAnnotation(category_name=names.C.RN, category_id=2),
    ]

    col_sub_cats = [
        CategoryAnnotation(category_name=names.C.CN, category_id=1),
        CategoryAnnotation(category_name=names.C.CN, category_id=2),
    ]

    cell_sub_cats = [
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=1),
            CategoryAnnotation(category_name=names.C.CN, category_id=1),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=2),
            CategoryAnnotation(category_name=names.C.CN, category_id=1),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=1),
            CategoryAnnotation(category_name=names.C.CN, category_id=2),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=2),
            CategoryAnnotation(category_name=names.C.CN, category_id=2),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=0),
            CategoryAnnotation(category_name=names.C.CN, category_id=0),
            CategoryAnnotation(category_name=names.C.RS, category_id=0),
            CategoryAnnotation(category_name=names.C.CS, category_id=0),
        ),
    ]

    cell_sub_cats_when_table_fully_tiled = [
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=1),
            CategoryAnnotation(category_name=names.C.CN, category_id=1),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=2),
            CategoryAnnotation(category_name=names.C.CN, category_id=1),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=1),
            CategoryAnnotation(category_name=names.C.CN, category_id=2),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=2),
            CategoryAnnotation(category_name=names.C.CN, category_id=2),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
        (
            CategoryAnnotation(category_name=names.C.RN, category_id=2),
            CategoryAnnotation(category_name=names.C.CN, category_id=2),
            CategoryAnnotation(category_name=names.C.RS, category_id=1),
            CategoryAnnotation(category_name=names.C.CS, category_id=1),
        ),
    ]

    summary_sub_cat_when_table_fully_tiled = (
        CategoryAnnotation(category_name=names.C.NR, category_id=2),
        CategoryAnnotation(category_name=names.C.NC, category_id=2),
        CategoryAnnotation(category_name=names.C.NRS, category_id=1),
        CategoryAnnotation(category_name=names.C.NCS, category_id=1),
    )

    summary_htab_sub_cat = ContainerAnnotation(
        category_name=names.C.HTAB, value="<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>"
    )

    word_results_list = [
        DetectionResult(
            box=[10.0, 10.0, 24.0, 23.0],
            score=0.8,
            text="foo",
            block="1",
            line="2",
            class_id=1,
            class_name=names.C.WORD,
        ),
        DetectionResult(
            box=[30.0, 20.0, 38.0, 25.0],
            score=0.2,
            text="bak",
            block="4",
            line="5",
            class_id=1,
            class_name=names.C.WORD,
        ),
    ]

    word_layout_ann = [
        ImageAnnotation(
            category_name=names.C.WORD,
            bounding_box=BoundingBox(absolute_coords=True, ulx=10.0, uly=10.0, width=14.0, height=13.0),
            score=0.8,
            category_id="1",
        ),
        ImageAnnotation(
            category_name=names.C.WORD,
            bounding_box=BoundingBox(absolute_coords=True, ulx=30.0, uly=20.0, width=8.0, height=5.0),
            score=0.2,
            category_id="1",
        ),
    ]

    word_layout_ann_for_matching = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=112.0, uly=210.0, lrx=119.0, lry=220.0, absolute_coords=True),
            score=0.9,
            category_name=names.C.WORD,
            category_id="8",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=145.0, uly=210.0, lrx=155.0, lry=220.0, absolute_coords=True),
            score=0.95,
            category_name=names.C.WORD,
            category_id="8",
        ),
    ]

    word_layout_for_ordering = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=110.0, uly=165.0, lrx=130.0, lry=180.0, absolute_coords=True),
            score=0.9,
            category_name=names.C.WORD,
            category_id="8",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=140.0, uly=162.0, lrx=180.0, lry=180.0, absolute_coords=True),
            score=0.8,
            category_name=names.C.WORD,
            category_id="8",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=320.0, lrx=130.0, lry=340.0, absolute_coords=True),
            score=0.7,
            category_name=names.C.WORD,
            category_id="8",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=175.0, uly=320.0, lrx=205.0, lry=340.0, absolute_coords=True),
            score=0.6,
            category_name=names.C.WORD,
            category_id="8",
        ),
    ]

    word_sub_cats = [
        [
            ContainerAnnotation(category_name=names.C.CHARS, value="foo"),
            CategoryAnnotation(category_name=names.C.BLOCK, category_id="1"),
            CategoryAnnotation(category_name=names.C.TLINE, category_id="2"),
        ],
        [
            ContainerAnnotation(category_name=names.C.CHARS, value="bak"),
            CategoryAnnotation(category_name=names.C.BLOCK, category_id="4"),
            CategoryAnnotation(category_name=names.C.TLINE, category_id="5"),
        ],
    ]

    word_sub_cats_for_ordering = [
        [
            ContainerAnnotation(category_name=names.C.CHARS, value="hello"),
            CategoryAnnotation(category_name=names.C.BLOCK, category_id="1"),
            CategoryAnnotation(category_name=names.C.TLINE, category_id="1"),
        ],
        [
            ContainerAnnotation(category_name=names.C.CHARS, value="world"),
            CategoryAnnotation(category_name=names.C.BLOCK, category_id="1"),
            CategoryAnnotation(category_name=names.C.TLINE, category_id="2"),
        ],
        [
            ContainerAnnotation(category_name=names.C.CHARS, value="bye"),
            CategoryAnnotation(category_name=names.C.BLOCK, category_id="2"),
            CategoryAnnotation(category_name=names.C.TLINE, category_id="2"),
        ],
        [
            ContainerAnnotation(category_name=names.C.CHARS, value="world"),
            CategoryAnnotation(category_name=names.C.BLOCK, category_id="2"),
            CategoryAnnotation(category_name=names.C.TLINE, category_id="2"),
        ],
    ]

    word_box_global = [
        BoundingBox(absolute_coords=True, ulx=60.0, uly=60.0, width=14.0, height=13.0),
        BoundingBox(absolute_coords=True, ulx=80.0, uly=70.0, width=8.0, height=5.0),
        BoundingBox(absolute_coords=True, ulx=210.0, uly=60.0, width=14.0, height=13.0),
        BoundingBox(absolute_coords=True, ulx=230.0, uly=70.0, width=8.0, height=5.0),
    ]  # global coordinates calculated depending on table annotation from fixture image_annotations

    def __post_init__(self) -> None:
        self.layout_anns[0]._annotation_id = "a6e3e759-2ad8-3767-9855-bfe611d77073"  # pylint: disable=W0212
        self.layout_anns[1]._annotation_id = "06bf6a32-2cc8-3fdb-b742-d22381f135d1"  # pylint: disable=W0212
        self.layout_anns[2]._annotation_id = "982a2a72-c9af-39bd-a40e-bde5abb37927"  # pylint: disable=W0212
        self.layout_anns[3]._annotation_id = "07f8c1f9-a67a-32bf-a253-a394220cbac6"  # pylint: disable=W0212

    def get_layout_detect_results(self) -> List[DetectionResult]:
        """
        layout_detect_results
        """
        return self.layout_detect_results

    def get_layout_annotation(self, segmentation: bool = False) -> List[ImageAnnotation]:
        """
        layout_annotations
        """
        if segmentation:
            all_anns = []
            all_anns.extend(self.table_layout_ann)
            all_anns.extend(self.cell_layout_anns_one_table)
            all_anns.extend(self.row_layout_anns)
            all_anns.extend(self.col_layout_anns)
            return all_anns
        return self.layout_anns

    def get_layout_ann_for_ordering(self) -> List[ImageAnnotation]:
        """
        layout_ann_for_ordering
        """
        return self.layout_ann_for_ordering

    def get_cell_detect_results(self) -> List[List[DetectionResult]]:
        """
        cell_detect_results
        """
        return self.cell_detect_results

    def get_cell_annotations(self) -> List[List[ImageAnnotation]]:
        """
        cell_annotations
        """
        return self.cell_layout_anns

    def get_global_cell_boxes(self) -> List[List[BoundingBox]]:
        """
        global_cell_boxes
        """
        return self.global_cell_boxes

    def get_row_sub_cats(self) -> List[CategoryAnnotation]:
        """
        row_sub_cats
        """
        return self.row_sub_cats

    def get_col_sub_cats(self) -> List[CategoryAnnotation]:
        """
        col_sub_cats
        """
        return self.col_sub_cats

    def get_cell_sub_cats(
        self,
    ) -> List[Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]]:
        """
        cell_sub_cats
        """
        return self.cell_sub_cats

    def get_row_box_tiling_table(self) -> List[BoundingBox]:
        """
        row_box_tiling_table
        """
        return self.row_box_tiling_table

    def get_col_box_tiling_table(self) -> List[BoundingBox]:
        """
        col_box_tiling_table
        """
        return self.col_box_tiling_table

    def get_cell_sub_cats_when_table_fully_tiled(
        self,
    ) -> List[Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]]:
        """
        cell_sub_cats_when_table_fully_tiled
        """
        return self.cell_sub_cats_when_table_fully_tiled

    def get_summary_sub_cats_when_table_fully_tiled(
        self,
    ) -> Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]:
        """
        summary_sub_cats_when_table_fully_tiled
        """
        return self.summary_sub_cat_when_table_fully_tiled

    def get_summary_htab_sub_cat(self) -> ContainerAnnotation:
        """
        summary_htab_sub_cat
        """
        return self.summary_htab_sub_cat

    def get_word_detect_results(self) -> List[DetectionResult]:
        """
        word_detect_results
        """
        return self.word_results_list

    def get_double_word_detect_results(self) -> List[List[DetectionResult]]:
        """list of list of word results. The inner lists are identical"""
        return [self.word_results_list, self.word_results_list]

    def get_word_layout_ann(self) -> List[ImageAnnotation]:
        """
        word_layout_ann
        """
        return self.word_layout_ann

    def get_word_layout_ann_for_matching(self) -> List[ImageAnnotation]:
        """
        word_layout_ann_for_matching
        """
        return self.word_layout_ann_for_matching

    def get_word_layout_annotations_for_ordering(self) -> List[ImageAnnotation]:
        """
        word_layout_annotations_for_ordering
        """
        return self.word_layout_for_ordering

    def get_word_sub_cats(self) -> List[List[Union[ContainerAnnotation, CategoryAnnotation, CategoryAnnotation]]]:
        """
        word_sub_cats
        """
        return self.word_sub_cats

    def get_word_sub_cats_for_ordering(self) -> List[List[CategoryAnnotation]]:
        """
        word_sub_cats_for_ordering
        """
        return self.word_sub_cats_for_ordering

    def get_word_box_global(self) -> List[BoundingBox]:
        """
        word_box_global
        """
        return self.word_box_global


_SAMPLE_TEXTRACT = {
    "DocumentMetadata": {"Pages": 1},
    "Blocks": [
        {
            "BlockType": "PAGE",
            "Geometry": {
                "BoundingBox": {"Width": 1.0, "Height": 1.0, "Left": 0.0, "Top": 0.0},
                "Polygon": [
                    {"X": 1.5848370254932184e-16, "Y": 0.0},
                    {"X": 1.0, "Y": 9.463179803411622e-17},
                    {"X": 1.0, "Y": 1.0},
                    {"X": 0.0, "Y": 1.0},
                ],
            },
            "Id": "cc95c471-ee7f-4b22-bb2a-b5d5bd590e35",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "e4963e9b-97f5-4383-99a5-8cb93dc044ed",
                        "f70070e0-bc0b-4562-b624-0ede0f399c08",
                        "af4bd19f-5d3e-46ed-98b6-e95e41a080cd",
                        "9df29731-35d1-4651-8c58-675b5b62fe43",
                        "e2f7ae15-0350-46d3-bea1-87702e9820d4",
                        "42dcf959-4477-4a61-abc5-4f5677036e9c",
                        "12ea8d4f-089f-42d1-adf9-c16a1f53cf17",
                        "106c234a-bff2-45a2-8a7e-7adf0dbb00c3",
                        "467f8cec-661d-40d7-bdf4-4307eeeef7a3",
                        "116dd305-c168-4697-8fa4-2a4ad93332dc",
                        "38c504ec-51de-46c1-81f0-7c079991b03b",
                        "3b71267e-9dca-4584-8ca0-bb23e733c756",
                        "c6b8f01d-ffc6-4b12-8d80-9981a5552179",
                        "5b7d593c-6dec-4f3b-a8de-7c8ac7c48838",
                        "cc0ac51d-49f7-4c95-b00a-6e5013d37563",
                        "99480661-b153-45c2-bf31-34e4239215c7",
                        "5bc26570-4f78-4324-b93f-c3c83d224b91",
                        "b3448271-fc97-4a36-acee-85015f744c24",
                        "a1c546c2-e46d-4436-b36a-3c9b1eac2abf",
                        "ebea875d-524d-4402-982f-333e5e043a09",
                        "88cfb3ef-9b2a-4d96-b264-bc43c4142138",
                        "2664b0e6-3a05-479f-9c2c-eb06390b1f4d",
                        "29e92472-20bf-4ddb-b586-83d8754a3d9f",
                        "454b6467-cdef-49bf-9783-cd932140ed0d",
                        "3652465f-2feb-45f0-9f31-45741f8f8f7d",
                        "7a893676-a4f0-4377-a1e6-894e692bc3a3",
                        "350bd81b-4df2-45b4-a0a2-94176f09d1de",
                        "65b1da32-d752-46d7-8519-d7ebb019c66e",
                        "3e7bfda2-2996-4f32-95ef-e6a04488e212",
                        "ae0c808b-ee5d-4449-aa9d-b03e6d13b14c",
                        "410de6e4-d253-428e-841b-8feff9ae7ad3",
                        "4fd10a64-58dc-487f-b661-15055c105ec6",
                        "91eace35-6434-471b-9e77-7da7d09a1e44",
                        "d35b8be8-3abf-470a-b178-b47fc1e67e32",
                        "dc52e827-b241-4869-9610-9284b2a22171",
                        "5f38e5d7-765a-435d-a0e0-8b684b08533f",
                        "388154c9-2874-4f46-8c97-c2815457ff83",
                        "8568d129-1a9d-4471-98ea-5e04987e44b6",
                        "2bbed55b-4d18-49af-a2df-637cada5f816",
                        "ef02e0e2-d9e8-4b9c-ae39-f1d5625f02ef",
                        "e3d81f9d-dfa8-4396-b8da-01ccd57cabfa",
                        "3bac1e7b-57b5-429e-9019-1e4dc086ba79",
                        "b2578933-bf36-4aa4-8ca2-c8b988c41f48",
                        "3ba85aef-0ea9-46d2-94fa-6a3d829c0830",
                        "da194351-b0ac-4596-8c4e-81d505c8f72d",
                        "d1ca40e3-2fa1-4f2c-bb27-98d717152cdb",
                        "969f6c43-1758-4cb1-9f5b-63627042881a",
                        "285077de-3348-4a84-a25c-1afcb32a177b",
                        "e68350cc-f5d6-490d-85cc-c49fa3b8d898",
                        "620bf038-c482-4753-ad85-b3ab8e7c0b31",
                        "7eec4e66-ab04-48df-9398-f6bc2003fee2",
                        "535bb673-b0d1-4f2f-9ed7-9e6829298a1f",
                        "91cccd15-7687-4bd8-9ee4-7156256f0868",
                        "022ff055-7658-4271-a417-820044003a25",
                        "df42bc0b-b85a-4499-9086-1d7b7500ee47",
                        "368009c6-745c-4030-a8d7-b69e6a8b3b24",
                        "fd738280-375b-4cb0-8006-7242e075ec0b",
                        "5b6ec3e9-6444-425e-a6ab-dd06222d2bec",
                        "6930e9a2-1f25-4b00-82af-6c5bf64a74e2",
                        "78c92a3f-8110-4021-b751-b0369d29ed03",
                        "b3829ffc-4748-486e-9618-88b0c04cd948",
                        "a982ee1e-2738-4601-91f5-2a81c7b8d939",
                        "2260c739-6097-4051-92f6-cc8ad161a3fd",
                        "afdc558e-7edb-4e3a-aab1-905e63dc557b",
                        "9b8429ae-27ea-4ccc-a30f-7a30016e3c27",
                        "7d5d321d-9b59-417b-bc92-fd3353f06b8e",
                        "bb67ce1e-9d26-4e6c-bec6-5969d6070198",
                        "84405e45-e781-4bfa-b7dc-7fd87cf7d5d7",
                        "0b5c97a7-ab51-4572-9113-0aea272f9d91",
                        "e2c87a2d-c820-449b-85c7-9688e4c2746a",
                        "c9b52748-4bb0-4c80-83cd-65e6e9031786",
                        "f9a82720-4276-45c1-a6ab-29fb0243f7da",
                        "4b9ccb37-64ba-4967-9d7d-7b5514af3a8f",
                        "579ce934-859d-4392-9df3-6bd5254766a7",
                        "fc0af094-d103-4466-aaa7-148d5f4ed784",
                        "e2850f43-6b56-41fc-8b1d-762307514399",
                        "42d90809-d8d3-4d7d-b7f0-030171497a57",
                        "1215bd63-d5ae-462f-a457-b0d1fcb2bd43",
                        "3d6fd650-0ed0-4965-a406-7b11eb4de214",
                        "38d73491-8c2e-4670-960f-ffe407af651c",
                        "ced7b2a1-e28f-45ce-b11e-b37b6f7ed25d",
                        "2e002c46-68ef-4940-8baf-7a49e0d6c51e",
                        "990fc5e4-777a-4aa8-bb6e-9c8ac6c3b768",
                        "7d548462-4535-464a-bd42-7727f554285a",
                        "b8716c4d-b2f6-43c1-bddb-e44a1960ef71",
                        "ab018ba7-ef8f-4621-bd18-fdfd5ba000e0",
                        "d179fa85-8cb0-4313-851d-5449c96d988b",
                        "61ea4043-0642-4914-85f5-64d9ee954e77",
                        "23d717c2-eb61-4cd9-845a-f8b65382d5d3",
                        "5504d5a1-7a24-4e93-a023-ac496f15b710",
                        "0a2016cf-908e-41fc-8a47-1bcd15be4c59",
                        "5779dfc0-1c64-4c95-aa0a-86c59ceac4fc",
                        "6eea40be-e8de-46d1-b34b-6c266c1de9c1",
                        "d22460dd-6c17-4977-9dc8-30e90a70ffdc",
                        "8ff31f45-eba3-441f-8c44-0ee583c4f11a",
                        "71f91a49-5749-42cc-8560-7e25a506a2cc",
                        "18815038-2f5b-42c4-a084-231520d93b06",
                        "7c4fcb91-8ca0-4dd7-9911-e15e89418c24",
                        "e54a0c7a-eb8a-46de-8209-4f5a00ef00e7",
                        "6287c55c-56eb-4db7-b8c7-b59f85970e52",
                        "38d9fc21-6510-4aaf-a1e0-0896235be7bd",
                        "bb040e9b-5a2b-4902-ac07-7d4846412f57",
                        "24cc72a5-edb5-488e-98f0-a17780479aa0",
                        "075dc4fe-bb21-43dd-9077-350a33bc609e",
                        "484828bd-26fd-4444-bd87-614423e20317",
                        "e59fd3f8-e1b3-428d-9756-8dfa2f57e57b",
                        "7eb987eb-1781-4ac8-a2f7-8db40bf82237",
                        "f18669b4-211a-412a-8393-ffefb8a9c313",
                        "805f9aac-e6b9-408f-954d-b5b30e058011",
                        "307de933-9e4c-4b4b-9c44-c123bad84b83",
                        "74e3d491-b920-4a23-8159-74087cd8ea61",
                        "19434308-8701-40ed-9ee1-753c4d0ca0a1",
                        "32176a68-dde3-42f9-85c0-93ebe77d8e58",
                        "8d93a45f-e007-4657-bf7c-1e506c9d1389",
                        "ebfab80c-e007-438b-8f62-82e8d6f12014",
                        "8aed06af-4d82-4e12-bd96-598c466cf32b",
                        "82a16924-0a92-4922-b434-2ed13b167eba",
                        "9eba4393-2199-4acd-bf0b-c2515c7c6c16",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 74.47697448730469,
            "Text": "MOTES to CONSOLIDATED FINANCIAL",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.15513421595096588,
                    "Height": 0.00902241189032793,
                    "Left": 0.7530537247657776,
                    "Top": 0.02961716055870056,
                },
                "Polygon": [
                    {"X": 0.7530537247657776, "Y": 0.02961716055870056},
                    {"X": 0.908187985420227, "Y": 0.02961716055870056},
                    {"X": 0.908187985420227, "Y": 0.03863957151770592},
                    {"X": 0.7530537247657776, "Y": 0.03863957151770592},
                ],
            },
            "Id": "e4963e9b-97f5-4383-99a5-8cb93dc044ed",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "91663486-5b2e-48db-80ac-c0de1f1ecf37",
                        "cf234ec9-52cf-4710-94ce-288f0e055091",
                        "ba7d7a8e-5ba6-4844-8ca2-667e3fc2196e",
                        "b31c9d09-d226-4fe8-8a02-bf9ee134cef4",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 87.73054504394531,
            "Text": "STATEMENTS",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.056317396461963654,
                    "Height": 0.00884819496423006,
                    "Left": 0.7532274127006531,
                    "Top": 0.03825380280613899,
                },
                "Polygon": [
                    {"X": 0.7532274127006531, "Y": 0.03825380280613899},
                    {"X": 0.8095448017120361, "Y": 0.03825380280613899},
                    {"X": 0.8095448017120361, "Y": 0.04710199683904648},
                    {"X": 0.7532274127006531, "Y": 0.04710199683904648},
                ],
            },
            "Id": "f70070e0-bc0b-4562-b624-0ede0f399c08",
            "Relationships": [{"Type": "CHILD", "Ids": ["1338f0a5-2904-40db-bbd5-5c1653f9eba5"]}],
        },
        {
            "BlockType": "LINE",
            "Confidence": 93.70662689208984,
            "Text": "65",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.013070599175989628,
                    "Height": 0.009730551391839981,
                    "Left": 0.9296612739562988,
                    "Top": 0.06673943251371384,
                },
                "Polygon": [
                    {"X": 0.9296612739562988, "Y": 0.06673943251371384},
                    {"X": 0.9427318572998047, "Y": 0.06673943251371384},
                    {"X": 0.9427318572998047, "Y": 0.07646998763084412},
                    {"X": 0.9296612739562988, "Y": 0.07646998763084412},
                ],
            },
            "Id": "af4bd19f-5d3e-46ed-98b6-e95e41a080cd",
            "Relationships": [{"Type": "CHILD", "Ids": ["9c14566a-98ab-4f90-80bb-2929a3c8bb2a"]}],
        },
        {
            "BlockType": "LINE",
            "Confidence": 98.82858276367188,
            "Text": "A summary of future minimum lease payments under capital",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.404915452003479,
                    "Height": 0.014473780058324337,
                    "Left": 0.10605736076831818,
                    "Top": 0.1575692743062973,
                },
                "Polygon": [
                    {"X": 0.10605736076831818, "Y": 0.1575692743062973},
                    {"X": 0.510972797870636, "Y": 0.1575692743062973},
                    {"X": 0.510972797870636, "Y": 0.1720430552959442},
                    {"X": 0.10605736076831818, "Y": 0.1720430552959442},
                ],
            },
            "Id": "9df29731-35d1-4651-8c58-675b5b62fe43",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "8c6e33a5-fcc0-4e74-babd-4f7dd25d373c",
                        "6b327141-872f-46cd-a7d7-0ed9a4207efe",
                        "e0f96344-e11e-4312-aa29-efad7919be38",
                        "97b75108-1005-4c45-90fe-e5daf375f209",
                        "ed2e0aa8-6bb4-4689-aea7-c24af8c2a127",
                        "b82a2af4-e570-49cb-aee8-cf37e1aa21cb",
                        "8d677ac3-7f65-4d47-831e-1d316acc06a6",
                        "14a24899-d3ca-4379-acee-356b3a2d566d",
                        "28d5884f-7829-451f-a719-cfa3610e9dfb",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 99.65743255615234,
            "Text": "Option-vesting periods range from one to four years with more",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.40415629744529724,
                    "Height": 0.014144454151391983,
                    "Left": 0.538150429725647,
                    "Top": 0.15785999596118927,
                },
                "Polygon": [
                    {"X": 0.538150429725647, "Y": 0.15785999596118927},
                    {"X": 0.9423066973686218, "Y": 0.15785999596118927},
                    {"X": 0.9423066973686218, "Y": 0.17200444638729095},
                    {"X": 0.538150429725647, "Y": 0.17200444638729095},
                ],
            },
            "Id": "e2f7ae15-0350-46d3-bea1-87702e9820d4",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "f0e21005-bfc2-468d-80c3-26da6b44c02f",
                        "cf2e7627-2518-4f34-b341-f0a79a01dd4a",
                        "aa9c47e9-3fd7-4d8e-a168-1fbbee3413d8",
                        "4b77262c-8651-4897-9366-492831702400",
                        "ba90f86e-dfc9-4064-a063-d4a8ac46b243",
                        "03d98d2c-b12f-4797-8b4a-71dd0ca00a4d",
                        "f7055f7e-f12b-4ff9-8a55-ce8fa64a4046",
                        "a8a19a00-a345-45a1-865c-d21dd2f3f2cd",
                        "e2a55a73-8443-421e-b3a2-71605ba125a4",
                        "2d8bdea8-a6fc-45cf-8a03-f92d4b257647",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 99.59542846679688,
            "Text": "leases and noncancelable operating leases (principally aircraft",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.4043440520763397,
                    "Height": 0.014157670550048351,
                    "Left": 0.10662936419248581,
                    "Top": 0.1730063408613205,
                },
                "Polygon": [
                    {"X": 0.10662936419248581, "Y": 0.1730063408613205},
                    {"X": 0.5109733939170837, "Y": 0.1730063408613205},
                    {"X": 0.5109733939170837, "Y": 0.18716402351856232},
                    {"X": 0.10662936419248581, "Y": 0.18716402351856232},
                ],
            },
            "Id": "42dcf959-4477-4a61-abc5-4f5677036e9c",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "d5be6d05-a7ad-4a61-9908-14d6588c2cf7",
                        "62edca2c-1ddb-4a43-8fd1-06d18fa37df8",
                        "fe358c6f-4850-472d-8dd0-a32511566e6c",
                        "d9e80c1d-a78a-400a-8c4d-c636d73744ab",
                        "94f21e73-6dd8-4ce5-94bb-8e42e496b8b4",
                        "02ac1f4e-5470-42e9-9921-644a5ec9f8c3",
                        "27a049b3-ae4f-47c1-ade9-f929665ee052",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 99.25838470458984,
            "Text": "than 80% of stock option grants vesting ratably over 4 years. At",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.40484702587127686,
                    "Height": 0.014191791415214539,
                    "Left": 0.5379953384399414,
                    "Top": 0.17317181825637817,
                },
                "Polygon": [
                    {"X": 0.5379953384399414, "Y": 0.17317181825637817},
                    {"X": 0.9428423643112183, "Y": 0.17317181825637817},
                    {"X": 0.9428423643112183, "Y": 0.1873636096715927},
                    {"X": 0.5379953384399414, "Y": 0.1873636096715927},
                ],
            },
            "Id": "12ea8d4f-089f-42d1-adf9-c16a1f53cf17",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "5f9e6444-65ab-41af-9978-47539c139ad4",
                        "d6285f6b-8350-4cc3-a5e3-aa3786f00f5a",
                        "d3296eb1-261b-43d7-950d-83d91228c214",
                        "415abdc9-d598-4ceb-a588-6e70fa1d4388",
                        "0e53764d-66d3-4d86-bc30-f6f8d01841fd",
                        "89b0619c-ddf0-4164-99f9-e10d4f1fab31",
                        "08a569ad-38b5-45e4-8863-69f6ad00ffe9",
                        "516b1383-f70f-4ca5-940e-4070e7fb60fb",
                        "a42e8345-e5ae-4a25-a24c-70198d19a863",
                        "10d2fed0-23e9-40f8-a7be-2f071ccc1435",
                        "adf4b84b-a851-4f53-bd3e-b7c84b471d30",
                        "9258c235-da88-44fe-89bc-252f68909087",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 99.80220031738281,
            "Text": "shares of common stock at a price not less than its fair market",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.4045187830924988,
                    "Height": 0.013797413557767868,
                    "Left": 0.10698878020048141,
                    "Top": 0.8323672413825989,
                },
                "Polygon": [
                    {"X": 0.10698878020048141, "Y": 0.8323672413825989},
                    {"X": 0.5115075707435608, "Y": 0.8323672413825989},
                    {"X": 0.5115075707435608, "Y": 0.8461646437644958},
                    {"X": 0.10698878020048141, "Y": 0.8461646437644958},
                ],
            },
            "Id": "32176a68-dde3-42f9-85c0-93ebe77d8e58",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "56e56c55-4dca-4f0f-a4db-509c6fd07f99",
                        "addced16-a021-49a5-ac15-7edc1ad4745d",
                        "102a22dd-ee53-4c30-95a5-6e7986d00be9",
                        "ee93b38b-344b-4193-86f2-a3c8149056df",
                        "5972a02d-f2a5-400d-9f7a-d71222a21d6f",
                        "9e5fe241-8fa4-4445-82ab-8548157b780b",
                        "71295a90-8fb2-45e2-addc-c8ea9c0357f9",
                        "eac4abac-4f1f-49aa-81cf-de45d0fe7257",
                        "fd16f503-1769-44c7-bdae-766a8228da66",
                        "f5859129-6577-4b47-b928-b22ae51f37dd",
                        "d9bff45b-7b1b-4139-bc2c-80780dd7107e",
                        "1b0cd98f-74e5-485c-8d98-1fa24e8da2b0",
                        "5cb613eb-81ae-4bc4-9cdf-c83b89994e36",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 98.70418548583984,
            "Text": "cal experience and will lower pro forma compensation",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.40376970171928406,
                    "Height": 0.013870161958038807,
                    "Left": 0.5385613441467285,
                    "Top": 0.8372853398323059,
                },
                "Polygon": [
                    {"X": 0.5385613441467285, "Y": 0.8372853398323059},
                    {"X": 0.942331075668335, "Y": 0.8372853398323059},
                    {"X": 0.942331075668335, "Y": 0.8511555194854736},
                    {"X": 0.5385613441467285, "Y": 0.8511555194854736},
                ],
            },
            "Id": "8d93a45f-e007-4657-bf7c-1e506c9d1389",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "37221973-c166-491b-8428-b09c718b7f6e",
                        "44655d81-4360-47e0-be7e-e449f258a832",
                        "e1ade23e-be78-4cd4-a800-628a9ab8d46f",
                        "d9b3f84f-00ca-4218-a257-c714d8e7c455",
                        "77265d66-f0ea-462f-9467-dc0858052d8a",
                        "78c634f5-6a3b-475e-9d83-2def30524569",
                        "42f41c9b-c819-4c3d-9bf6-74eb3d0ce962",
                        "c2444160-1aba-4c7b-8c6c-abed6ba7db4a",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 99.01702117919922,
            "Text": "value at the date of grant. Options granted have a maximum term",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.40413445234298706,
                    "Height": 0.014036313630640507,
                    "Left": 0.10650232434272766,
                    "Top": 0.8473659753799438,
                },
                "Polygon": [
                    {"X": 0.10650232434272766, "Y": 0.8473659753799438},
                    {"X": 0.5106367468833923, "Y": 0.8473659753799438},
                    {"X": 0.5106367468833923, "Y": 0.8614022731781006},
                    {"X": 0.10650232434272766, "Y": 0.8614022731781006},
                ],
            },
            "Id": "ebfab80c-e007-438b-8f62-82e8d6f12014",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "f6f76f25-d975-4a8f-98df-d512c915e39c",
                        "5ce61ccc-a897-4510-b6d3-31a8e641bff1",
                        "c75d22a8-a009-48b0-96fb-2b58d1fb3ace",
                        "910becd9-0ab2-4c20-8040-8ab97816d7d5",
                        "1fd355cf-06a3-4fdd-af6c-25199c8a099e",
                        "3d2c5309-8eb1-41a7-b398-0e780a85c613",
                        "0dfb5d27-c54a-4b57-aa3d-ec7dfc1050c5",
                        "3d358ec3-7977-432e-82eb-3fc6ba92bb2a",
                        "447bd097-74f7-4d72-8f09-5b28343108b4",
                        "18737926-c016-4403-b253-82c06a1c7519",
                        "55979cbb-b128-4acb-851e-dddb78060537",
                        "f0ea2bbf-07cb-4665-8870-668834014368",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 92.21876525878906,
            "Text": "expense. Our forfeiture rate is approximately 8%",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.3170889914035797,
                    "Height": 0.01387424673885107,
                    "Left": 0.5379155874252319,
                    "Top": 0.8525774478912354,
                },
                "Polygon": [
                    {"X": 0.5379155874252319, "Y": 0.8525774478912354},
                    {"X": 0.855004608631134, "Y": 0.8525774478912354},
                    {"X": 0.855004608631134, "Y": 0.8664516806602478},
                    {"X": 0.5379155874252319, "Y": 0.8664516806602478},
                ],
            },
            "Id": "8aed06af-4d82-4e12-bd96-598c466cf32b",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "e118af8f-a7df-49db-b7ad-6279d1784a97",
                        "3a9a11d6-8e83-4725-9e16-ed74eff24681",
                        "657e8a6f-d5d6-4320-b9ee-27fa761546c4",
                        "64e3c412-6447-45cc-b765-fb87ffc4e04c",
                        "43b971e7-1005-46b6-be4f-5bbfe9557b6e",
                        "01a0f554-cebd-45b0-a0b6-b250772a153f",
                        "d16d42dd-e0c5-4519-aac4-ffae94437f84",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 99.5235595703125,
            "Text": "of 10 years. Vesting requirements are determined at the discre-",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.4033922851085663,
                    "Height": 0.014303342439234257,
                    "Left": 0.10678572952747345,
                    "Top": 0.8625104427337646,
                },
                "Polygon": [
                    {"X": 0.10678572952747345, "Y": 0.8625104427337646},
                    {"X": 0.5101780295372009, "Y": 0.8625104427337646},
                    {"X": 0.5101780295372009, "Y": 0.8768137693405151},
                    {"X": 0.10678572952747345, "Y": 0.8768137693405151},
                ],
            },
            "Id": "82a16924-0a92-4922-b434-2ed13b167eba",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "008bdb0b-529a-4a27-a2db-c400e31263d5",
                        "7732062c-3973-40e5-8199-59dbd5030cef",
                        "b50f88f3-47d2-41be-972c-627dd2f5d40d",
                        "e150f37b-b0b3-40cd-8d55-56902252d1b6",
                        "e1c53440-adba-4575-8562-2fcc47aeb900",
                        "feb44f8b-be31-4ab9-9297-41f953835528",
                        "3f53f662-56cc-4a6a-876f-8ed2d6189023",
                        "4a9e4048-1633-469f-9997-9bccc998e311",
                        "8105f299-3e61-4e19-86cb-63485657426d",
                        "1ca96edc-b440-42d5-9603-3bc07ab28544",
                    ],
                }
            ],
        },
        {
            "BlockType": "LINE",
            "Confidence": 97.06932067871094,
            "Text": "tion of the Compensation Committee of our Board of Directors.",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.4041402041912079,
                    "Height": 0.014050262048840523,
                    "Left": 0.10583650320768356,
                    "Top": 0.8776900172233582,
                },
                "Polygon": [
                    {"X": 0.10583650320768356, "Y": 0.8776900172233582},
                    {"X": 0.5099766850471497, "Y": 0.8776900172233582},
                    {"X": 0.5099766850471497, "Y": 0.8917402625083923},
                    {"X": 0.10583650320768356, "Y": 0.8917402625083923},
                ],
            },
            "Id": "9eba4393-2199-4acd-bf0b-c2515c7c6c16",
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [
                        "35edf877-5e79-46d4-89fe-d6fcc246e4cb",
                        "59e659da-078f-4cf4-9160-f9670543f867",
                        "b47bf102-998d-4c36-833c-d1115fea9791",
                        "6aaf622e-6f1a-4b0a-a530-a8408b64bb35",
                        "5ddef0a5-32a8-417b-b425-2be38651af28",
                        "44cd50c5-6e20-4c3f-bc8c-99a22cf4627f",
                        "f8000e14-b3de-472d-bc7f-745821bd7c2d",
                        "d8c43f0d-f6c6-494f-bd43-337f01f80f71",
                        "bb68ab5d-5a01-4f96-a556-6c4afd25f3fb",
                        "2319bcd7-5618-4513-b171-2c39cccf0725",
                    ],
                }
            ],
        },
        {
            "BlockType": "WORD",
            "Confidence": 81.72636413574219,
            "Text": "MOTES",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.02966049313545227,
                    "Height": 0.008166818879544735,
                    "Left": 0.7530537247657776,
                    "Top": 0.029821127653121948,
                },
                "Polygon": [
                    {"X": 0.7530537247657776, "Y": 0.029821127653121948},
                    {"X": 0.7827142477035522, "Y": 0.029821127653121948},
                    {"X": 0.7827142477035522, "Y": 0.03798794746398926},
                    {"X": 0.7530537247657776, "Y": 0.03798794746398926},
                ],
            },
            "Id": "91663486-5b2e-48db-80ac-c0de1f1ecf37",
        },
        {
            "BlockType": "WORD",
            "Confidence": 98.72019958496094,
            "Text": "to",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.013574687764048576,
                    "Height": 0.008314942009747028,
                    "Left": 0.7826061844825745,
                    "Top": 0.02961716055870056,
                },
                "Polygon": [
                    {"X": 0.7826061844825745, "Y": 0.02961716055870056},
                    {"X": 0.7961809039115906, "Y": 0.02961716055870056},
                    {"X": 0.7961809039115906, "Y": 0.037932101637125015},
                    {"X": 0.7826061844825745, "Y": 0.037932101637125015},
                ],
            },
            "Id": "cf234ec9-52cf-4710-94ce-288f0e055091",
        },
        {
            "BlockType": "WORD",
            "Confidence": 52.50002670288086,
            "Text": "CONSOLIDATED",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.06536715477705002,
                    "Height": 0.008873525075614452,
                    "Left": 0.7960178852081299,
                    "Top": 0.02976604737341404,
                },
                "Polygon": [
                    {"X": 0.7960178852081299, "Y": 0.02976604737341404},
                    {"X": 0.8613850474357605, "Y": 0.02976604737341404},
                    {"X": 0.8613850474357605, "Y": 0.03863957151770592},
                    {"X": 0.7960178852081299, "Y": 0.03863957151770592},
                ],
            },
            "Id": "ba7d7a8e-5ba6-4844-8ca2-667e3fc2196e",
        },
        {
            "BlockType": "WORD",
            "Confidence": 64.96131134033203,
            "Text": "FINANCIAL",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.046277016401290894,
                    "Height": 0.008866322226822376,
                    "Left": 0.8619109392166138,
                    "Top": 0.029761875048279762,
                },
                "Polygon": [
                    {"X": 0.8619109392166138, "Y": 0.029761875048279762},
                    {"X": 0.908187985420227, "Y": 0.029761875048279762},
                    {"X": 0.908187985420227, "Y": 0.03862819820642471},
                    {"X": 0.8619109392166138, "Y": 0.03862819820642471},
                ],
            },
            "Id": "b31c9d09-d226-4fe8-8a02-bf9ee134cef4",
        },
        {
            "BlockType": "WORD",
            "Confidence": 87.73054504394531,
            "Text": "STATEMENTS",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.056317396461963654,
                    "Height": 0.00884819496423006,
                    "Left": 0.7532274127006531,
                    "Top": 0.03825380280613899,
                },
                "Polygon": [
                    {"X": 0.7532274127006531, "Y": 0.03825380280613899},
                    {"X": 0.8095448017120361, "Y": 0.03825380280613899},
                    {"X": 0.8095448017120361, "Y": 0.04710199683904648},
                    {"X": 0.7532274127006531, "Y": 0.04710199683904648},
                ],
            },
            "Id": "1338f0a5-2904-40db-bbd5-5c1653f9eba5",
        },
        {
            "BlockType": "WORD",
            "Confidence": 93.70662689208984,
            "Text": "65",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.013070599175989628,
                    "Height": 0.009730551391839981,
                    "Left": 0.9296612739562988,
                    "Top": 0.06673943251371384,
                },
                "Polygon": [
                    {"X": 0.9296612739562988, "Y": 0.06673943251371384},
                    {"X": 0.9427318572998047, "Y": 0.06673943251371384},
                    {"X": 0.9427318572998047, "Y": 0.07646998763084412},
                    {"X": 0.9296612739562988, "Y": 0.07646998763084412},
                ],
            },
            "Id": "9c14566a-98ab-4f90-80bb-2929a3c8bb2a",
        },
        {
            "BlockType": "WORD",
            "Confidence": 99.17750549316406,
            "Text": "A",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.013529310002923012,
                    "Height": 0.011601833626627922,
                    "Left": 0.10605736076831818,
                    "Top": 0.15770190954208374,
                },
                "Polygon": [
                    {"X": 0.10605736076831818, "Y": 0.15770190954208374},
                    {"X": 0.11958666890859604, "Y": 0.15770190954208374},
                    {"X": 0.11958666890859604, "Y": 0.1693037450313568},
                    {"X": 0.10605736076831818, "Y": 0.1693037450313568},
                ],
            },
            "Id": "8c6e33a5-fcc0-4e74-babd-4f7dd25d373c",
        },
        {
            "BlockType": "WORD",
            "Confidence": 99.81588745117188,
            "Text": "summary",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.06285500526428223,
                    "Height": 0.012029532343149185,
                    "Left": 0.12134599685668945,
                    "Top": 0.16001352667808533,
                },
                "Polygon": [
                    {"X": 0.12134599685668945, "Y": 0.16001352667808533},
                    {"X": 0.18420100212097168, "Y": 0.16001352667808533},
                    {"X": 0.18420100212097168, "Y": 0.1720430552959442},
                    {"X": 0.12134599685668945, "Y": 0.1720430552959442},
                ],
            },
            "Id": "6b327141-872f-46cd-a7d7-0ed9a4207efe",
        },
        {
            "BlockType": "WORD",
            "Confidence": 99.97655487060547,
            "Text": "of",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.01616419292986393,
                    "Height": 0.011867513880133629,
                    "Left": 0.18603920936584473,
                    "Top": 0.1575692743062973,
                },
                "Polygon": [
                    {"X": 0.18603920936584473, "Y": 0.1575692743062973},
                    {"X": 0.2022034078836441, "Y": 0.1575692743062973},
                    {"X": 0.2022034078836441, "Y": 0.16943679749965668},
                    {"X": 0.18603920936584473, "Y": 0.16943679749965668},
                ],
            },
            "Id": "e0f96344-e11e-4312-aa29-efad7919be38",
        },
        {
            "BlockType": "WORD",
            "Confidence": 98.6279067993164,
            "Text": "future",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.04204616695642471,
                    "Height": 0.011879025027155876,
                    "Left": 0.20390979945659637,
                    "Top": 0.15758204460144043,
                },
                "Polygon": [
                    {"X": 0.20390979945659637, "Y": 0.15758204460144043},
                    {"X": 0.24595597386360168, "Y": 0.15758204460144043},
                    {"X": 0.24595597386360168, "Y": 0.16946107149124146},
                    {"X": 0.20390979945659637, "Y": 0.16946107149124146},
                ],
            },
            "Id": "97b75108-1005-4c45-90fe-e5daf375f209",
        },
        {
            "BlockType": "WORD",
            "Confidence": 93.04338836669922,
            "Text": "minimum",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.06212374195456505,
                    "Height": 0.011797991581261158,
                    "Left": 0.24762386083602905,
                    "Top": 0.15766611695289612,
                },
                "Polygon": [
                    {"X": 0.24762386083602905, "Y": 0.15766611695289612},
                    {"X": 0.3097476065158844, "Y": 0.15766611695289612},
                    {"X": 0.3097476065158844, "Y": 0.169464111328125},
                    {"X": 0.24762386083602905, "Y": 0.169464111328125},
                ],
            },
            "Id": "ed2e0aa8-6bb4-4689-aea7-c24af8c2a127",
        },
        {
            "BlockType": "WORD",
            "Confidence": 99.58238220214844,
            "Text": "lease",
            "TextType": "PRINTED",
            "Geometry": {
                "BoundingBox": {
                    "Width": 0.03811819851398468,
                    "Height": 0.011727862991392612,
                    "Left": 0.31180301308631897,
                    "Top": 0.1579698920249939,
                },
                "Polygon": [
                    {"X": 0.31180301308631897, "Y": 0.1579698920249939},
                    {"X": 0.34992119669914246, "Y": 0.1579698920249939},
                    {"X": 0.34992119669914246, "Y": 0.16969774663448334},
                    {"X": 0.31180301308631897, "Y": 0.16969774663448334},
                ],
            },
            "Id": "b82a2af4-e570-49cb-aee8-cf37e1aa21cb",
        },
    ],
    "DetectDocumentTextModelVersion": "1.0",
    "ResponseMetadata": {
        "RequestId": "50c33596-a5f6-4c60-a041-ac2571610815",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "50c33596-a5f6-4c60-a041-ac2571610815",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "438413",
            "date": "Wed, 01 Dec 2021 08:48:46 GMT",
        },
        "RetryAttempts": 0,
    },
}


def get_textract_response() -> JsonDict:
    """
    sample aws textract response
    """
    return _SAMPLE_TEXTRACT


_LAYOUT_INPUT = {
    "image": np.ones((1000, 1000, 3)),
    "ids": [
        "CLS",
        "3a696daf-15d5-3b88-be63-02912ef35cfb",
        "37d79fd7-ab87-30fe-b460-9b6e62e901b9",
        "5d40236e-430c-3d56-a8a3-fe9e46b872ac",
        "f8227d59-ea7f-342a-97fa-23df1f189762",
        "SEP",
    ],
    "boxes": [
        [0.0, 0.0, 0.0, 0.0],
        [110.0, 165.0, 130.0, 180.0],
        [140.0, 162.0, 180.0, 180.0],
        [100.0, 320.0, 130.0, 340.0],
        [175.0, 320.0, 205.0, 340.0],
        [1000.0, 1000.0, 1000.0, 1000.0],
    ],
    "tokens": ["CLS", "hello", "world", "bye", "word", "SEP"],
    "input_ids": [[101, 9875, 3207, 15630, 8569, 102]],
    "attention_mask": [[1, 1, 1, 1, 1, 1]],
    "token_type_ids": [[0, 0, 0, 0, 0, 0]],
}


def get_layoutlm_input() -> JsonDict:
    """
    layout lm model input from tokenizer
    """
    return _LAYOUT_INPUT


def get_token_class_result() -> List[TokenClassResult]:
    """
    token class result
    """
    uuids = _LAYOUT_INPUT["ids"]
    input_ids = _LAYOUT_INPUT["input_ids"]
    token_class_predictions = [0, 1, 1, 2, 2, 0]
    tokens = _LAYOUT_INPUT["tokens"]
    class_name = ["O", "B-FOO", "B-FOO", "I-FOO", "I-FOO", "O"]
    semantic_name = ["OTHER", "FOO", "FOO", "FOO", "FOO", "OTHER"]
    bio_tag = ["O", "B", "B", "I", "I", "O"]
    return [
        TokenClassResult(
            uuid=out[0],
            token_id=out[1],
            class_id=out[2],
            token=out[3],
            class_name=out[4],
            semantic_name=out[5],
            bio_tag=out[6],
        )
        for out in zip(
            uuids, input_ids[0], token_class_predictions, tokens, class_name, semantic_name, bio_tag  # type: ignore
        )
    ]
