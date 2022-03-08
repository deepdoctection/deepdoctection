# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixtures
"""

import itertools
from typing import List

import numpy as np
from pytest import fixture

from deepdoctection.datapoint import BoundingBox, Image, ImageAnnotation
from deepdoctection.datasets import DatasetCategories
from deepdoctection.extern.base import DetectionResult


@fixture(name="datapoint_image")
def fixture_datapoint_image() -> Image:
    """
    fixture Image datapoint
    """
    img = Image(location="/test/to/path", file_name="test_name")
    img.image = np.ones([400, 600, 3], dtype=np.float32)
    return img


@fixture(name="annotations")
def fixture_annotation() -> List[ImageAnnotation]:
    """
    annotations
    """
    row_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=15.0, uly=100.0, lrx=60.0, lry=150.0, absolute_coords=True),
            category_name="ROW",
            category_id="1",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=15.0, uly=200.0, lrx=70.0, lry=240.0, absolute_coords=True),
            category_name="ROW",
            category_id="1",
        ),
    ]

    col_anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=10.0, uly=50.0, lrx=20.0, lry=250.0, absolute_coords=True),
            category_name="COLUMN",
            category_id="2",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=40.0, uly=20.0, lrx=50.0, lry=240.0, absolute_coords=True),
            category_name="COLUMN",
            category_id="2",
        ),
    ]
    return list(itertools.chain(row_anns, col_anns))


@fixture(name="image_with_anns")
def fixture_image_with_anns(datapoint_image: Image, annotations: List[ImageAnnotation]) -> Image:
    """
    image with annotations
    """
    for ann in annotations:
        datapoint_image.dump(ann)

    return datapoint_image


@fixture(name="categories")
def fixture_categories() -> DatasetCategories:
    """
    categories
    """
    return DatasetCategories(init_categories=["ROW", "COLUMN"])


@fixture(name="detection_results")
def fixture_detection_results() -> List[DetectionResult]:
    """
    detection results
    """
    detect_results_list = [
        DetectionResult(box=[15.0, 100.0, 60.0, 150.0], score=0.9, class_id=1, class_name="ROW"),
        DetectionResult(box=[15.0, 200.0, 70.0, 240.0], score=0.8, class_id=1, class_name="ROW"),
        DetectionResult(box=[10.0, 50.0, 20.0, 250.0], score=0.7, class_id=2, class_name="COLUMN"),
    ]

    return detect_results_list
