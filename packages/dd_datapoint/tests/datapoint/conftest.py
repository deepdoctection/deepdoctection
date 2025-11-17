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
Fixtures for datapoint package testing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pytest import fixture

import shared_test_utils as stu


@fixture(name="white_image")
def fixture_image() -> stu.WhiteImage:
    """Provide a white test image"""
    return stu.build_white_image()


@fixture(name="pdf_page")
def fixture_pdf_page() -> stu.TestPdfPage:
    """Provide a deterministic 1-page PDF for rendering tests."""
    return stu.build_test_pdf_page()


#@fixture(name="black_image")
#def fixture_black_image() -> np.ndarray:
#    """Provide a black test image (10x10x3)"""
#    return np.zeros([10, 10, 3], dtype=np.uint8)


#@fixture(name="colored_image")
#def fixture_colored_image() -> np.ndarray:
#    """Provide a colored test image with RGB channels"""
#    img = np.zeros([8, 12, 3], dtype=np.uint8)
#    img[:, :, 0] = 255  # Blue channel
#    img[:, :, 1] = 128  # Green channel
#    img[:, :, 2] = 64   # Red channel
#    return img


#@fixture(name="large_image")
#def fixture_large_image() -> np.ndarray:
#    """Provide a larger test image (100x150x3)"""
#    return np.ones([100, 150, 3], dtype=np.uint8) * 127


#def anns_to_ids(annotations: list) -> list[str]:
#    """Helper function to extract annotation ids from a list of annotations"""
#    return [ann.annotation_id for ann in annotations]


#@fixture(name="dp_image_with_layout_and_word_annotations")
#def fixture_dp_image_with_layout_and_word_annotations():
#    """Provide an image with layout and word annotations for testing"""
#    from dd_datapoint.datapoint import Image, ImageAnnotation, BoundingBox

#    img = Image(file_name="test_layout.png", location="/test/path")
#    img.image = np.ones([100, 100, 3], dtype=np.uint8)

#    # Add layout annotations
#    layout_ann = ImageAnnotation(
#        category_name="LAYOUT",
#        category_id=1,
#        bounding_box=BoundingBox(ulx=10.0, uly=10.0, width=80.0, height=80.0, absolute_coords=True),
#        service_id="test_service",
#    )
#    layout_ann.annotation_id = "51fca38d-b181-3ea2-9c97-7e265febcc86"
#    img.dump(layout_ann)

    # Add word annotation
#    word_ann = ImageAnnotation(
#        category_name="WORD",
#        category_id=2,
#        bounding_box=BoundingBox(ulx=20.0, uly=20.0, width=30.0, height=10.0, absolute_coords=True),
#        service_id="test_service",
#    )
#    word_ann.annotation_id = "1413d499-ce19-3a50-861c-7d8c5a7ba772"
#    img.dump(word_ann)

#    return img


#@fixture(name="annotation_maps")
#def fixture_annotation_maps():
#    """Provide expected annotation maps for testing"""
#    from dd_datapoint.datapoint import AnnotationMap
#    from collections import defaultdict

#    maps: defaultdict[str, list[AnnotationMap]] = defaultdict(list)
#    maps["51fca38d-b181-3ea2-9c97-7e265febcc86"].append(
#        AnnotationMap(
#            image_annotation_id="51fca38d-b181-3ea2-9c97-7e265febcc86",
#            sub_category_key=None,
#            relationship_key=None,
#            summary_key=None,
#        )
#    )
#    maps["1413d499-ce19-3a50-861c-7d8c5a7ba772"].append(
#        AnnotationMap(
#            image_annotation_id="1413d499-ce19-3a50-861c-7d8c5a7ba772",
#            sub_category_key=None,
#            relationship_key=None,
#            summary_key=None,
#        )
#    )
#    return maps


@fixture(name="service_id_to_ann_id")
def fixture_service_id_to_ann_id():
    """Provide expected service_id to annotation_id mapping"""
    return {
        "test_service": [
            "51fca38d-b181-3ea2-9c97-7e265febcc86",
            "1413d499-ce19-3a50-861c-7d8c5a7ba772"
        ]
    }


def get_test_path():
    """Get the test data path"""
    from pathlib import Path
    return Path(__file__).parent / "test_data"


