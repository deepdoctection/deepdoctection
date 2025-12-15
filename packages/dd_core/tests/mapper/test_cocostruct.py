# -*- coding: utf-8 -*-
# File: test_cocostruct.py

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
Unit tests for COCO structure mapping functions.

This module provides tests to validate the mapping of COCO annotations
to the internal ``Image`` structure and vice versa. It ensures that the
mapping correctly handles various scenarios such as image loading, fake
scores, and coarse sub-categories.

Tests in this module cover:
- Mapping COCO annotations to ``Image`` objects without loading images.
- Mapping COCO annotations with options for fake scores and sub-categories.
- Validating the correct handling of annotation details during the mapping.
- Exporting an ``Image`` object to COCO format and validating its correctness.

The test cases utilize pytest for assertions and mocking where applicable.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from dd_core.datapoint.image import Image
from dd_core.mapper.cocostruct import coco_to_image, image_to_coco

# name -> id mapping (spec input)
CATEGORIES_NAME_TO_ID: Dict[str, int] = {"text": 1, "title": 2, "list": 3, "table": 4, "figure": 5}
# id -> name mapping required by coco_to_image(categories=...)
CATEGORIES_ID_TO_NAME: Dict[int, str] = {v: k for k, v in CATEGORIES_NAME_TO_ID.items()}
COARSE_MAPPING: Dict[int, int] = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


def test_coco_to_image_no_load_no_fake_score(coco_datapoint: Dict[str, Any]) -> None:
    """Map COCO without image loading or fake scores."""
    img: Optional[Image] = coco_to_image(
        categories=CATEGORIES_ID_TO_NAME,
        load_image=False,
        filter_empty_image=True,
        fake_score=False,
    )(coco_datapoint)
    assert img is not None
    anns: List[Any] = img.get_annotation()
    assert len(anns) > 0
    assert all(ann.score is None for ann in anns)


def test_coco_to_image_no_load_fake_score(coco_datapoint: Dict[str, Any]) -> None:
    """Map COCO without image loading but with fake scores and coarse sub-category."""
    img: Optional[Image] = coco_to_image(
        categories=CATEGORIES_ID_TO_NAME,
        load_image=False,
        filter_empty_image=True,
        fake_score=True,
        coarse_mapping=COARSE_MAPPING,
        coarse_sub_cat_name="sub_cat_1",
    )(coco_datapoint)
    assert img is not None
    anns: List[Any] = img.get_annotation()
    assert len(anns) > 0
    for ann in anns:
        assert ann.score is not None
        sub = ann.get_sub_category("sub_cat_1")
        assert sub is not None
        assert sub.category_name == "text"
        assert sub.category_id == 1


def test_coco_to_image_load_no_fake_score(monkeypatch: pytest.MonkeyPatch, coco_datapoint: Dict[str, Any]) -> None:
    """Map COCO with image loading and no fake scores."""
    monkeypatch.setattr(
        "dd_core.mapper.cocostruct.load_image_from_file",
        lambda fn: np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    img: Optional[Image] = coco_to_image(
        categories=CATEGORIES_ID_TO_NAME,
        load_image=True,
        filter_empty_image=True,
        fake_score=False,
    )(coco_datapoint)
    assert img is not None
    anns: List[Any] = img.get_annotation()
    assert len(anns) > 0
    assert all(ann.score is None for ann in anns)


def test_coco_to_image_load_fake_score(monkeypatch: pytest.MonkeyPatch, coco_datapoint: Dict[str, Any]) -> None:
    """Map COCO with image loading, fake scores and coarse sub-category."""
    monkeypatch.setattr(
        "dd_core.mapper.cocostruct.load_image_from_file",
        lambda fn: np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    img: Optional[Image] = coco_to_image(
        categories=CATEGORIES_ID_TO_NAME,
        load_image=True,
        filter_empty_image=True,
        fake_score=True,
        coarse_mapping=COARSE_MAPPING,
        coarse_sub_cat_name="sub_cat_1",
    )(coco_datapoint)
    assert img is not None
    anns: List[Any] = img.get_annotation()
    assert len(anns) > 0
    for ann in anns:
        assert ann.score is not None
        sub = ann.get_sub_category("sub_cat_1")
        assert sub is not None
        assert sub.category_name == "text"
        assert sub.category_id == 1


def test_image_to_coco_table_image(table_image: Image) -> None:
    """Validate COCO export for table_image."""
    img_dict, anns = image_to_coco(table_image)

    # Image dict checks
    assert img_dict["id"] == 862774443987137741935
    assert img_dict["height"] == 163
    assert img_dict["width"] == 165
    assert img_dict["file_name"] == "PMC1253705_00002.jpg"

    # Annotation list checks
    assert len(anns) == 24
    assert sum(1 for a in anns if a["category_id"] == 2) == 3
    assert sum(1 for a in anns if a["category_id"] == 3) == 5
