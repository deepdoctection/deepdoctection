# -*- coding: utf-8 -*-
# File: test_pascalstruct.py

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
Unit tests for validating the functionality of the `pascal_voc_dict_to_image` method,
with emphasis on annotation mapping and filtering behavior.

This module defines unit tests to verify the accurate operation of the Pascal VOC dictionary
mapping functionalities, ensuring correct category mappings and appropriate handling of empty
datapoints. Test functions utilize fixtures and mock dependencies where necessary.
"""

from typing import Any

import numpy as np
import pytest

from dd_core.mapper.pascalstruct import pascal_voc_dict_to_image
from dd_core.utils.object_types import LayoutType, ObjectTypes


@pytest.fixture
def categories_name_as_keys() -> dict[ObjectTypes, int]:
    """Category names as keys for mapping Pascal VOC annotations to Image annotations."""
    return {
        LayoutType.TABLE: 1,
        LayoutType.LOGO: 2,
        LayoutType.FIGURE: 3,
        LayoutType.SIGNATURE: 4,
    }


@pytest.fixture
def category_names_mapping() -> dict[str, ObjectTypes]:
    """Category names mapping for Pascal VOC annotations to Image annotations."""
    return {
        "natural_image": LayoutType.FIGURE,
        "figure": LayoutType.FIGURE,
        "logo": LayoutType.LOGO,
        "signature": LayoutType.SIGNATURE,
        "table": LayoutType.TABLE,
    }


def test_pascal_voc_dict_to_image_maps_annotations(
    monkeypatch: pytest.MonkeyPatch,
    iiitar13k_datapoint: dict[str, Any],
    categories_name_as_keys: dict[str, int],  # pylint:disable=W0621
    category_names_mapping: dict[str, str],  # pylint:disable=W0621
) -> None:
    """Test mapping Pascal VOC annotations to Image annotations."""
    monkeypatch.setattr(
        "dd_core.mapper.pascalstruct.load_image_from_file",
        lambda fn: np.zeros((1100, 850, 3), dtype=np.uint8),
    )
    image = pascal_voc_dict_to_image(categories_name_as_keys, True, False, False, category_names_mapping)(
        iiitar13k_datapoint
    )
    assert image is not None
    assert image.image is not None
    annotations = image.get_annotation()
    assert len(annotations) == len(iiitar13k_datapoint["objects"])
    first_category = annotations[0].category_name
    assert first_category in categories_name_as_keys
    assert annotations[0].category_id == categories_name_as_keys[first_category]


def test_pascal_voc_dict_to_image_filters_empty_datapoints(
    monkeypatch: pytest.MonkeyPatch,
    iiitar13k_datapoint: dict[str, Any],
    categories_name_as_keys: dict[str, int],  # pylint:disable=W0621
    category_names_mapping: dict[str, str],  # pylint:disable=W0621
) -> None:
    """Test filtering empty Pascal VOC datapoints."""
    monkeypatch.setattr(
        "dd_core.mapper.pascalstruct.load_image_from_file",
        lambda fn: np.zeros((1100, 850, 3), dtype=np.uint8),
    )
    empty_dp = {**iiitar13k_datapoint, "objects": []}
    image = pascal_voc_dict_to_image(categories_name_as_keys, False, True, False, category_names_mapping)(empty_dp)
    assert image is None
