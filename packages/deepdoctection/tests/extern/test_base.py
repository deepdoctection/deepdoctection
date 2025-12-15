# -*- coding: utf-8 -*-
# File: test_base.py

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
Unit tests for various functionalities of DeepDoctection model categories and transformations.

This module contains tests that verify the behavior of the `ModelCategories` and `DeterministicImageTransformer` classes
and their corresponding methods. Tests include validation of category manipulation, coordinate transformations, and
image transformations.

"""

from types import MappingProxyType

import numpy as np

from dd_core.utils.object_types import get_type
from deepdoctection.extern.base import (
    BaseTransform,
    DetectionResult,
    DeterministicImageTransformer,
    ModelCategories,
    NerModelCategories,
)


def test_model_categories_get_categories_dict(model_categories: ModelCategories) -> None:
    """test get_categories_dict"""
    cats = model_categories.get_categories()
    expected = MappingProxyType(
        {
            1: get_type("word"),
            2: get_type("line"),
            3: get_type("table"),
            4: get_type("figure"),
            5: get_type("header"),
            6: get_type("footnote"),
        }
    )
    assert cats == expected


def test_model_categories_get_categories_dict_name_as_key(model_categories: ModelCategories) -> None:
    """test get_categories_dict with name_as_key=True"""
    cats = model_categories.get_categories(name_as_key=True)  # type: ignore
    expected = MappingProxyType(
        {
            get_type("word"): 1,
            get_type("line"): 2,
            get_type("table"): 3,
            get_type("figure"): 4,
            get_type("header"): 5,
            get_type("footnote"): 6,
        }
    )
    assert cats == expected


def test_model_categories_get_categories_tuple(model_categories: ModelCategories) -> None:
    """test get_categories_tuple"""
    cats = model_categories.get_categories(as_dict=False)
    expected = (
        get_type("word"),
        get_type("line"),
        get_type("table"),
        get_type("figure"),
        get_type("header"),
        get_type("footnote"),
    )
    assert cats == expected


def test_model_categories_filter(model_categories: ModelCategories) -> None:
    """test filter_categories"""
    model_categories.filter_categories = (get_type("word"), get_type("header"))
    cats = model_categories.get_categories()
    expected = MappingProxyType(
        {2: get_type("line"), 3: get_type("table"), 4: get_type("figure"), 6: get_type("footnote")}
    )
    assert cats == expected


def test_model_categories_shift_ids(model_categories: ModelCategories) -> None:
    """test shift_category_ids"""
    shifted = model_categories.shift_category_ids(-1)
    expected = MappingProxyType(
        {
            0: get_type("word"),
            1: get_type("line"),
            2: get_type("table"),
            3: get_type("figure"),
            4: get_type("header"),
            5: get_type("footnote"),
        }
    )
    assert shifted == expected


def test_ner_model_categories_merge(ner_model_categories: NerModelCategories) -> None:
    """test merge_ner_categories"""
    cats = ner_model_categories.get_categories()
    expected = MappingProxyType(
        {
            1: get_type("B-answer"),
            2: get_type("B-question"),
            3: get_type("I-answer"),
            4: get_type("I-question"),
        }
    )
    assert cats == expected


def test_ner_model_categories_preserve_init(ner_semantics: tuple[str, str], ner_bio: tuple[str, str]) -> None:
    """test merge_ner_categories with init_categories"""
    nm = NerModelCategories(
        init_categories={1: get_type("B-answer"), 2: get_type("B-question")},
        categories_semantics=ner_semantics,
        categories_bio=ner_bio,
    )
    cats = nm.get_categories()
    expected = MappingProxyType(
        {1: get_type("B-answer"), 2: get_type("B-question"), 3: get_type("I-answer"), 4: get_type("I-question")}
    )
    assert cats == expected


def test_transform_image(transformer: DeterministicImageTransformer, mock_base_transform: BaseTransform) -> None:
    """test transform_image"""
    img = np.zeros((10, 10, 3))
    spec = DetectionResult()
    out = transformer.transform_image(img, spec)  # type: ignore
    mock_base_transform.apply_image.assert_called_once_with(img)  # type: ignore
    assert np.array_equal(out, np.ones((10, 10, 3)))


def test_transform_coords(
    transformer: DeterministicImageTransformer,
    mock_base_transform: BaseTransform,
    detection_results: list[DetectionResult],
) -> None:
    """test transform_coords"""
    out = transformer.transform_coords(detection_results)
    mock_base_transform.apply_coords.assert_called_once()  # type: ignore
    assert len(out) == 2
    assert out[0].uuid == detection_results[0].uuid
    assert out[1].uuid == detection_results[1].uuid
    assert out[0].box == [10, 10, 20, 20]
    assert out[1].box == [30, 30, 40, 40]
    assert out[0].class_name == "report_date"
    assert out[1].class_name == "umbrella"


def test_inverse_transform_coords(
    transformer: DeterministicImageTransformer,
    mock_base_transform: BaseTransform,
    detection_results: list[DetectionResult],
) -> None:
    """test inverse_transform_coords"""
    out = transformer.inverse_transform_coords(detection_results)
    mock_base_transform.inverse_apply_coords.assert_called_once()  # type: ignore
    assert len(out) == 2
    assert out[0].uuid == detection_results[0].uuid
    assert out[1].uuid == detection_results[1].uuid
    assert out[0].box == [5, 5, 15, 15]
    assert out[1].box == [25, 25, 35, 35]
    assert out[0].class_id == 1
    assert out[1].class_id == 2


def test_predict(transformer: DeterministicImageTransformer) -> None:
    """test predict"""
    img = np.zeros((10, 10, 3))
    dr = transformer.predict(img)  # type: ignore
    assert dr.angle == 90
