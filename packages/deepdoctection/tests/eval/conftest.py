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

"""
Module providing fixtures for testing.

This module contains reusable pytest fixtures related to dataset categories.
Fixtures help in externalizing commonly used data or configurations, enabling
test clarity and reusability. These fixtures define mock objects for testing
purposes.

Functions:
    dataset_categories: Provides a DatasetCategories object initialized with
        predefined categories and subcategories.
    fixture_categories: Returns a dictionary mapping integer keys to
        predefined ObjectTypes.
    eval_doc_structured_output: Resolved `structured_output` of the `eval_doc` fixture document.
    eval_doc_dataset: `CustomDataset` built from the `eval_doc` fixture document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pytest

import shared_test_utils as stu
from dd_core.doc import Document
from dd_core.utils.object_types import LayoutLabel, ObjectTypes, get_type

try:
    from dd_datasets.base import CustomDataset, DatasetCategories
    from dd_datasets.doc_factory import DocumentDatasetFactory
except ImportError:
    DatasetCategories = None  # type: ignore
    CustomDataset = None  # type: ignore
    DocumentDatasetFactory = None  # type: ignore

if TYPE_CHECKING:
    from dd_datasets.info import DatasetCategories


@pytest.fixture
def dataset_categories() -> DatasetCategories:
    """
    fixture categories
    """
    _categories = [get_type("table"), get_type("cell"), get_type("row"), get_type("column")]
    _sub_categories: Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]] = {
        get_type("row"): {get_type("row_number"): []},
        get_type("column"): {get_type("column_number"): []},
        get_type("cell"): {
            get_type("row_number"): [],
            get_type("column_number"): [],
            get_type("row_span"): [],
            get_type("column_span"): [],
        },
    }
    return DatasetCategories(_categories, _sub_categories)


@pytest.fixture
def fixture_categories() -> dict[int, ObjectTypes]:
    """
    Categories as Dict
    """
    return {
        1: get_type("text"),
        2: get_type("title"),
        3: get_type("table"),
        4: get_type("figure"),
        5: get_type("list"),
    }


@pytest.fixture
def eval_doc_structured_output(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Resolved `structured_output` of the `eval_doc` fixture document: a two-page bank statement with
    `accountHolder`, `balances` and `bookings` fields, the latter being a list of booking records.
    """
    monkeypatch.setenv("ENABLE_DYNAMIC_OBJECT_TYPES", "True")
    doc = Document.from_json(stu.asset_path("eval_doc"))
    return doc.structured_output


@pytest.fixture
def eval_doc_dataset(monkeypatch: pytest.MonkeyPatch) -> CustomDataset:
    """
    `CustomDataset` built from the `eval_doc` fixture document. `init_categories` covers `table`, `text` and
    `title`, the categories `RecordAlignMetric`'s default `image_or_docs_to_cat_id` mapper compares.

    `location` is passed as the absolute path to the fixture directory: `SETTINGS.DATASET_DIR / location`
    then resolves to that absolute path regardless of `SETTINGS.DATASET_DIR`, so nothing needs to be
    copied or monkeypatched into place.
    """
    monkeypatch.setenv("ENABLE_DYNAMIC_OBJECT_TYPES", "True")

    return DocumentDatasetFactory.make(
        name="eval_doc",
        location=str(stu.asset_path("eval_doc").parent),
        dataset_type="object_detection",
        init_categories=[LayoutLabel.TABLE, LayoutLabel.TEXT, LayoutLabel.TITLE],
    )
