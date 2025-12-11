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

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import pytest

from dd_core.utils.object_types import ObjectTypes, get_type

try:
    from dd_datasets.base import DatasetCategories
except ImportError:
    DatasetCategories = None

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
