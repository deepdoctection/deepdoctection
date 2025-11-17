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
Fixtures for dataflow package testing
"""

from typing import Any, Dict, List

from pytest import fixture

from dd_datasets.dataflow.serialize import DataFromList


@fixture(name="simple_list_dataflow")
def fixture_simple_list_dataflow() -> DataFromList:
    """
    Simple dataflow with list elements
    """
    data = [["a", "b"], ["c", "d"], ["e", "f"]]
    return DataFromList(data, shuffle=False)


@fixture(name="simple_dict_dataflow")
def fixture_simple_dict_dataflow() -> DataFromList:
    """
    Simple dataflow with dict elements
    """
    data = [{"key1": "a", "key2": 1}, {"key1": "b", "key2": 2}, {"key1": "c", "key2": 3}]
    return DataFromList(data, shuffle=False)


@fixture(name="nested_list_dataflow")
def fixture_nested_list_dataflow() -> DataFromList:
    """
    Dataflow with nested lists for FlattenData testing
    """
    data = [["item1", "item2"], ["item3", "item4"]]
    return DataFromList(data, shuffle=False)


@fixture(name="dict_list_for_flatten")
def fixture_dict_list_for_flatten() -> List[Dict[str, Any]]:
    """
    List of dicts for FlattenData testing
    """
    return [{"a": 1, "b": 2}, {"c": 3, "d": 4}]

