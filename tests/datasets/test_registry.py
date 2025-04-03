# -*- coding: utf-8 -*-
# File: test_registry.py

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
Testing module datasets.instances.registry
"""

from unittest.mock import MagicMock

from pytest import mark

from deepdoctection.datasets.base import DatasetBase
from deepdoctection.datasets.dataflow_builder import DataFlowBaseBuilder
from deepdoctection.datasets.info import DatasetCategories, DatasetInfo
from deepdoctection.datasets.registry import dataset_registry, get_dataset


@mark.basic
def test_dataset_registry_has_all_build_in_datasets_registered() -> None:
    """
    test dataset registry has all pipeline components registered
    """
    assert len(dataset_registry.get_all()) == 12


@mark.basic
def test_dataset_registry_registered_new_dataset() -> None:
    """
    test, that the new generated dataset component "TestDataset" can be registered and retrieved from registry
    """

    @dataset_registry.register("testdata")
    class TestDataset(DatasetBase):
        """
        TestDataset
        """

        @classmethod
        def _info(cls) -> DatasetInfo:
            """
            Processing an image through the whole pipeline component.
            """
            return MagicMock()

        def _categories(self) -> DatasetCategories:
            """
            _categories
            """
            return MagicMock()

        def _builder(self) -> DataFlowBaseBuilder:
            """
            _builder
            """
            return MagicMock()

    # Act
    test = get_dataset("testdata")

    # Assert
    assert isinstance(test, TestDataset)
