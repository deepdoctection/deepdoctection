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
Testing the module eval.registry
"""

from typing import Any, List, Tuple, Union
from unittest.mock import MagicMock

from deepdoctection.dataflow import DataFlow
from deepdoctection.datasets.info import DatasetCategories
from deepdoctection.eval import MetricRegistry
from deepdoctection.eval.base import MetricBase
from deepdoctection.utils.detection_types import JsonDict


def test_metric_registry_has_all_build_in_metric_registered() -> None:
    """
    test metric registry has all metrics registered
    """
    assert len(MetricRegistry.get_metric_names()) == 3


def test_metric_registry_registered_new_metric() -> None:
    """
    test, that the new generated metric "TestMetric" can be registered and retrieved from registry
    """

    class TestMetric(MetricBase):
        """
        TestMetric
        """

        @classmethod
        def get_distance(
            cls,
            dataflow_gt: DataFlow,
            dataflow_predictions: DataFlow,
            categories: DatasetCategories,
            as_dict: bool = False,
        ) -> Union[List[JsonDict], JsonDict]:
            """
            get distance
            """
            return MagicMock()

        @classmethod
        def dump(
            cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
        ) -> Tuple[Any, Any]:
            """
            dump
            """
            return MagicMock()

    # Act
    MetricRegistry.register_metric("testmetric", TestMetric)
    test = MetricRegistry.get_metric("testmetric")

    # Assert
    assert test == TestMetric
