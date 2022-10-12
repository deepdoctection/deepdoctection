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

from typing import Any, List, Tuple
from unittest.mock import MagicMock

from pytest import mark

from deepdoctection.dataflow import DataFlow
from deepdoctection.datasets.info import DatasetCategories
from deepdoctection.eval import metric_registry
from deepdoctection.eval.base import MetricBase
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.file_utils import apted_available


@mark.basic
def test_metric_registry_has_all_build_in_metric_registered() -> None:
    """
    test metric registry has all metrics registered
    """
    if apted_available():
        # cannot register when apted not installed
        assert len(metric_registry.get_all()) == 10
    else:
        assert len(metric_registry.get_all()) == 9


@mark.basic
def test_metric_registry_registered_new_metric() -> None:
    """
    test, that the new generated metric "TestMetric" can be registered and retrieved from registry
    """

    @metric_registry.register("testmetric")
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
        ) -> List[JsonDict]:
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
    test = metric_registry.get("testmetric")

    # Assert
    assert test == TestMetric
