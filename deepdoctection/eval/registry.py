# -*- coding: utf-8 -*-
# File: registry.py

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
Module for MetricRegistry
"""

from typing import Dict, List, Type

from ..eval.base import MetricBase
from .accmetric import AccuracyMetric, ConfusionMetric
from .cocometric import CocoMetric

_METRICS: Dict[str, Type[MetricBase]] = dict(
    [("accuracy", AccuracyMetric), ("confusion", ConfusionMetric), ("coco", CocoMetric)]
)


class MetricRegistry:
    """
    The MetricRegistry is the class for receiving metrics and registering new ones.
    """

    @staticmethod
    def print_metric_names() -> None:
        """
        Print a list of registered metric names
        """
        print(list(_METRICS.keys()))

    @staticmethod
    def get_metric(name: str) -> Type[MetricBase]:
        """
        Returns metric class with a given name

        :param name: metric name
        :return: An metric class
        """
        return _METRICS[name]

    @staticmethod
    def register_metric(name: str, metric: Type[MetricBase]) -> None:
        """
        Register a new metric.

        :param name: A metric name
        :param metric: A new metric class to add to the registry.
        """
        _METRICS[name] = metric

    @staticmethod
    def get_metric_names() -> List[str]:
        """
        Get a list of available metric names

        :return: A list of names
        """
        return list(_METRICS.keys())
