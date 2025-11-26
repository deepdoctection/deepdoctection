# -*- coding: utf-8 -*-
# File: base.py

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
Base classes for evaluations and metrics
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ..dataflow import DataFlow
from ..datasets.info import DatasetCategories
from ..utils.error import DependencyError
from ..utils.file_utils import Requirement
from ..utils.types import MetricResults


class MetricBase(ABC):
    """
    Base class for metrics. Metrics only exist as classes and are not instantiated. They consist of two class variables:

        - A metric function that reads ground truth and prediction in an already transformed data format and
          returns a distance as an evaluation result.

        - A mapping function that transforms an image datapoint into a valid input format ready to ingest by the metric
          function.

    Using `get_distance`, ground truth and prediction dataflow can be read in and evaluated.
    `dump` is a helper method that is often called via `get_distance`. Here, the dataflows should be
    executed and the results should be saved in separate lists.

    Attributes:
        name (str): Name of the metric, usually the class name.
        metric (Callable[[Any, Any], Optional[Any]]): The metric function that computes the distance.
        _results (list[MetricResults]): Internal storage for results of the metric computation.
    """

    name: str
    metric: Callable[[Any, Any], Optional[Any]]
    _results: list[MetricResults]

    def __new__(cls, *args, **kwargs):  # type: ignore # pylint: disable=W0613
        requirements = cls.get_requirements()
        name = cls.__name__ if hasattr(cls, "__name__") else cls.__class__.__name__
        if not all(requirement[1] for requirement in requirements):
            raise DependencyError(
                "\n".join(
                    [f"{name} has the following dependencies:"]
                    + [requirement[2] for requirement in requirements if not requirement[1]]
                )
            )
        return super().__new__(cls)

    @classmethod
    @abstractmethod
    def get_requirements(cls) -> list[Requirement]:
        """
        Get a list of requirements for running the detector

        Returns:
            List of requirements
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        """
        Takes of the ground truth dataflow as well as the dataflow and generates the metric results.

        Args:
            dataflow_gt: Dataflow with ground truth annotations.
            dataflow_predictions: Dataflow with predictions.
            categories: DatasetCategories with respect to the underlying dataset.

        Returns:
            List of metric results
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> tuple[Any, Any]:
        """
        Dump the dataflow with ground truth annotations and predictions. Use it as auxiliary method and call it from
        `get_distance`.

        Args:
            dataflow_gt: Dataflow with ground truth annotations.
            dataflow_predictions: Dataflow with predictions.
            categories: DatasetCategories with respect to the underlying dataset.

        Returns:
            Tuple containing ground truth and predictions
        """
        raise NotImplementedError()

    @classmethod
    def result_list_to_dict(cls, results: list[MetricResults]) -> MetricResults:
        """
        Converts the result from `get_distance` to a dict. It concatenates all keys of the inner dict and uses
        the metric result 'val' as value.

        Args:
            results: List of dict as input

        Returns:
            MetricResults: Dict with metric results.
        """
        output: MetricResults = {}
        for res in results:
            new_key = ""
            new_val = 0.0
            for k, val in res.items():
                if str(val) != "":
                    if k != "val":
                        if k != "key":
                            new_key += k + "/" + str(val) + "/"
                        else:
                            new_key += str(val) + "/"
                    else:
                        new_val = val
            output[new_key[:-1]] = new_val

        return output

    @classmethod
    def print_result(cls) -> None:
        """Print metric result. Overwrite this method if you want a specific output"""
