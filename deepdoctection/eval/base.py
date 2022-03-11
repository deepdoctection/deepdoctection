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
Module for the base class for evaluations and metrics
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

from ..dataflow import DataFlow
from ..datasets.info import DatasetCategories
from ..mapper.maputils import DefaultMapper
from ..utils.detection_types import JsonDict


class MetricBase(ABC):
    """
    Base class for metrics. Metrics only exist as classes and are not instantiated. They consist of two class variables:

        - A metric function that reads ground truth and prediction in an already transformed data format and
          returns a distance as an evaluation result.

        - A mapping function that transforms an image datapoint into a valid input format ready to ingest by the metric
          function.

    Using :meth:`get_distance`, ground truth and prediction dataflow can be read in and evaluated.
    :meth:`dump` is a helper method that is often called via :meth:`get_distance`. Here, the dataflows should be
    executed and the results should be saved in separate lists.
    """

    metric: Optional[Callable[[Any], Tuple[Any, Any]]] = None
    mapper: Optional[DefaultMapper] = None

    @classmethod
    @abstractmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories, as_dict: bool = False
    ) -> Union[List[JsonDict], JsonDict]:
        """
        Takes of the ground truth processing strand as well as the prediction strand and generates the metric results.

        :param dataflow_gt: Dataflow with ground truth annotations.
        :param dataflow_predictions: Dataflow with predictions.
        :param categories:  DatasetCategories with respect to the underlying dataset.
        :param as_dict: Will return the metric result as a dict (Default: False). See :meth:`result_list_to_dict`.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> Tuple[Any, Any]:
        """
        Dump the dataflow with ground truth annotations and predictions. Use it as auxiliary method and call it from
        :meth:`get_distance`.

        :param dataflow_gt: Dataflow with ground truth annotations.
        :param dataflow_predictions: Dataflow with predictions.
        :param categories: DatasetCategories with respect to the underlying dataset.
        """
        raise NotImplementedError

    @classmethod
    def result_list_to_dict(cls, results: List[JsonDict]) -> JsonDict:
        """
        Converts the result from :meth:`get_distance` to a dict. It concatenates all keys of the inner dict and uses
        the metric result 'val' as value.

        :param results: List of dict as input
        :return: Dict with metric results.
        """
        output: JsonDict = {}
        for res in results:
            new_key = ""
            new_val = 0.0
            for k, val in res.items():

                if str(val) != "":
                    if k != "val":
                        new_key += k + "/" + str(val) + "/"
                    else:
                        new_val = val
            output[new_key[:-1]] = new_val

        return output
