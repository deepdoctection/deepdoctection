# -*- coding: utf-8 -*-
# File: dataflow_builder.py

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
Module for DataflowBase class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from ..dataflow import DataFlow
from ..utils.systools import get_dataset_dir_path
from .info import DatasetCategories


class DataFlowBaseBuilder(ABC):
    """
    Abstract base class for building the dataflow of a dataset.

    DataFlowBase has an abstract :meth:`build` that returns the dataflow of a dataset. The dataflow should be
    designed in such a way that each data point is already mapped in the form of the core data model and thus
    corresponds to a :class:`datapoint.Image` instance. Any characteristics can be passed as arguments and implemented,
    which influence the return of the dataflow. These include, for example, the "split", "max_datapoints" but also
    specific further transformations, such as cutting and returning an annotation as a sub image. Within this method,
    checks and consistency checks should also be carried out so that a curated data flow is available as return value.
    Such specific transformations should be implemented by transferring a value of the argument "build_mode".
    """

    def __init__(
        self,
        location: str,
        annotation_files: Optional[Dict[str, Union[str, List[str]]]] = None,
    ):
        """
        :param location: Relative path of the physical dataset.
        :param annotation_files: Dict of annotation files e.g. depending on the split.
        """
        self.location = location
        if annotation_files is None:
            annotation_files = {}
        self.annotation_files = annotation_files
        self._categories: Optional[DatasetCategories] = None
        self._splits: Dict[str, str] = {}

    @property
    def categories(self) -> Optional[DatasetCategories]:
        """
        categories
        """
        return self._categories

    @categories.setter
    def categories(self, categories: DatasetCategories) -> None:
        """
        categories setter
        """
        self._categories = categories

    def get_split(self, key: str) -> str:
        """
        split value
        """
        return self._splits.get(key, "")

    @property
    def splits(self) -> Dict[str, str]:
        """
        splits
        """
        return self._splits

    @splits.setter
    def splits(self, splits: Dict[str, str]) -> None:
        """
        set splits
        """
        self._splits = splits

    def get_workdir(self) -> str:
        """
        Get the absolute path to the locally physically stored dataset.

        :return: local workdir
        """
        return get_dataset_dir_path() + self.location

    @abstractmethod
    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Consult the docstring w.r.t :class:`DataFlowBaseBuilder`.

        :param kwargs: A custom set of arguments/values
        :return: dataflow
        """
        raise NotImplementedError
