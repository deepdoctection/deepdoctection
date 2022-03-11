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
Module for the base class of datasets.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from .dataflow_builder import DataFlowBaseBuilder
from .info import DatasetCategories, DatasetInfo


class DatasetBase(ABC):
    """
    Base class for a dataset. Requires to implementing :meth:`_categories` :meth:`_info` and :meth:`_builder` by
    yourself. These methods must return a DatasetCategories, a DatasetInfo and a DataFlow_Builder instance, which
    together give a complete description of the dataset. Compare some specific dataset cards in the :mod:`instance` .
    """

    def __init__(self) -> None:
        assert self._info() is not None, "Dataset requires at least a name defined in DatasetInfo"
        self._dataset_info = self._info()
        self._dataflow_builder = self._builder()
        self._dataflow_builder.categories = self._categories()
        self._dataflow_builder.splits = self._dataset_info.splits

        if not self.dataset_available() and self.is_built_in():
            print(
                f"Dataset {self._dataset_info.name} not locally found. Please download at {self._dataset_info.url}"
                f" and place under {self._dataflow_builder.get_workdir()}"
            )

    @property
    def dataset_info(self) -> DatasetInfo:
        """
        dataset_info
        """
        return self._dataset_info

    @property
    def dataflow(self) -> DataFlowBaseBuilder:
        """
        dataflow
        """
        return self._dataflow_builder

    @abstractmethod
    def _categories(self) -> DatasetCategories:
        """
        Construct the DatasetCategory object.
        """

        raise NotImplementedError

    @abstractmethod
    def _info(self) -> DatasetInfo:
        """
        Construct the DatasetInfo object.
        """

        raise NotImplementedError

    @abstractmethod
    def _builder(self) -> DataFlowBaseBuilder:
        """
        Construct the DataFlowBaseBuilder object. It needs to be implemented in the derived class.
        """

        raise NotImplementedError

    def dataset_available(self) -> bool:
        """
        Datasets must be downloaded and maybe unzipped manually. Checks, if the folder exists, where the dataset is
        expected.
        """
        if os.path.isdir(self._dataflow_builder.get_workdir()):
            return True
        return False

    @staticmethod
    def is_built_in() -> bool:
        """
        Returns flag to indicate if dataset is custom or built int.
        """
        return False


class _BuiltInDataset(DatasetBase, ABC):
    """
    Dataclass for built-in dataset. Do not use this it
    """

    _name: Optional[str] = None

    @staticmethod
    def is_built_in() -> bool:
        """
        Overwritten from base class
        """
        return True
