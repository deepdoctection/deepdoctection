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
Module for DatasetRegistry
"""

import inspect
from typing import Dict, List, Type

from deepdoctection.datasets import instances
from deepdoctection.datasets.base import DatasetBase

__all__ = ["DatasetRegistry"]


_DATASETS: Dict[str, Type[DatasetBase]] = dict(
    (m[1]._name, m[1])  # pylint: disable=W0212
    for m in inspect.getmembers(instances, inspect.isclass)
    if issubclass(m[1], DatasetBase) and m[0] != "DatasetBase"
)


class DatasetRegistry:
    """
    The DatasetRegistry is the object for receiving datasets and registering new ones. Use the registry to
    easily generate instances of a dataset.
    """

    @staticmethod
    def print_dataset_names() -> None:
        """
        Print a list of registered dataset names
        """
        print(list(_DATASETS.keys()))

    @staticmethod
    def get_dataset(name: str) -> DatasetBase:
        """
        Returns an instance of a dataset with a given name.

        :param name: A dataset name
        :return: An instance of a dataset
        """
        return _DATASETS[name]()

    @staticmethod
    def register_dataset(name: str, dataset_class: Type[DatasetBase]) -> None:
        """
        Register a new dataset.

        :param name: A dataset name
        :param dataset_class: A new dataset class to add to the registry.
        """
        _DATASETS[name] = dataset_class

    @staticmethod
    def get_dataset_names() -> List[str]:
        """
        Get a list of available dataset names. This will not print the names.

        :return: A list of dataset names.
        """
        return list(_DATASETS.keys())
