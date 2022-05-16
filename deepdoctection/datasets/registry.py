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
import catalogue
from tabulate import tabulate
from termcolor import colored

from deepdoctection.datasets import instances
from deepdoctection.datasets.base import DatasetBase

__all__ = ["dataset_registry"]


dataset_registry = catalogue.create("deepdoctection", "datasets", entry_points=True)


def get_dataset(name: str) -> DatasetBase:
    """
    Returns an instance of a dataset with a given name.

    :param name: A dataset name
    :return: An instance of a dataset
    """
    return dataset_registry.get(name)()


def print_datasets_infos():
    data = dataset_registry.get_all()
    num_columns = min(6, len(data))
    infos = []
    for dataset in data.items():
        infos.append((dataset[0], dataset[1]._info().license, dataset[1]._info().description))
    table = tabulate(
        infos, headers=["dataset", "license", "description"] * (num_columns // 2), tablefmt="fancy_grid",
        stralign="left", numalign="left"
    )
    print(colored(table, "cyan"))





