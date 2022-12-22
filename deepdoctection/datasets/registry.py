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

import catalogue  # type: ignore
from tabulate import tabulate
from termcolor import colored

from .base import DatasetBase

__all__ = ["dataset_registry", "get_dataset", "print_dataset_infos"]


dataset_registry = catalogue.create("deepdoctection", "datasets", entry_points=True)


def get_dataset(name: str) -> DatasetBase:
    """
    Returns an instance of a dataset with a given name. This instance can be used to customize the dataflow output

    **Example:**

            dataset = get_dataset("some_name")
            dataset.dataflow.categories.filter_categories(["cat1","cat2"])
            df = dataset.dataflow.build(split="train")

            for dp in df:
                # do something

    :param name: A dataset name
    :return: An instance of a dataset
    """
    return dataset_registry.get(name)()


def print_dataset_infos(add_license: bool = True, add_info: bool = True) -> None:
    """
    Prints a table with all registered datasets and some basic information (name, license and optionally description)

    :param add_license: Whether to add the license type of the dataset
    :param add_info: Whether to add a description of the dataset
    """

    data = dataset_registry.get_all()
    num_columns = min(6, len(data))
    infos = []
    for dataset in data.items():
        info = [dataset[0]]
        if add_license:
            info.append(dataset[1]._info().license)  # pylint: disable=W0212
        if add_info:
            info.append(dataset[1]._info().description)  # pylint: disable=W0212
        infos.append(info)
    header = ["dataset"]
    if add_license:
        header.append("license")
    if add_info:
        header.append("description")
    table = tabulate(
        infos, headers=header * (num_columns // 2), tablefmt="fancy_grid", stralign="left", numalign="left"
    )
    print(colored(table, "cyan"))
