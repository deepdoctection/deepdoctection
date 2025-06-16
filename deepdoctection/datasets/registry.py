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
`DatasetRegistry` for registering built-in and custom datasets
"""

import inspect

import catalogue  # type: ignore
from tabulate import tabulate
from termcolor import colored

from .base import CustomDataset, DatasetBase

__all__ = ["dataset_registry", "get_dataset", "print_dataset_infos"]


dataset_registry = catalogue.create("deepdoctection", "datasets", entry_points=True)


def get_dataset(name: str) -> DatasetBase:
    """
    Returns an instance of a dataset with a given name. This instance can be used to customize the dataflow output

    Example:

        ```python
        dataset = get_dataset("some_name")
        dataset.dataflow.categories.filter_categories(["cat1","cat2"])
        df = dataset.dataflow.build(split="train")

        for dp in df:
            # do something
        ```

    Args:
        name: A dataset name

    Returns:
        An instance of a dataset
    """
    ds = dataset_registry.get(name)
    if inspect.isclass(ds):
        return ds()
    return ds


def print_dataset_infos(add_license: bool = True, add_info: bool = True) -> None:
    """
    Prints a table with all registered datasets and some basic information (name, license and optionally description)

    Args:
        add_license: Whether to add the license type of the dataset
        add_info: Whether to add a description of the dataset
    """

    data = dataset_registry.get_all()
    num_columns = min(6, len(data))
    infos = []

    for dataset in data.items():
        info = [dataset[0]]
        if isinstance(dataset[1], CustomDataset):
            ds = dataset[1]
        else:
            ds = dataset[1]()
        info.append(ds.dataset_info.type)
        if add_license:
            info.append(ds.dataset_info.license)  # pylint: disable=W0212
        if add_info:
            info.append(ds.dataset_info.short_description)  # pylint: disable=W0212
        if ds.dataflow.categories is not None:  # pylint: disable=W0212
            categories = "\n".join(ds.dataflow.categories.init_categories)  # Format categories as multi-line string
            sub_categories = "\n".join(
                f"{key}: {', '.join(values)}" for key, values in ds.dataflow.categories.init_sub_categories.items()
            )  # Format sub-categories as multi-line string
            info.append(categories)
            info.append(sub_categories)
        else:
            info.append("")
            info.append("")
        infos.append(info)

    header = ["dataset", "type"]
    if add_license:
        header.append("license")
    if add_info:
        header.append("description")
    header.append("categories")
    table = tabulate(
        infos, headers=header * (num_columns // 2), tablefmt="fancy_grid", stralign="left", numalign="left"
    )
    print(colored(table, "cyan"))
