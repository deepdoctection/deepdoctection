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
from __future__ import annotations

import json
import os
import pprint
from abc import ABC, abstractmethod
from collections import defaultdict
from inspect import signature
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type, Union

import numpy as np

from ..dataflow import CacheData, ConcatData, CustomDataFromList, DataFlow
from ..datapoint.image import Image
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import DatasetType, ObjectTypes, TypeOrStr, get_type
from ..utils.types import PathLikeOrStr
from .dataflow_builder import DataFlowBaseBuilder
from .info import DatasetCategories, DatasetInfo, get_merged_categories


class DatasetBase(ABC):
    """
    Base class for a dataset. Requires to implementing `_categories` `_info` and `_builder` by
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
            logger.warning(
                LoggingRecord(
                    f"Dataset {self._dataset_info.name} not locally found. Please download at {self._dataset_info.url}"
                    f" and place under {self._dataflow_builder.get_workdir()}"
                )
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

        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _info(cls) -> DatasetInfo:
        """
        Construct the DatasetInfo object.
        """

        raise NotImplementedError()

    @abstractmethod
    def _builder(self) -> DataFlowBaseBuilder:
        """
        Construct the DataFlowBaseBuilder object. It needs to be implemented in the derived class.
        """

        raise NotImplementedError()

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
    Dataclass for built-in dataset. Do not use this
    """

    _name: Optional[str] = None

    @staticmethod
    def is_built_in() -> bool:
        """
        Overwritten from base class
        """
        return True


class SplitDataFlow(DataFlowBaseBuilder):
    """
    Dataflow builder for splitting datasets
    """

    def __init__(self, train: list[Image], val: list[Image], test: Optional[list[Image]]):
        """
        :param train: Cached train split
        :param val: Cached val split
        :param test: Cached test split
        """
        super().__init__(location="")
        self.split_cache: dict[str, list[Image]]
        if test is None:
            self.split_cache = {"train": train, "val": val}
        else:
            self.split_cache = {"train": train, "val": val, "test": test}

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Dataflow builder for merged split datasets
        :param kwargs: Only split and max_datapoints arguments will be considered.
        :return: Dataflow
        """

        split = kwargs.get("split", "train")
        if not isinstance(split, str):
            raise ValueError("'split' must be a string")
        max_datapoints = kwargs.get("max_datapoints")
        if isinstance(max_datapoints, str):
            max_datapoints = int(max_datapoints)

        return CustomDataFromList(self.split_cache[split], max_datapoints=max_datapoints)


class MergeDataset(DatasetBase):
    """
    A class for merging dataset ready to feed a training or an evaluation script. The dataflow builder will generate
    samples from all datasets and will exhaust if every dataflow of the merged datasets are exhausted as well. To
    guarantee flexibility it is possible to pass customized dataflows explicitly to maybe reduce the dataflow size from
    one dataset or to use different splits from different datasets.

    When yielding datapoint from :build(), note that one dataset will pass all its samples successively which
    might reduce randomness for training, especially when using datasets from the same domain. Buffering all datasets
    (without loading heavy components like images) is therefore possible and the merged dataset can be shuffled.

    When the datasets are buffered are split functionality can divide the buffered samples into an train, val and test
    set.

    While the selection of categories is given by the union of all categories of all datasets, sub categories need to
    be handled with care: Only sub categories for one specific category are available provided that every dataset has
    this sub category available for this specific category. The range of sub category values again is defined as the
    range of all values from all datasets.

    **Example:**

            dataset_1 = get_dataset("dataset_1")
            dataset_2 = get_dataset("dataset_2")

            union_dataset = MergeDataset(dataset_1,dataset_2)
            union_dataset.buffer_datasets(split="train")     # will cache the train split of dataset_1 and dataset_2
            merge.split_datasets(ratio=0.1, add_test=False)  # will create a new split of the union.


    **Example:**

            dataset_1 = get_dataset("dataset_1")
            dataset_2 = get_dataset("dataset_2")

            df_1 = dataset_1.dataflow.build(max_datapoints=20)  # handle separate dataflow configs ...
            df_2 = dataset_1.dataflow.build(max_datapoints=30)

            union_dataset = MergeDataset(dataset_1,dataset_2)
            union_dataset.explicit_dataflows(df_1,df_2)   # ... and pass them explicitly. Filtering is another
                                                          # possibility
    """

    def __init__(self, *datasets: DatasetBase):
        """
        :param datasets: An arbitrary number of datasets
        """
        self.datasets = datasets
        self.dataflows: Optional[tuple[DataFlow, ...]] = None
        self.datapoint_list: Optional[list[Image]] = None
        super().__init__()
        self._dataset_info.type = datasets[0].dataset_info.type
        self._dataset_info.name = "merge_" + "_".join([dataset.dataset_info.name for dataset in self.datasets])

    def _categories(self) -> DatasetCategories:
        return get_merged_categories(
            *(dataset.dataflow.categories for dataset in self.datasets if dataset.dataflow.categories is not None)
        )

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(name="merge")

    def _builder(self) -> DataFlowBaseBuilder:
        class MergeDataFlow(DataFlowBaseBuilder):
            """
            Dataflow builder for merged datasets
            """

            def __init__(self, *dataflow_builders: DataFlowBaseBuilder):
                super().__init__("")
                self.dataflow_builders = dataflow_builders
                self.dataflows: Optional[tuple[DataFlow, ...]] = None

            def build(self, **kwargs: Union[str, int]) -> DataFlow:
                """
                Building the dataflow of merged datasets. No argument will affect the stream if the dataflows have
                been explicitly passed. Otherwise, all kwargs will be passed to all dataflows. Note that each dataflow
                will iterate until it is exhausted. To guarantee randomness across different datasets cache all
                datapoints and shuffle them afterwards (e.g. use :buffer_dataset() ).

                :param kwargs: arguments for :build()
                :return: Dataflow
                """
                df_list = []
                if self.dataflows is not None:
                    logger.info(LoggingRecord("Will used dataflow from previously explicitly passed configuration"))
                    return ConcatData(list(self.dataflows))

                logger.info(LoggingRecord("Will use the same build setting for all dataflows"))
                for dataflow_builder in self.dataflow_builders:
                    df_list.append(dataflow_builder.build(**kwargs))
                df = ConcatData(df_list)
                return df

        builder = MergeDataFlow(*(dataset.dataflow for dataset in self.datasets))
        if self.dataflows is not None:
            builder.dataflows = self.dataflows
        return builder

    def explicit_dataflows(self, *dataflows: DataFlow) -> None:
        """
        Pass explicit dataflows for each dataset. Using several dataflow configurations for one dataset is possible as
        well. However, the number of dataflow must exceed the number of merged datasets.

        :param dataflows: An arbitrary number of dataflows
        """
        self.dataflows = dataflows
        if len(self.datasets) > len(self.dataflows):
            raise ValueError(
                f"len(self.datasets) = {len(self.datasets)} must be" f" <= len(self.dataflows) = {len(self.dataflows)}"
            )
        self._dataflow_builder = self._builder()
        self._dataflow_builder.categories = self._categories()

    def buffer_datasets(self, **kwargs: Union[str, int]) -> None:
        """
        Buffer datasets with given configs. If dataflows are passed explicitly it will cache their streamed output.

        :param kwargs: arguments for :build()
        :return: Dataflow
        """
        df = self.dataflow.build(**kwargs)
        self.datapoint_list = CacheData(df, shuffle=True).get_cache()

    def split_datasets(self, ratio: float = 0.1, add_test: bool = True) -> None:
        """
        Split cached datasets into train/val(/test).

        :param ratio: 1-ratio will be assigned to the train split. The remaining bit will be assigned to val and test
                      split.
        :param add_test: Add a test split
        """
        assert self.datapoint_list is not None, "Datasets need to be buffered before splitting"
        number_datapoints = len(self.datapoint_list)
        indices = np.random.binomial(1, ratio, number_datapoints)
        train_dataset = [self.datapoint_list[i] for i in range(number_datapoints) if indices[i] == 0]
        val_dataset = [self.datapoint_list[i] for i in range(number_datapoints) if indices[i] == 1]
        test_dataset = None

        if add_test:
            test_dataset = [dp for id, dp in enumerate(val_dataset) if id % 2]
            val_dataset = [dp for id, dp in enumerate(val_dataset) if not id % 2]

        logger.info(LoggingRecord("___________________ Number of datapoints per split ___________________"))
        logger.info(
            pprint.pformat(
                {
                    "train": len(train_dataset),
                    "val": len(val_dataset),
                    "test": len(test_dataset) if test_dataset is not None else 0,
                },
                width=100,
                compact=True,
            )
        )

        self._dataflow_builder = SplitDataFlow(train_dataset, val_dataset, test_dataset)
        self._dataflow_builder.categories = self._categories()

    def get_ids_by_split(self) -> dict[str, list[str]]:
        """
        To reproduce a dataset split at a later stage, get a summary of the by having a dict of list with split and
        the image ids contained in the split.

        :return: E.g. `{"train": ['ab','ac'],"val":['bc','bd']}`
        """
        if isinstance(self._dataflow_builder, SplitDataFlow):
            return {
                key: [img.image_id for img in self._dataflow_builder.split_cache.get(key, [])]
                for key in ("train", "val", "test")
            }
        return {"train": [], "val": [], "test": []}

    def create_split_by_id(
        self, split_dict: Mapping[str, Sequence[str]], **dataflow_build_kwargs: Union[str, int]
    ) -> None:
        """
        Reproducing a dataset split from a dataset or a dataflow by a dict of list of image ids.

            merge = dd.MergeDataset(doclaynet)
            merge.explicit_dataflows(df_doc)
            merge.buffer_datasets()
            merge.split_datasets(ratio=0.1)
            out = merge.get_ids_by_split()   # Save out somewhere

            merge_2 = dd.MergeDataset(doclaynet)
            df_doc_2 = doclaynet.dataflow.build(split="train", max_datapoints=4000)
            merge_2.explicit_dataflows(df_doc_2)
            merge_2.create_split_by_id(out)   # merge_2 now has the same split as merge

        :param split_dict: e.g. `{"train":['ab','ac',...],"val":['bc'],"test":[]}`
        """

        if set(split_dict.keys()) != {"train", "val", "test"}:
            raise KeyError("split_dict must contain keys for 'train', 'val' and 'test'")
        ann_id_to_split = {ann_id: "train" for ann_id in split_dict["train"]}
        ann_id_to_split.update({ann_id: "val" for ann_id in split_dict["val"]})
        ann_id_to_split.update({ann_id: "test" for ann_id in split_dict["test"]})
        self.buffer_datasets(**dataflow_build_kwargs)
        split_defaultdict = defaultdict(list)
        for image in self.datapoint_list:  # type: ignore
            split_defaultdict[ann_id_to_split[image.image_id]].append(image)
        train_dataset = split_defaultdict["train"]
        val_dataset = split_defaultdict["val"]
        test_dataset = split_defaultdict["test"]
        self._dataflow_builder = SplitDataFlow(train_dataset, val_dataset, test_dataset)
        self._dataflow_builder.categories = self._categories()


class CustomDataset(DatasetBase):
    """
    A simple dataset interface that implements the boilerplate code and reduces complexity by merely leaving
    the user to write a `DataFlowBaseBuilder` (mapping the annotation format into deepdoctection data model is
    something that has to be left to the user for obvious reasons). Check the tutorial on how to approach the mapping
    problem.
    """

    def __init__(
        self,
        name: str,
        dataset_type: TypeOrStr,
        location: PathLikeOrStr,
        init_categories: Sequence[ObjectTypes],
        dataflow_builder: Type[DataFlowBaseBuilder],
        init_sub_categories: Optional[Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]]] = None,
        annotation_files: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    ):
        """
        :param name: Name of the dataset. It will not be used in the code, however it might be helpful, if several
                     custom datasets are in use.
        :param dataset_type: Datasets need to be characterized by one of the `enum` members `DatasetType` that describe
                     the machine learning task the dataset is built for. You can get all registered types with

                            types = dd.object_types_registry.get("DatasetType")
                            print({t for t in types})

        :param location: Datasets should be stored a sub folder of name `location` in the local cache
                         `get_dataset_dir_path()`. There are good reasons to use `name`.
        :param init_categories: A list of all available categories in this dataset. You must use a list as the order
                                of the categories must always be preserved: they determine the category id that in turn
                                will be used for model training.
        :param dataflow_builder: A subclass of `DataFlowBaseBuilder`. Do not instantiate the class by yourself.
        :param init_sub_categories: A dict mapping main categories to sub categories, if there are any available.
                                    Suppose an object `LayoutType.cell` has two additional information in the annotation
                                    file: `CellType.header, CellType.body`. You can then write:

                                        {LayoutType.cell: {CellType.header: [CellType.header, CellType.body]}

                                    This setting assumes that later in the mapping the `ImageAnnotation` with
                                    `category_name=LayoutType.cell` will have a sub category of key `CellType.header`
                                    and one of the two values `CellType.header, CellType.body`
        :param annotation_files: A mapping to one or more annotation files, e.g.

                                       annotation_file = {"train": "train_file.json", "test": "test_file.json"}
        """

        self.name = name
        self.type: DatasetType = get_type(dataset_type)  # type: ignore
        self.location = location
        self.init_categories = init_categories
        if init_sub_categories is None:
            self.init_sub_categories: Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]] = {}
        else:
            self.init_sub_categories = init_sub_categories
        self.annotation_files = annotation_files
        if signature(dataflow_builder.__init__).parameters.keys() != {"self", "location", "annotation_files"}:
            raise TypeError(
                "Dataflow builder must have the signature `def __init__(self, location: Pathlike, "
                "annotation_files: Optional[Mapping[str, Union[str, Sequence[str]]]] = None):`"
            )
        self.dataflow_builder = dataflow_builder(self.location, self.annotation_files)
        super().__init__()

    def _info(self) -> DatasetInfo:  # type: ignore  # pylint: disable=W0221
        return DatasetInfo(name=self.name, type=self.type, description="", license="", url="", splits={})

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=self.init_categories, init_sub_categories=self.init_sub_categories)

    def _builder(self) -> DataFlowBaseBuilder:
        return self.dataflow_builder

    @staticmethod
    def from_dataset_card(file_path: PathLikeOrStr, dataflow_builder: Type[DataFlowBaseBuilder]) -> CustomDataset:
        """
        This static method creates a CustomDataset instance from a dataset card.

        A dataset card is a JSON file that contains metadata about the dataset such as its name, type, location,
        initial categories, initial sub categories, and annotation files. The dataflow_builder parameter is a class
        that inherits from DataFlowBaseBuilder and is used to build the dataflow for the dataset.

        :param file_path: The path to the dataset card (JSON file).
        :param dataflow_builder: The class used to build the dataflow for the dataset.
        :return: A CustomDataset instance created from the dataset card.
        """

        with open(file_path, "r", encoding="UTF-8") as file:
            meta_data = json.load(file)
        meta_data["dataset_type"] = get_type(meta_data["dataset_type"])
        meta_data["location"] = Path(meta_data["location"])
        meta_data["init_categories"] = [get_type(cat) for cat in meta_data["init_categories"]]
        meta_data["init_sub_categories"] = (
            {
                get_type(cat): {
                    get_type(sub_cat_key): [get_type(sub_cat_value) for sub_cat_value in sub_cat_values]
                    for sub_cat_key, sub_cat_values in sub_cats.items()
                }
                for cat, sub_cats in meta_data["init_sub_categories"].items()
            }
            if meta_data["init_sub_categories"] is not None
            else None
        )
        return CustomDataset(**meta_data, dataflow_builder=dataflow_builder)

    def as_dict(self) -> Mapping[str, Any]:
        """
        Return the meta-data of the dataset as a dictionary.

        :return: A dictionary containing the meta-data of the dataset.
        """
        return {
            "name": self.name,
            "dataset_type": self.type,
            "location": str(self.location),
            "annotation_files": self.annotation_files,
            "init_categories": [cat.value for cat in self.init_categories],
            "init_sub_categories": {
                cat.value: {
                    sub_cat_key.value: [sub_cat_value.value for sub_cat_value in sub_cat_values]
                    for sub_cat_key, sub_cat_values in sub_cats.items()
                }
                for cat, sub_cats in self.init_sub_categories.items()
            }
            if self.init_sub_categories is not None
            else None,
        }

    def save_dataset_card(self, file_path: str) -> None:
        """
        Save the dataset card to a JSON file.

        :param file_path: file_path
        """
        with open(file_path, "w", encoding="UTF-8") as file:
            json.dump(self.as_dict(), file, indent=4)
