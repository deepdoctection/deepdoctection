# -*- coding: utf-8 -*-
# File: adapter.py

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
Wrapping datasets into a PyTorch dataset framework
"""


from typing import Any, Callable, Iterator, Mapping, Optional, Union

from lazy_imports import try_import

from ..dataflow import CustomDataFromList, MapData, RepeatedData
from ..datapoint.image import Image
from ..datasets.base import DatasetBase
from ..mapper.maputils import LabelSummarizer
from ..utils.logger import LoggingRecord, log_once, logger
from ..utils.settings import DatasetType, LayoutType, ObjectTypes, PageType, WordType
from ..utils.tqdm import get_tqdm
from ..utils.types import DP, JsonDict
from .registry import get_dataset

with try_import() as import_guard:
    from torch.utils.data import IterableDataset
if not import_guard.is_successful():
    from ..utils.mocks import IterableDataset  # type: ignore


class DatasetAdapter(IterableDataset):  # type: ignore
    """
    A helper class derived from `torch.utils.data.IterableDataset` to process datasets within
    pytorch frameworks (e.g. Detectron2). It wraps the dataset and defines the compulsory
    `__iter__` using  `dataflow.build` .

    `DatasetAdapter` is meant for training and will therefore produce an infinite number of datapoints
    by shuffling and restart iteration once the previous dataflow is exhausted.
    """

    def __init__(
        self,
        name_or_dataset: Union[str, DatasetBase],
        cache_dataset: bool,
        image_to_framework_func: Optional[Callable[[DP], Optional[JsonDict]]] = None,
        use_token_tag: bool = True,
        number_repetitions: int = -1,
        **build_kwargs: str,
    ) -> None:
        """
        Args:
            name_or_dataset: Registered name of the dataset or an instance.
            cache_dataset: If set to `True`, it will cache the dataset (without loading images). If possible,
                           some statistics, e.g. number of specific labels will be printed.
            image_to_framework_func: A mapping function that converts image datapoints into the framework format
            use_token_tag: Will only be used for dataset_type="token_classification". If `use_token_tag=True`, will use
                           labels from sub category `WordType.token_tag` (with `B,I,O` suffix), otherwise
                           `WordType.token_class`.
            build_kwargs: optional parameters for defining the dataflow.
        """
        if number_repetitions == -1 and not cache_dataset:
            raise ValueError(
                "Number of repetitions cannot be infinite when not caching the dataset. Instead try to"
                " set a high number of repetitions"
            )

        if isinstance(name_or_dataset, str):
            self.dataset = get_dataset(name_or_dataset)
        else:
            self.dataset = name_or_dataset

        df = self.dataset.dataflow.build(**build_kwargs)

        if cache_dataset:
            logger.info(LoggingRecord("Yielding dataflow into memory and create torch dataset"))
            categories: Mapping[int, ObjectTypes] = {}
            _data_statistics = True
            if self.dataset.dataset_info.type in (DatasetType.OBJECT_DETECTION, DatasetType.SEQUENCE_CLASSIFICATION):
                categories = self.dataset.dataflow.categories.get_categories(filtered=True)
            elif self.dataset.dataset_info.type in (DatasetType.TOKEN_CLASSIFICATION,):
                if use_token_tag:
                    categories = self.dataset.dataflow.categories.get_sub_categories(
                        categories=LayoutType.WORD,
                        sub_categories={LayoutType.WORD: [WordType.TOKEN_TAG]},
                        keys=False,
                        values_as_dict=True,
                    )[LayoutType.WORD][WordType.TOKEN_TAG]
                else:
                    categories = self.dataset.dataflow.categories.get_sub_categories(
                        categories=LayoutType.WORD, sub_categories={LayoutType.WORD: [WordType.TOKEN_CLASS]}, keys=False
                    )[LayoutType.WORD][WordType.TOKEN_CLASS]
            else:
                logger.info(
                    LoggingRecord(f"dataset is of type {self.dataset.dataset_info.type}. Cannot generate statistics.")
                )
                _data_statistics = False

            if _data_statistics:
                summarizer = LabelSummarizer(categories)

            df.reset_state()
            max_datapoints_str = build_kwargs.get("max_datapoints")

            if max_datapoints_str is not None:
                max_datapoints = int(max_datapoints_str)
            else:
                max_datapoints = None

            datapoints = []
            with get_tqdm(total=max_datapoints) as status_bar:
                for dp in df:
                    if dp.image is not None:
                        log_once(
                            "Datapoint have images as np arrays stored and they will be loaded into memory. "
                            "To avoid OOM set 'load_image'=False in dataflow build config. This will load "
                            "images when needed and reduce memory costs!!!",
                            "warn",
                        )
                    if self.dataset.dataset_info.type == DatasetType.OBJECT_DETECTION:
                        anns = dp.get_annotation()
                        cat_ids = [ann.category_id for ann in anns]

                    elif self.dataset.dataset_info.type == DatasetType.SEQUENCE_CLASSIFICATION:
                        cat_ids = dp.summary.get_sub_category(PageType.DOCUMENT_TYPE).category_id

                    elif self.dataset.dataset_info.type == DatasetType.TOKEN_CLASSIFICATION:
                        anns = dp.get_annotation(category_names=LayoutType.WORD)
                        if use_token_tag:
                            cat_ids = [ann.get_sub_category(WordType.TOKEN_TAG).category_id for ann in anns]
                        else:
                            cat_ids = [ann.get_sub_category(WordType.TOKEN_CLASS).category_id for ann in anns]

                    if _data_statistics:
                        summarizer.dump(cat_ids)

                    datapoints.append(dp)
                    status_bar.update()

            if _data_statistics:
                summarizer.print_summary_histogram()
            self.number_datapoints = len(datapoints)
            if not self.number_datapoints:
                raise ValueError("DatasetAdapter receives no datapoints. Please check your dataflow build config.")

            df = CustomDataFromList(datapoints, shuffle=True)
            df = RepeatedData(df, number_repetitions)
        else:
            df = RepeatedData(df, number_repetitions)

        if image_to_framework_func:
            df = MapData(df, image_to_framework_func)

        self.df = df
        self.df.reset_state()

    def __iter__(self) -> Iterator[Image]:
        return iter(self.df)

    def __len__(self) -> int:
        if self.number_datapoints:
            return self.number_datapoints
        return len(self.df)

    def __getitem__(self, item: Any) -> None:
        raise NotImplementedError()
