# -*- coding: utf-8 -*-
# File: xfund.py

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
Module for XFUND dataset. Install the dataset following the folder structure

    xfund
    ├── de_train
    │ ├── de_train_0.jpg
    │ ├── de_train_1.jpg
    ├── de.train.json
    ├── de_val
    │ ├── de_val_0.jpg
    ├── es_train
"""
from __future__ import annotations

import json
import os
from typing import Mapping, Sequence, Union

from ...dataflow import CustomDataFromList, DataFlow, MapData
from ...datasets.info import DatasetInfo
from ...mapper.cats import cat_to_sub_cat, filter_cat
from ...mapper.xfundstruct import xfund_to_image
from ...utils.settings import BioTag, DatasetType, LayoutType, ObjectTypes, TokenClasses, TokenClassWithTag, WordType
from ...utils.types import FunsdDict
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories
from ..registry import dataset_registry

_NAME = "xfund"
_SHORT_DESCRIPTION = "XFUND is a multilingual form understanding benchmark dataset."
_DESCRIPTION = (
    "XFUND is a multilingual form understanding benchmark dataset that includes human-labeled forms with \n"
    "key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese)."
)
_LICENSE = (
    "The content of this project itself is licensed under the Attribution-NonCommercial-ShareAlike 4.0 \n"
    "International (CC BY-NC-SA 4.0) Portions of the source code are based on the transformers project. \n"
    "Microsoft Open Source Code of Conduct"
)
_URL = "https://github.com/doc-analysis/XFUND/releases/tag/v1.0"
_SPLITS: Mapping[str, str] = {"train": "train", "val": "val"}
_TYPE = DatasetType.TOKEN_CLASSIFICATION
_LOCATION = "xfund"
_ANNOTATION_FILES: Mapping[str, Union[str, Sequence[str]]] = {
    "train": [
        "de.train.json",
        "es.train.json",
        "fr.train.json",
        "it.train.json",
        "ja.train.json",
        "pt.train.json",
        "zh.train.json",
    ],
    "val": ["de.val.json", "es.val.json", "fr.val.json", "it.val.json", "ja.val.json", "pt.val.json", "zh.val.json"],
}
_INIT_CATEGORIES: Sequence[ObjectTypes] = [LayoutType.WORD, LayoutType.TEXT]
_SUB_CATEGORIES: Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]] = {
    LayoutType.WORD: {
        WordType.TOKEN_CLASS: [TokenClasses.OTHER, TokenClasses.QUESTION, TokenClasses.ANSWER, TokenClasses.HEADER],
        WordType.TAG: [BioTag.INSIDE, BioTag.OUTSIDE, BioTag.BEGIN],
        WordType.TOKEN_TAG: [
            TokenClassWithTag.B_ANSWER,
            TokenClassWithTag.B_HEADER,
            TokenClassWithTag.B_QUESTION,
            TokenClassWithTag.I_ANSWER,
            TokenClassWithTag.I_HEADER,
            TokenClassWithTag.I_QUESTION,
            BioTag.OUTSIDE,
        ],
    },
    LayoutType.TEXT: {
        WordType.TOKEN_CLASS: [TokenClasses.OTHER, TokenClasses.QUESTION, TokenClasses.ANSWER, TokenClasses.HEADER]
    },
}

_LANGUAGES = ["de", "es", "fr", "it", "ja", "pt", "zh"]


@dataset_registry.register("xfund")
class Xfund(_BuiltInDataset):
    """
    Xfund
    """

    _name = _NAME

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(
            name=_NAME,
            short_description=_SHORT_DESCRIPTION,
            description=_DESCRIPTION,
            license=_LICENSE,
            url=_URL,
            splits=_SPLITS,
            type=_TYPE,
        )

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> XfundBuilder:
        return XfundBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class XfundBuilder(DataFlowBaseBuilder):
    """
    Xfund dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        Args:
            kwargs:
                (split) Split of the dataset. `train` and `val` are available. Default: `val`
                (load_image) Will load the image for each datapoint. Default: `False`
                (max_datapoints) Will stop iterating after `max_datapoints`. Default: `None`
                (languages) Will select only samples of selected languages. Available languages:
                            `de`, `es`, `fr`, `it`, `ja`, `pt`, `zh`. If `None`, all
                            languages are taken. Default: `None`

        Returns:
            Dataflow
        """

        split = str(kwargs.get("split", "val"))
        load_image = kwargs.get("load_image", False)
        max_datapoints = kwargs.get("max_datapoints")
        language = kwargs.get("languages")

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)

        if language is None:
            languages = _LANGUAGES
        elif isinstance(language, str):
            languages = [language]
        else:
            raise TypeError("language requires to be a string")

        if not all(elem in _LANGUAGES for elem in languages):
            raise ValueError("Not all languages available")

        # Load
        df: DataFlow
        path_ann_files = [
            os.path.join(self.get_workdir(), ann_file)
            for ann_file in self.annotation_files[split]
            if ann_file.split(".")[0] in languages
        ]

        datapoints = []
        for path_ann in path_ann_files:
            with open(path_ann, "r", encoding="utf-8") as file:
                anns = json.loads(file.read())
                datapoints.extend(anns["documents"])
        df = CustomDataFromList(datapoints, max_datapoints=max_datapoints)

        # Map
        def replace_filename(dp: FunsdDict) -> FunsdDict:
            folder = "_".join(dp["id"].split("_", 2)[:2])
            dp["img"]["fname"] = os.path.join(self.get_workdir(), folder, dp["img"]["fname"])
            return dp

        df = MapData(df, replace_filename)
        categories_name_as_key = self.categories.get_categories(init=True, name_as_key=True)
        token_class_names_mapping = {
            "other": TokenClasses.OTHER,
            "question": TokenClasses.QUESTION,
            "answer": TokenClasses.ANSWER,
            "header": TokenClasses.HEADER,
        }
        if LayoutType.WORD in self.categories.get_categories(filtered=True, name_as_key=True):
            ner_token_to_id_mapping = self.categories.get_sub_categories(
                categories=LayoutType.WORD,
                sub_categories={LayoutType.WORD: [WordType.TOKEN_TAG, WordType.TAG, WordType.TOKEN_CLASS]},
                keys=False,
                values_as_dict=True,
                name_as_key=True,
            )
        else:
            ner_token_to_id_mapping = {
                LayoutType.WORD: {
                    WordType.TAG: {BioTag.BEGIN: 3, BioTag.INSIDE: 1, BioTag.OUTSIDE: 2},
                    WordType.TOKEN_CLASS: {
                        TokenClasses.ANSWER: 3,
                        TokenClasses.HEADER: 4,
                        TokenClasses.OTHER: 1,
                        TokenClasses.QUESTION: 2,
                    },
                    WordType.TOKEN_TAG: {
                        TokenClassWithTag.B_ANSWER: 1,
                        TokenClassWithTag.B_HEADER: 2,
                        TokenClassWithTag.B_QUESTION: 3,
                        TokenClassWithTag.I_ANSWER: 4,
                        TokenClassWithTag.I_HEADER: 5,
                        TokenClassWithTag.I_QUESTION: 6,
                        BioTag.OUTSIDE: 7,
                    },
                }
            }
        df = MapData(
            df,
            xfund_to_image(
                load_image, False, categories_name_as_key, token_class_names_mapping, ner_token_to_id_mapping
            ),
        )
        if self.categories.is_filtered():
            df = MapData(
                df,
                filter_cat(  # pylint: disable=E1120
                    self.categories.get_categories(as_dict=False, filtered=True),
                    self.categories.get_categories(as_dict=False, filtered=False),
                ),
            )

        if self.categories.is_cat_to_sub_cat():
            df = MapData(
                df,
                cat_to_sub_cat(self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat),
            )
        return df
