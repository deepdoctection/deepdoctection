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

|    xfund
|    ├── de_train
|    │ ├── de_train_0.jpg
|    │ ├── de_train_1.jpg
|    ├── de.train.json
|    ├── de_val
|    │ ├── de_val_0.jpg
|    ├── es_train
"""

import json
import os
from typing import Dict, List, Union

from ...dataflow import CustomDataFromList, DataFlow, MapData  # type: ignore
from ...datasets.info import DatasetInfo
from ...mapper.cats import cat_to_sub_cat
from ...mapper.xfundstruct import xfund_to_image
from ...utils.detection_types import JsonDict
from ...utils.settings import names
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories

_NAME = "xfund"
_DESCRIPTION = (
    "XFUND is a multilingual form understanding benchmark dataset that includes human-labeled forms with "
    "key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese)."
)
_LICENSE = (
    "The content of this project itself is licensed under the Attribution-NonCommercial-ShareAlike 4.0 "
    "International (CC BY-NC-SA 4.0) Portions of the source code are based on the transformers project. "
    "Microsoft Open Source Code of Conduct"
)
_URL = "https://github.com/doc-analysis/XFUND/releases/tag/v1.0"
_SPLITS = {"train": "train", "val": "val"}
_LOCATION = "/xfund"
_ANNOTATION_FILES: Dict[str, Union[str, List[str]]] = {
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
_INIT_CATEGORIES = [names.C.WORD]
_SUB_CATEGORIES: Dict[str, Dict[str, List[str]]]
_SUB_CATEGORIES = {names.C.WORD: {names.C.SE: [names.C.O, names.C.Q, names.C.A, names.C.HEAD]}}

_LANGUAGES = ["de", "es", "fr", "it", "ja", "pt", "zh"]


class Xfund(_BuiltInDataset):
    """
    Xfund
    """

    _name = _NAME

    def _info(self) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> "XfundBuilder":
        return XfundBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class XfundBuilder(DataFlowBaseBuilder):
    """
    Xfund dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        :param split: Split of the dataset. "train" and "val" is available
        :param load_image: Will load the image for each datapoint.  Default: False
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None
        :param languages: Will select only samples of selected languages. Available languages: de, es, fr, it, ja , pt,
                          zh. If default will take any language.
        :return: Dataflow
        """

        split = str(kwargs.get("split", "val"))
        load_image = kwargs.get("load_image", False)
        max_datapoints = kwargs.get("max_datapoints")
        language = kwargs.get("languages")

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)

        if language is None:
            languages = _LANGUAGES
        else:
            languages = [language]  # type: ignore

        if not all(elem in _LANGUAGES for elem in languages):
            raise ValueError("Not all languages available")

        # Load
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
        def replace_filename(dp: JsonDict) -> JsonDict:
            folder = "_".join(dp["id"].split("_", 2)[:2])
            dp["img"]["fname"] = os.path.join(self.get_workdir(), folder, dp["img"]["fname"])
            return dp

        df = MapData(df, replace_filename)
        category_names_mapping = {
            "other": names.C.O,
            "question": names.C.Q,
            "answer": names.C.A,
            "header": names.C.HEAD,
        }
        df = MapData(
            df, xfund_to_image(load_image, False, category_names_mapping)  # type: ignore # pylint: disable=E1120
        )

        if self.categories.is_cat_to_sub_cat():  # type: ignore
            df = MapData(
                df,
                cat_to_sub_cat(
                    self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat  # type: ignore
                ),
            )
        return df
