# -*- coding: utf-8 -*-
# File: funsd.py

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
Module for Funsd dataset.  Install the dataset following the folder structure

|    funsd
|    ├── testing_data
|    │ ├── annotations
|    │ │ ├── 82092117.json
|    │ │ ├── 82092117_0338.json
|    │ ├── images
|    │ │ ├── 82092117.png
|    │ │ ├── 82092117_0338.png
|    ├── training_data
|    │ ├── annotations
|    │ │ ├── ...
|    │ ├── images
|    │ │ ├── ...
"""

import os
from typing import Dict, List, Mapping, Union

from ...dataflow import DataFlow, MapData, SerializerFiles
from ...datasets.info import DatasetInfo
from ...mapper.cats import cat_to_sub_cat
from ...mapper.xfundstruct import xfund_to_image
from ...utils.detection_types import JsonDict, Pathlike
from ...utils.fs import load_json
from ...utils.settings import BioTag, DatasetType, LayoutType, ObjectTypes, TokenClasses, TokenClassWithTag, WordType
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories
from ..registry import dataset_registry


def load_file(path_ann: Pathlike) -> JsonDict:
    """
    Loading json file

    :param path_ann: path
    :return: dict
    """
    anns = load_json(path_ann)
    path, file_name = os.path.split(path_ann)
    base_path, _ = os.path.split(path)
    path = os.path.join(base_path, "images")
    anns["file_name"] = os.path.join(path, file_name[:-4] + "png")
    return anns


_NAME = "funsd"
_DESCRIPTION = (
    "FUNSD: Form Understanding in Noisy Scanned Documents. A dataset for Text Detection, Optical Character \n"
    " Recognition, Spatial Layout Analysis and Form Understanding."
)
_LICENSE = (
    "Use of the FUNSD Dataset is solely for non-commercial, research and educational purposes. The FUNSD  \n"
    "Dataset include annotations and images of real scanned forms. Licensee’s use of the images is governed \n"
    "by a copyright. Licensee is solely responsible for determining what additional licenses, clearances, \n"
    "consents and releases, if any, must be obtained for its use of the images. Original images are part of \n"
    "the dataset RVL-CDIP."
)

_URL = "https://guillaumejaume.github.io/FUNSD/download/"
_SPLITS: Mapping[str, str] = {"train": "training_data", "test": "testing_data"}
_TYPE = DatasetType.token_classification
_LOCATION = "funsd"
_ANNOTATION_FILES: Mapping[str, str] = {"train": "annotations", "test": "annotations"}

_INIT_CATEGORIES = [LayoutType.word]
_SUB_CATEGORIES: Dict[ObjectTypes, Dict[ObjectTypes, List[ObjectTypes]]]
_SUB_CATEGORIES = {
    LayoutType.word: {
        WordType.token_class: [TokenClasses.other, TokenClasses.question, TokenClasses.answer, TokenClasses.header],
        WordType.tag: [BioTag.inside, BioTag.outside, BioTag.begin],
        WordType.token_tag: [
            TokenClassWithTag.b_answer,
            TokenClassWithTag.b_header,
            TokenClassWithTag.b_question,
            TokenClassWithTag.i_answer,
            TokenClassWithTag.i_header,
            TokenClassWithTag.i_question,
            BioTag.outside,
        ],
    }
}


@dataset_registry.register("funsd")
class Funsd(_BuiltInDataset):
    """
    Funsd
    """

    _name = _NAME

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS, type=_TYPE)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> "FunsdBuilder":
        return FunsdBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class FunsdBuilder(DataFlowBaseBuilder):
    """
    Funsd dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        :param split: Split of the dataset. "train" and "test" is available
        :param load_image: Will load the image for each datapoint.  Default: False
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None

        :return: Dataflow
        """

        split = str(kwargs.get("split", "test"))
        load_image = kwargs.get("load_image", False)
        max_datapoints = kwargs.get("max_datapoints")

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)

        # Load
        path_ann_files = self.get_workdir() / self.splits[split] / self.get_annotation_file(split)

        df = SerializerFiles.load(path_ann_files, ".json", max_datapoints)

        df = MapData(df, load_file)

        # Map
        category_names_mapping = {
            "other": TokenClasses.other,
            "question": TokenClasses.question,
            "answer": TokenClasses.answer,
            "header": TokenClasses.header,
        }
        ner_token_to_id_mapping = self.categories.get_sub_categories(
            categories=LayoutType.word,
            sub_categories={LayoutType.word: [WordType.token_tag]},
            keys=False,
            values_as_dict=True,
            name_as_key=True,
        )[LayoutType.word][WordType.token_tag]
        df = MapData(df, xfund_to_image(load_image, False, category_names_mapping, ner_token_to_id_mapping))
        if self.categories.is_cat_to_sub_cat():
            df = MapData(
                df,
                cat_to_sub_cat(self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat),
            )
        return df
