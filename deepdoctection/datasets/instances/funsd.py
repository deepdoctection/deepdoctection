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

import json
import os
from typing import Dict, List, Union

from ...dataflow import DataFlow, MapData, SerializerFiles  # type: ignore
from ...datasets.info import DatasetInfo
from ...mapper.cats import cat_to_sub_cat
from ...mapper.xfundstruct import xfund_to_image
from ...utils.detection_types import JsonDict
from ...utils.settings import names
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories

_NAME = "funsd"
_DESCRIPTION = (
    "FUNSD: Form Understanding in Noisy Scanned Documents. A dataset for Text Detection, Optical Character"
    " Recognition, Spatial Layout Analysis and Form Understanding."
)
_LICENSE = (
    "Use of the FUNSD Dataset is solely for non-commercial, research and educational purposes. The FUNSD  "
    "Dataset include annotations and images of real scanned forms. Licensee’s use of the images is governed "
    "by a copyright. Licensee is solely responsible for determining what additional licenses, clearances, "
    "consents and releases, if any, must be obtained for its use of the images. Original images are part of "
    "the dataset RVL-CDIP."
)

_URL = "https://guillaumejaume.github.io/FUNSD/download/"
_SPLITS = {"train": "training_data", "test": "testing_data"}
_LOCATION = "/funsd"
_ANNOTATION_FILES: Dict[str, Union[str, List[str]]] = {"train": "annotations", "test": "annotations"}

_INIT_CATEGORIES = [names.C.WORD]
_SUB_CATEGORIES: Dict[str, Dict[str, List[str]]]
_SUB_CATEGORIES = {
    names.C.WORD: {
        names.C.SE: [names.C.O, names.C.Q, names.C.A, names.C.HEAD],
        names.NER.TAG: [names.NER.I, names.NER.O, names.NER.B],
    }
}


class Funsd(_BuiltInDataset):
    """
    Funsd
    """

    _name = _NAME

    def _info(self) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS)

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
        path_ann_files = os.path.join(
            self.get_workdir(), self.splits[split], self.annotation_files[split]  # type: ignore
        )
        df = SerializerFiles.load(path_ann_files, ".json", max_datapoints)

        def load_json(path_ann: str) -> JsonDict:
            with open(path_ann, "r", encoding="utf-8") as file:
                anns = json.loads(file.read())
                path, file_name = os.path.split(path_ann)
                base_path, _ = os.path.split(path)
                path = os.path.join(base_path, "images")
                anns["file_name"] = os.path.join(path, file_name[:-4] + "png")
            return anns

        df = MapData(df, load_json)

        # Map
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
