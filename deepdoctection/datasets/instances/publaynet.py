# -*- coding: utf-8 -*-
# File: publaynet.py

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
Module for Publaynet dataset. Place the dataset as follows

|    publaynet
|    ├── test
|    │ ├── PMC_1.png
|    ├── train
|    │ ├── PMC_2.png
|    ├── val
|    │ ├── PMC_3.png
|    ├── train.json
|    ├── val.json
"""

import os
from typing import Dict, List, Union

from ...dataflow import DataFlow, MapData, MapDataComponent  # type: ignore
from ...dataflow.custom_serialize import SerializerCoco
from ...datasets.info import DatasetInfo
from ...mapper.cats import filter_cat
from ...mapper.cocostruct import coco_to_image
from ...utils.settings import names
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories

_NAME = "publaynet"
_DESCRIPTION = (
    "PubLayNet is a dataset for document layout analysis. It contains images of research papers and "
    "articles"
    " and annotations for various elements in a page such as “text”, “list”, “figure” etc in these "
    "research"
    " paper images. The dataset was obtained by automatically matching the XML representations and the"
    " content of over 1 million PDF articles that are publicly available on PubMed Central."
)
_LICENSE = (
    "The annotations in this dataset belong to IBM and are licensed under a Community Data License Agreement"
    " – Permissive – Version 1.0 License. IBM does not own the copyright of the images. Use of the images"
    " must abide by the PMC Open Access Subset Terms of Use."
)
_URL = (
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/"
    "publaynet.tar.gz?_ga=2.23017467.1796315263.1628754613-1173244232.1625045842"
)
_SPLITS = {"train": "/train", "val": "/val"}

_LOCATION = "/publaynet"

_ANNOTATION_FILES: Dict[str, Union[str, List[str]]] = {"train": "train.json", "val": "val.json"}
_INIT_CATEGORIES = [names.C.TEXT, names.C.TITLE, names.C.LIST, names.C.TAB, names.C.FIG]


class Publaynet(_BuiltInDataset):
    """
    Publaynet
    """

    _name = _NAME

    def _info(self) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES)

    def _builder(self) -> "PublaynetBuilder":
        return PublaynetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class PublaynetBuilder(DataFlowBaseBuilder):
    """
    Publaynet dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:  # pylint: disable=W0221 (#3812)   type: ignore
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        :param split: Split of the dataset. Can be "train","val" or "test". Default: "val"
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None
        :param load_image: Will load the image for each datapoint.  Default: False
        :param fake_score: Will add a fake score so that annotations look like predictions

        :return: dataflow
        """
        split = str(kwargs.get("split", "val"))
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        load_image = kwargs.get("load_image", False)
        fake_score = kwargs.get("fake_score", False)

        # Load
        path = os.path.join(self.get_workdir(), self.annotation_files[split])  # type: ignore
        df = SerializerCoco.load(path, max_datapoints=max_datapoints)

        # Map
        df = MapDataComponent(df, lambda dp: self.get_workdir() + self.get_split(split) + "/" + dp, "file_name")
        coco_mapper = coco_to_image(  # type: ignore # pylint: disable=E1120  # 259
            self.categories.get_categories(init=True),  # type: ignore
            load_image,
            filter_empty_image=True,
            fake_score=fake_score,
        )
        df = MapData(df, coco_mapper)

        assert self.categories is not None  # avoid many typing issues
        if self.categories.is_filtered():
            df = MapData(
                df,
                filter_cat(  # type: ignore  # pylint: disable=E1120
                    self.categories.get_categories(as_dict=False, filtered=True),
                    self.categories.get_categories(as_dict=False, filtered=False),
                ),
            )

        return df
