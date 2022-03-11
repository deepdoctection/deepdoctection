# -*- coding: utf-8 -*-
# File: pubtabnet.py

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
Module for Pubtabnet dataset. Place the dataset as follows

|    pubtabnet
|    ├── test
|    │ ├── PMC1.png
|    ├── train
|    │ ├── PMC2.png
|    ├── val
|    │ ├── PMC3.png
|    ├── PubTabNet_2.0.0.jsonl
"""

import os
from typing import Dict, List, Union

from ...dataflow import DataFlow, MapData  # type: ignore
from ...dataflow.custom_serialize import SerializerJsonlines
from ...datasets.info import DatasetInfo
from ...mapper.cats import cat_to_sub_cat, filter_cat
from ...mapper.pubstruct import pub_to_image
from ...utils.detection_types import JsonDict
from ...utils.logger import log_once
from ...utils.settings import names
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories

_NAME = "pubtabnet"
_DESCRIPTION = (
    "PubTabNet is a large dataset for image-based table recognition, containing 568k+ images of "
    "tabular data annotated with the corresponding HTML representation of the tables. The table images"
    " are extracted from the scientific publications included in the PubMed Central Open Access Subset"
    " (commercial use collection). Table regions are identified by matching the PDF format and"
    " the XML format of the articles in the PubMed Central Open Access Subset. More details are"
    " available in our paper 'Image-based table recognition: data, model, and evaluation'. "
    "Pubtabnet can be used for training cell detection models as well as for semantic table "
    "understanding algorithms. For detection it has cell bounding box annotations as "
    "well as precisely described table semantics like row - and column numbers and row and col spans. "
    "Moreover, every cell can be classified as header or non-header cell. The dataflow builder can also "
    "return captions of bounding boxes of rows and columns. Moreover, various filter conditions on "
    "the table structure are available: maximum cell numbers, maximal row and column numbers and their "
    "minimum equivalents can be used as filter condition"
)
_LICENSE = (
    "The annotations in this dataset belong to IBM and are licensed under a Community Data License Agreement"
    " – Permissive – Version 1.0 License. IBM does not own the copyright of the images."
    " Use of the images must abide by the PMC Open Access Subset Terms of Use."
)
_URL = (
    "https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/"
    "pubtabnet.tar.gz?_ga=2.267291150.146828643.1629125962-1173244232.1625045842"
)
_SPLITS = {"train": "/train", "val": "/val", "test": "/test"}

_LOCATION = "/pubtabnet"
_ANNOTATION_FILES: Dict[str, Union[str, List[str]]] = {"all": "PubTabNet_2.0.0.jsonl"}

_INIT_CATEGORIES = [names.C.CELL, names.C.ITEM]
_SUB_CATEGORIES: Dict[str, Dict[str, List[str]]]
_SUB_CATEGORIES = {
    names.C.ITEM: {"row_col": [names.C.ROW, names.C.COL]},
    names.C.CELL: {
        names.C.HEAD: [names.C.HEAD, names.C.BODY],
        names.C.RN: [],
        names.C.CN: [],
        names.C.RS: [],
        names.C.CS: [],
    },
    names.C.HEAD: {names.C.RN: [], names.C.CN: [], names.C.RS: [], names.C.CS: []},
    names.C.BODY: {names.C.RN: [], names.C.CN: [], names.C.RS: [], names.C.CS: []},
}


class Pubtabnet(_BuiltInDataset):
    """
    Pubtabnet
    """

    _name = _NAME

    def _info(self) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> "PubtabnetBuilder":
        return PubtabnetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class PubtabnetBuilder(DataFlowBaseBuilder):
    """
    Pubtabnet dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        :param split: Split of the dataset. Can be "train","val" or "test". Default: "val"
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None
        :param load_image: Will load the image for each datapoint.  Default: False
        :param rows_and_cols: Will add a "ITEM" image annotations that either represent a row or a column of a table.
                              Note, that the type of the item (i.e. being a row or a column) can be inferred from the
                              sub category added. Note further, that "ITEM" are not originally part of the annotations
                              and are inferred from cell positions and their associated table semantic. Default: True
        :param fake_score: Will add a fake score so that annotations look like predictions

        :return: dataflow
        """
        split = kwargs.get("split", "val")
        if split == "val":
            log_once("Loading annotations for 'val' split from Pubtabnet will take some time...")
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
            max_datapoints = max_datapoints if split == "train" else 500777 + max_datapoints
        load_image = kwargs.get("load_image", False)
        rows_and_cols = kwargs.get("rows_and_cols", False)
        fake_score = kwargs.get("fake_score", False)

        # Load
        path = os.path.join(self.get_workdir(), self.annotation_files["all"])  # type: ignore
        df = SerializerJsonlines.load(path, max_datapoints=max_datapoints)

        # Map
        def replace_filename(dp: JsonDict) -> JsonDict:
            dp["filename"] = self.get_workdir() + "/" + dp["split"] + "/" + dp["filename"]
            return dp

        df = MapData(df, replace_filename)
        df = MapData(df, lambda dp: dp if dp["split"] == split else None)
        pub_mapper = pub_to_image(
            self.categories.get_categories(name_as_key=True, init=True),  # type: ignore
            # pylint: disable=E1120  # 259
            load_image,
            fake_score=fake_score,
            rows_and_cols=rows_and_cols,
        )

        df = MapData(df, pub_mapper)
        assert self.categories is not None  # avoid many typing issues
        if self.categories.is_cat_to_sub_cat():
            df = MapData(
                df,
                cat_to_sub_cat(
                    self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat  # type: ignore
                ),
            )

        if self.categories.is_filtered():
            df = MapData(
                df,
                filter_cat(  # type: ignore # pylint: disable=E1120
                    self.categories.get_categories(as_dict=False, filtered=True),
                    self.categories.get_categories(as_dict=False, filtered=False),
                ),
            )
        return df
