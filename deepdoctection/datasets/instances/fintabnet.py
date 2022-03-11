# -*- coding: utf-8 -*-
# File: fintabnet.py

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
Module for Fintabnet dataset. Place the dataset as follows

|    fintabnet
|    ├── pdf
|    │ ├── A
|    │ │├── 2003
|    │ │ ...
|    ├── FinTabNet_1.0.0_cell_test.jsonl
|    ├── FinTabNet_1.0.0_cell_train.jsonl
|    ├── FinTabNet_1.0.0_cell_val.jsonl
|    ├── FinTabNet_1.0.0_table_test.jsonl
|    ├── FinTabNet_1.0.0_table_train.jsonl
|    ├── FinTabNet_1.0.0_table_val.jsonl
"""

import os
from typing import Dict, List, Union

from ...dataflow import DataFlow, MapData, MultiProcessMapData  # type: ignore
from ...dataflow.common import FlattenData
from ...dataflow.custom_serialize import SerializerJsonlines
from ...datapoint.image import Image
from ...mapper.cats import cat_to_sub_cat, filter_cat
from ...mapper.maputils import cur
from ...mapper.misc import image_ann_to_image, maybe_ann_to_sub_image
from ...mapper.pubstruct import pub_to_image
from ...utils.detection_types import JsonDict
from ...utils.file_utils import set_mp_spawn
from ...utils.logger import logger
from ...utils.settings import names
from ...utils.utils import to_bool
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories, DatasetInfo

_NAME = "fintabnet"
_DESCRIPTION = (
    "FinTabNet dataset contains complex tables from the annual reports of S&P 500 companies with detailed"
    " table structure annotations to help train and test structure recognition. "
    "To generate the cell structure labels, one uses token matching between the PDF and HTML version"
    " of each article from public records and filings. Financial tables often have diverse styles when "
    "compared to ones in scientific and government documents, with fewer graphical lines and larger"
    " gaps within each table and more colour variations. Fintabnet can be used for training cell "
    "detection models as well as for semantic table understanding algorithms. "
    "For detection it has cell bounding box annotations as well as precisely described table semantics "
    "like row - and column numbers and row and col spans. The dataflow builder can also "
    "return captions of bounding boxes of rows and columns. Moreover, various filter conditions on "
    "the table structure are available: maximum cell numbers, maximal row and column numbers and their "
    "minimum equivalents can be used as filter condition. Header information of cells are not available. "
    "As work around you can artificially add header sub-category to every first row cell. "
    "All later row cells will receive a no header  sub-category. Note, that this assumption "
    "will generate noise."
)
_LICENSE = (
    "Community Data License Agreement – Permissive – Version 1.0  ---- "
    "This is the Community Data License Agreement – Permissive, Version 1.0 (“Agreement”).  "
    "Data is provided to You under this Agreement by each of the Data Providers.  Your exercise of any of"
    " the rights and permissions granted below constitutes Your acceptance and agreement to be bound by"
    " the terms and conditions of this Agreement."
)
_URL = (
    "https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/"
    "fintabnet.tar.gz?_ga=2.17492593.994196051.1634564576-1173244232.1625045842"
)
_SPLITS = {"train": "/train", "val": "/val", "test": "/test"}
_LOCATION = "/fintabnet"
_ANNOTATION_FILES: Dict[str, Union[str, List[str]]] = {
    "train": "FinTabNet_1.0.0_table_train.jsonl",
    "test": "FinTabNet_1.0.0_table_test.jsonl",
    "val": "FinTabNet_1.0.0_table_val.jsonl",
}
_INIT_CATEGORIES = [names.C.TAB, names.C.CELL, names.C.ITEM]
_SUB_CATEGORIES: Dict[str, Dict[str, List[str]]]
_SUB_CATEGORIES = {
    names.C.CELL: {
        names.C.HEAD: [names.C.HEAD, names.C.BODY],
        names.C.RN: [],
        names.C.CN: [],
        names.C.RS: [],
        names.C.CS: [],
    },
    names.C.ITEM: {"row_col": [names.C.ROW, names.C.COL]},
    names.C.HEAD: {names.C.RN: [], names.C.CN: [], names.C.RS: [], names.C.CS: []},
    names.C.BODY: {names.C.RN: [], names.C.CN: [], names.C.RS: [], names.C.CS: []},
}


class Fintabnet(_BuiltInDataset):
    """
    Fintabnet
    """

    _name = _NAME

    def _info(self) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> "FintabnetBuilder":
        return FintabnetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class FintabnetBuilder(DataFlowBaseBuilder):
    """
    Fintabnet builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        :param split: Split of the dataset. Can be "train","val" or "test". Default: "val"
        :param build_mode: Returns the full image or crops a table according to the table bounding box. Pass "table"
                           if you only want the cropped table. Default: ""
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None
        :param rows_and_cols: Will add a "ITEM" image annotations that either represent a row or a column of a table.
                              Note, that the type of the item (i.e. being a row or a column) can be inferred from the
                              sub category added. Note further, that "ITEM" are not originally part of the annotations
                              and are inferred from cell positions and their associated table semantic. Default: True
        :param load_image: Will load the image for each datapoint.  Default: False
        :param use_multi_proc: As the original files are stored as pdf conversion into a numpy array is time-consuming.
                               When setting use_multi_proc to True is will use several processes depending on the number
                               of CPUs available.
        :param use_multi_proc_strict: Will use strict mode in multiprocessing.
        :param fake_score: Will add a fake score so that annotations look like predictions

        :return: dataflow
        """

        split = kwargs.get("split", "val")
        max_datapoints = kwargs.get("max_datapoints")
        rows_and_cols = kwargs.get("rows_and_cols", True)
        load_image = kwargs.get("load_image", False)
        use_multi_proc = to_bool(kwargs.get("use_multi_proc", True))
        use_multi_proc_strict = to_bool(kwargs.get("use_multi_proc_strict", False))
        fake_score = kwargs.get("fake_score", False)

        if use_multi_proc or use_multi_proc_strict:
            set_mp_spawn()

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        if kwargs.get("build_mode", "") != "table":
            logger.info("Logic will currently display only ONE table per page, even if there are more !!")

        # Load
        path = os.path.join(self.get_workdir(), self.annotation_files[split])  # type: ignore
        df = SerializerJsonlines.load(path, max_datapoints=max_datapoints)

        # Map
        @cur  # type: ignore
        def _map_filename(dp: JsonDict, workdir: str) -> JsonDict:
            dp["filename"] = workdir + "/pdf/" + dp["filename"]
            return dp

        map_filename = _map_filename(self.get_workdir())  # pylint: disable=E1120  # 259  # type: ignore
        df = MapData(df, map_filename)

        buffer_size = 200 if max_datapoints is None else min(max_datapoints, 200) - 1

        pub_mapper = pub_to_image(
            self.categories.get_categories(name_as_key=True, init=True),  # type: ignore
            # pylint: disable=E1120  # 259
            load_image,
            fake_score=fake_score,
            rows_and_cols=rows_and_cols,
        )
        if use_multi_proc:
            df = MultiProcessMapData(
                df,
                num_proc=1 if buffer_size < 3 else 4,
                map_func=pub_mapper,
                strict=use_multi_proc_strict,
                buffer_size=buffer_size,
            )
        else:
            df = MapData(df, pub_mapper)

        if kwargs.get("build_mode", "") == "table":

            @cur  # type: ignore
            def _crop_and_add_image(dp: Image, category_names: List[str]) -> Image:
                return image_ann_to_image(dp, category_names=category_names)

            df = MapData(
                df,
                _crop_and_add_image(  # pylint: disable=E1120
                    category_names=[
                        names.C.TAB,
                        names.C.CELL,
                        names.C.HEAD,
                        names.C.BODY,  # type: ignore
                        names.C.ITEM,
                        names.C.ROW,
                        names.C.COL,
                    ]
                ),
            )
            ann_to_sub_image = maybe_ann_to_sub_image(  # pylint: disable=E1120  # 259
                category_names_sub_image=names.C.TAB,  # type: ignore
                category_names=[names.C.CELL, names.C.HEAD, names.C.BODY, names.C.ITEM, names.C.ROW, names.C.COL],
            )
            df = MapData(df, ann_to_sub_image)
            df = MapData(df, lambda dp: [ann.image for ann in dp.get_annotation_iter(category_names=names.C.TAB)])
            df = FlattenData(df)
            df = MapData(df, lambda dp: dp[0])

        if self.categories.is_cat_to_sub_cat():  # type: ignore
            df = MapData(
                df,
                cat_to_sub_cat(
                    self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat  # type: ignore
                ),
            )

        if self.categories.is_filtered():  # type: ignore
            df = MapData(
                df,
                filter_cat(  # pylint: disable=E1120
                    self.categories.get_categories(as_dict=False, filtered=True),  # type: ignore
                    self.categories.get_categories(as_dict=False, filtered=False),  # type: ignore
                ),
            )
        return df
