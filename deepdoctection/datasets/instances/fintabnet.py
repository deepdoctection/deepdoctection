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

    fintabnet
    ├── pdf
    │ ├── A
    │ │├── 2003
    │ │ ...
    ├── FinTabNet_1.0.0_cell_test.jsonl
    ├── FinTabNet_1.0.0_cell_train.jsonl
    ├── FinTabNet_1.0.0_cell_val.jsonl
    ├── FinTabNet_1.0.0_table_test.jsonl
    ├── FinTabNet_1.0.0_table_train.jsonl
    ├── FinTabNet_1.0.0_table_val.jsonl
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

from ...dataflow import DataFlow, MapData, MultiProcessMapData
from ...dataflow.common import FlattenData
from ...dataflow.custom_serialize import SerializerJsonlines
from ...datapoint.image import Image
from ...mapper.cats import cat_to_sub_cat, filter_cat
from ...mapper.maputils import curry
from ...mapper.misc import image_ann_to_image, maybe_ann_to_sub_image
from ...mapper.pubstruct import pub_to_image
from ...utils.file_utils import set_mp_spawn
from ...utils.logger import LoggingRecord, logger
from ...utils.settings import CellType, DatasetType, LayoutType, ObjectTypes, TableType
from ...utils.types import PubtabnetDict
from ...utils.utils import to_bool
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories, DatasetInfo
from ..registry import dataset_registry

_NAME = "fintabnet"
_SHORT_DESCRIPTION = "FinTabNet dataset contains complex tables from the annual reports of S&P 500 companies."
_DESCRIPTION = (
    "FinTabNet dataset contains complex tables from the annual reports of S&P 500 companies with detailed \n"
    " table structure annotations to help train and test structure recognition. \n"
    "To generate the cell structure labels, one uses token matching between the PDF and HTML version \n"
    " of each article from public records and filings. Financial tables often have diverse styles when \n"
    "compared to ones in scientific and government documents, with fewer graphical lines and larger \n"
    " gaps within each table and more colour variations. Fintabnet can be used for training cell \n"
    "detection models as well as for semantic table understanding algorithms. \n"
    "For detection it has cell bounding box annotations as well as precisely described table semantics \n"
    "like row - and column numbers and row and col spans. The dataflow builder can also \n"
    "return captions of bounding boxes of rows and columns. Moreover, various filter conditions on \n"
    "the table structure are available: maximum cell numbers, maximal row and column numbers and their \n"
    "minimum equivalents can be used as filter condition. Header information of cells are not available. \n"
    "As work around you can artificially add header sub-category to every first row cell. \n"
    "All later row cells will receive a no header  sub-category. Note, that this assumption \n"
    "will generate noise."
)
_LICENSE = (
    "Community Data License Agreement – Permissive – Version 1.0  ---- \n"
    "This is the Community Data License Agreement – Permissive, Version 1.0 (“Agreement”).  \n"
    "Data is provided to You under this Agreement by each of the Data Providers.  Your exercise of any of \n"
    " the rights and permissions granted below constitutes Your acceptance and agreement to be bound by \n"
    " the terms and conditions of this Agreement."
)
_URL = (
    "https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/"
    "fintabnet.tar.gz?_ga=2.17492593.994196051.1634564576-1173244232.1625045842"
)
_SPLITS: Mapping[str, str] = {"train": "train", "val": "val", "test": "test"}
_TYPE = DatasetType.OBJECT_DETECTION
_LOCATION = "fintabnet"
_ANNOTATION_FILES: Mapping[str, str] = {
    "train": "FinTabNet_1.0.0_table_train.jsonl",
    "test": "FinTabNet_1.0.0_table_test.jsonl",
    "val": "FinTabNet_1.0.0_table_val.jsonl",
}
_INIT_CATEGORIES = [LayoutType.TABLE, LayoutType.CELL, TableType.ITEM]
_SUB_CATEGORIES: Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]]
_SUB_CATEGORIES = {
    LayoutType.CELL: {
        CellType.HEADER: [CellType.HEADER, CellType.BODY],
        CellType.ROW_NUMBER: [],
        CellType.COLUMN_NUMBER: [],
        CellType.ROW_SPAN: [],
        CellType.COLUMN_SPAN: [],
        CellType.SPANNING: [CellType.SPANNING, LayoutType.CELL],
    },
    TableType.ITEM: {TableType.ITEM: [LayoutType.ROW, LayoutType.COLUMN]},
    CellType.HEADER: {
        CellType.ROW_NUMBER: [],
        CellType.COLUMN_NUMBER: [],
        CellType.ROW_SPAN: [],
        CellType.COLUMN_SPAN: [],
        CellType.SPANNING: [CellType.SPANNING, LayoutType.CELL],
    },
    CellType.BODY: {
        CellType.ROW_NUMBER: [],
        CellType.COLUMN_NUMBER: [],
        CellType.ROW_SPAN: [],
        CellType.COLUMN_SPAN: [],
        CellType.SPANNING: [CellType.SPANNING, LayoutType.CELL],
    },
}


@dataset_registry.register("fintabnet")
class Fintabnet(_BuiltInDataset):
    """
    Fintabnet
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

    def _builder(self) -> FintabnetBuilder:
        return FintabnetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class FintabnetBuilder(DataFlowBaseBuilder):
    """
    Fintabnet builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        Args:
            kwargs:
                (split) Split of the dataset. Can be `train`, `val` or `test`. Default: `val`
                (build_mode) Returns the full image or crops a table according to the table bounding box. Pass `table`
                             if you only want the cropped table. Default: `""`
                (max_datapoints) Will stop iterating after `max_datapoints`. Default: `None`
                (rows_and_cols) Will add 'item' image annotations that either represent a row or a column of a table.
                                Default: `True`
                (load_image) Will load the image for each datapoint. Default: `False`
                (use_multi_proc) Uses multiple processes for PDF conversion. Default: `True`
                (use_multi_proc_strict) Uses strict mode in multiprocessing. Default: `False`
                (fake_score) Adds a fake score so that annotations look like predictions. Default: `False`
                (pubtables_like) Treats the dataset as PubTables-like. Default: `False`

        Returns:
            Dataflow
        """

        split = str(kwargs.get("split", "val"))
        max_datapoints = kwargs.get("max_datapoints")
        rows_and_cols = kwargs.get("rows_and_cols", True)
        load_image = kwargs.get("load_image", False)
        use_multi_proc = to_bool(kwargs.get("use_multi_proc", True))
        use_multi_proc_strict = to_bool(kwargs.get("use_multi_proc_strict", False))
        fake_score = kwargs.get("fake_score", False)
        build_mode = kwargs.get("build_mode")
        pubtables_like = kwargs.get("pubtables_like", False)

        if build_mode and not load_image:
            logger.info(LoggingRecord("When 'build_mode' is set to True will reset 'load_image' to True"))
            load_image = True

        if use_multi_proc or use_multi_proc_strict:
            set_mp_spawn()

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        if kwargs.get("build_mode", "") != "table":
            logger.info(LoggingRecord("Logic will display only only table per page, even if there are more!!"))

        # Load
        df: DataFlow
        path = self.get_workdir() / self.get_annotation_file(split)
        df = SerializerJsonlines.load(path, max_datapoints=max_datapoints)

        # Map
        @curry
        def _map_filename(dp: PubtabnetDict, workdir: Path) -> PubtabnetDict:
            dp["filename"] = workdir / "pdf" / dp["filename"]
            return dp

        df = MapData(df, _map_filename(self.get_workdir()))

        buffer_size = 200 if max_datapoints is None else min(max_datapoints, 200) - 1

        pub_mapper = pub_to_image(
            categories_name_as_key=self.categories.get_categories(name_as_key=True, init=True),
            load_image=load_image,
            fake_score=fake_score,
            rows_and_cols=rows_and_cols,
            dd_pipe_like=False,
            is_fintabnet=True,
            pubtables_like=pubtables_like,
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

        if build_mode == "table":

            @curry
            def _crop_and_add_image(dp: Image, category_names: list[str]) -> Image:
                return image_ann_to_image(dp, category_names=category_names)

            df = MapData(
                df,
                _crop_and_add_image(  # pylint: disable=E1120
                    category_names=[
                        LayoutType.TABLE,
                        LayoutType.CELL,
                        CellType.HEADER,
                        CellType.BODY,
                        TableType.ITEM,
                        LayoutType.ROW,
                        LayoutType.COLUMN,
                    ]
                ),
            )
            df = MapData(
                df,
                maybe_ann_to_sub_image(  # pylint: disable=E1120  # 259
                    category_names_sub_image=LayoutType.TABLE,
                    category_names=[
                        LayoutType.CELL,
                        CellType.HEADER,
                        CellType.BODY,
                        TableType.ITEM,
                        LayoutType.ROW,
                        LayoutType.COLUMN,
                    ],
                    add_summary=True,
                ),
            )
            df = MapData(df, lambda dp: [ann.image for ann in dp.get_annotation(category_names=LayoutType.TABLE)])
            df = FlattenData(df)
            df = MapData(df, lambda dp: dp[0])

        if self.categories.is_cat_to_sub_cat():
            df = MapData(
                df,
                cat_to_sub_cat(self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat),
            )

        if self.categories.is_filtered():
            df = MapData(
                df,
                filter_cat(  # pylint: disable=E1120
                    self.categories.get_categories(as_dict=False, filtered=True),
                    self.categories.get_categories(as_dict=False, filtered=False),
                ),
            )
        return df
