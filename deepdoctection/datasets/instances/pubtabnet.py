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

    pubtabnet
    ├── test
    │ ├── PMC1.png
    ├── train
    │ ├── PMC2.png
    ├── val
    │ ├── PMC3.png
    ├── PubTabNet_2.0.0.jsonl
"""
from __future__ import annotations

from typing import Mapping, Union

from ...dataflow import DataFlow, MapData
from ...dataflow.custom_serialize import SerializerJsonlines
from ...datasets.info import DatasetInfo
from ...mapper.cats import cat_to_sub_cat, filter_cat
from ...mapper.pubstruct import pub_to_image
from ...utils.logger import LoggingRecord, logger
from ...utils.settings import CellType, DatasetType, LayoutType, ObjectTypes, TableType, WordType
from ...utils.types import PubtabnetDict
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories
from ..registry import dataset_registry

_NAME = "pubtabnet"
_SHORT_DESCRIPTION = "PubTabNet is a dataset for image-based table recognition."
_DESCRIPTION = (
    "PubTabNet is a large dataset for image-based table recognition, containing 568k+ images of \n"
    "tabular data annotated with the corresponding HTML representation of the tables. The table images \n"
    " are extracted from the scientific publications included in the PubMed Central Open Access Subset \n"
    " (commercial use collection). Table regions are identified by matching the PDF format and \n"
    " the XML format of the articles in the PubMed Central Open Access Subset. More details are \n"
    " available in our paper 'Image-based table recognition: data, model, and evaluation'. \n"
    "Pubtabnet can be used for training cell detection models as well as for semantic table \n"
    "understanding algorithms. For detection it has cell bounding box annotations as \n"
    "well as precisely described table semantics like row - and column numbers and row and col spans. \n"
    "Moreover, every cell can be classified as header or non-header cell. The dataflow builder can also \n"
    "return captions of bounding boxes of rows and columns. Moreover, various filter conditions on \n"
    "the table structure are available: maximum cell numbers, maximal row and column numbers and their \n"
    "minimum equivalents can be used as filter condition"
)
_LICENSE = (
    "The annotations in this dataset belong to IBM and are licensed under a Community Data License Agreement \n"
    " – Permissive – Version 1.0 License. IBM does not own the copyright of the images. \n"
    " Use of the images must abide by the PMC Open Access Subset Terms of Use."
)
_URL = (
    "https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/"
    "pubtabnet.tar.gz?_ga=2.267291150.146828643.1629125962-1173244232.1625045842"
)
_SPLITS: Mapping[str, str] = {"train": "train", "val": "val", "test": "test"}
_TYPE = DatasetType.OBJECT_DETECTION
_LOCATION = "pubtabnet"
_ANNOTATION_FILES: Mapping[str, str] = {"all": "PubTabNet_2.0.0.jsonl"}

_INIT_CATEGORIES = [LayoutType.CELL, TableType.ITEM, LayoutType.TABLE, LayoutType.WORD]
_SUB_CATEGORIES: dict[ObjectTypes, dict[ObjectTypes, list[ObjectTypes]]]
_SUB_CATEGORIES = {
    TableType.ITEM: {TableType.ITEM: [LayoutType.ROW, LayoutType.COLUMN]},
    LayoutType.CELL: {
        CellType.HEADER: [CellType.HEADER, CellType.BODY],
        CellType.ROW_NUMBER: [],
        CellType.COLUMN_NUMBER: [],
        CellType.ROW_SPAN: [],
        CellType.COLUMN_SPAN: [],
        CellType.SPANNING: [CellType.SPANNING],
    },
    CellType.HEADER: {
        CellType.ROW_NUMBER: [],
        CellType.COLUMN_NUMBER: [],
        CellType.ROW_SPAN: [],
        CellType.COLUMN_SPAN: [],
        CellType.SPANNING: [CellType.SPANNING],
    },
    CellType.BODY: {
        CellType.ROW_NUMBER: [],
        CellType.COLUMN_NUMBER: [],
        CellType.ROW_SPAN: [],
        CellType.COLUMN_SPAN: [],
        CellType.SPANNING: [CellType.SPANNING],
    },
    LayoutType.TABLE: {TableType.HTML: [TableType.HTML]},
    LayoutType.WORD: {WordType.CHARACTERS: [WordType.CHARACTERS]},
}


@dataset_registry.register("pubtabnet")
class Pubtabnet(_BuiltInDataset):
    """
    `Pubtabnet`
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

    def _builder(self) -> PubtabnetBuilder:
        return PubtabnetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class PubtabnetBuilder(DataFlowBaseBuilder):
    """
    Pubtabnet dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        Args:
            kwargs:
                (split) Split of the dataset. Can be `train`, `val` or `test`. Default: `val`
                (max_datapoints) Will stop iterating after `max_datapoints`. Default: `None`
                (load_image) Will load the image for each datapoint. Default: `False`
                (rows_and_cols) Will add 'item' image annotations that represent rows or columns of a
                                table. Default: `True`
                (fake_score) Will add a fake score so that annotations look like predictions. Default: `False`
                (dd_pipe_like) If `True`, sets `load_image` to `True`. Default: `False`

        Returns:
            Dataflow
        """
        split = str(kwargs.get("split", "val"))
        if split == "val":
            logger.info(LoggingRecord("Loading annotations for 'val' split from Pubtabnet will take some time."))
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
            max_datapoints = max_datapoints if split == "train" else 500777 + max_datapoints
        load_image = kwargs.get("load_image", False)
        rows_and_cols = kwargs.get("rows_and_cols", False)
        fake_score = kwargs.get("fake_score", False)
        dd_pipe_like = kwargs.get("dd_pipe_like", False)
        if dd_pipe_like:
            logger.info(LoggingRecord("When 'dd_pipe_like'=True will set 'load_image'=True"))
            load_image = True

        # Load
        df: DataFlow
        path = self.get_workdir() / self.get_annotation_file("all")
        df = SerializerJsonlines.load(path, max_datapoints=max_datapoints)

        # Map
        def replace_filename(dp: PubtabnetDict) -> PubtabnetDict:
            dp["filename"] = self.get_workdir() / dp["split"] / dp["filename"]
            return dp

        df = MapData(df, replace_filename)
        df = MapData(df, lambda dp: dp if dp["split"] == split else None)
        pub_mapper = pub_to_image(
            self.categories.get_categories(name_as_key=True, init=True),
            load_image=load_image,
            fake_score=fake_score,
            rows_and_cols=rows_and_cols,
            dd_pipe_like=dd_pipe_like,
            is_fintabnet=False,
            pubtables_like=False,
        )

        df = MapData(df, pub_mapper)

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
