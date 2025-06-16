# -*- coding: utf-8 -*-
# File: pubtables1m.py

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
Module for PubTables1M-Detection-PASCAL-VOC dataset. Install the dataset following the folder structure

    PubTables1M
    ├── PubTables1M-Detection-PASCAL-VOC
    │├── images
    ││ ├── PMC5700015_4.jpg
    ││ ├── PMC5700016_4.jpg
    │├── test
    ││ ├── PMC512281_8.xml
    ││ ├── PMC512281_9.xml
    │├── train
    ││ ├── ...
    │├── val
    ││ ├── ...
    │├── images_filelist.txt
    │├── test_filelist.txt
    │├── train_filelist.txt
    │├── val_filelist.txt
    ├── PubTables-1M-Structure_Annotations_Test
    ├── PubTables-1M-Structure_Images_Test
"""
from __future__ import annotations

import os
from typing import Mapping, Union

from lazy_imports import try_import

from ...dataflow import DataFlow, MapData, SerializerFiles
from ...datasets.info import DatasetInfo
from ...mapper.cats import filter_cat
from ...mapper.maputils import curry
from ...mapper.misc import xml_to_dict
from ...mapper.pascalstruct import pascal_voc_dict_to_image
from ...utils.file_utils import lxml_available
from ...utils.fs import get_package_path
from ...utils.settings import CellType, DatasetType, LayoutType
from ...utils.types import JsonDict
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories
from ..registry import dataset_registry

with try_import() as import_guard:
    from lxml import etree

_NAME = "pubtables1m_det"
_SHORT_DESCRIPTION = "PubTables1M is a dataset for table detection and structure recognition."
_DESCRIPTION = (
    "[excerpt from Brandon Smock et. all. PubTables-1M: Towards Comprehensive Table Extraction From Unstructured \n"
    "Documents] '...we release PubTables1M, a dataset of nearly one million tables from PubMed Central Open Access \n"
    " scientific articles, with complete bounding box annotations for both table detection and structure \n"
    "recognition. In addition to being the largest dataset of its kind, PubTables1M addresses issues such as \n"
    " inherent ambiguity and lack of consistency in the source annotations, attempting to provide definitive ground \n"
    " truth labels through a thorough canonicalization and quality control process.' This dataset is devoted to two "
    "different tasks: table detection and table structure recognition. For this first task use  'pubtables1m_det' "
    "whereas for the second 'pubtables1m_struct'"
)

_LICENSE = "Community Data License Agreement – Permissive, Version 1.0"

_URL = "https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3"

_SPLITS: Mapping[str, str] = {"train": "train", "val": "val", "test": "test"}
_TYPE = DatasetType.OBJECT_DETECTION
_LOCATION = "PubTables1M"
_ANNOTATION_FILES: Mapping[str, str] = {
    "train": "PubTables1M-Detection-PASCAL-VOC/train",
    "val": "PubTables1M-Detection-PASCAL-VOC/val",
    "test": "PubTables1M-Detection-PASCAL-VOC/test",
}
_INIT_CATEGORIES_DET = [LayoutType.TABLE, LayoutType.TABLE_ROTATED]


@dataset_registry.register("pubtables1m_det")
class Pubtables1MDet(_BuiltInDataset):
    """
    Pubtables1MDet
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
        return DatasetCategories(init_categories=_INIT_CATEGORIES_DET)

    def _builder(self) -> Pubtables1MBuilder:
        return Pubtables1MBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class Pubtables1MBuilder(DataFlowBaseBuilder):
    """
    `Pubtables1M` dataflow builder
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
                (fake_score) Will add a fake score so that annotations look like predictions. Default: `False`

        Returns:
            Dataflow
        """

        if not lxml_available():
            raise ModuleNotFoundError("Pubtables1MBuilder.build requires lxml but it is not installed.")

        split = str(kwargs.get("split", "val"))
        load_image = kwargs.get("load_image", False)
        max_datapoints = kwargs.get("max_datapoints")
        fake_score = kwargs.get("fake_score", False)

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)

        # Load
        path_ann_files = self.get_workdir() / self.get_annotation_file(split)

        df = SerializerFiles.load(path_ann_files, ".xml", max_datapoints)
        utf8_parser = etree.XMLParser(encoding="utf-8")

        @curry
        def load_xml(path_ann: str, utf8_parser: etree.XMLParser) -> JsonDict:
            with open(path_ann, "r", encoding="utf-8") as xml_file:
                root = etree.fromstring(xml_file.read().encode("utf_8"), parser=utf8_parser)
            return {"file_name": path_ann, "xml": root}

        df = MapData(df, load_xml(utf8_parser))  # pylint: disable=E1120

        with open(
            os.path.join(get_package_path(), "deepdoctection/datasets/instances/xsl/pascal_voc.xsl"),
            "r",
            encoding="utf-8",
        ) as xsl_file:
            xslt_file = xsl_file.read().encode("utf-8")
        xml_obj = etree.XML(xslt_file, parser=etree.XMLParser(encoding="utf-8"))
        xslt_obj = etree.XSLT(xml_obj)

        df = MapData(df, xml_to_dict(xslt_obj))  # pylint: disable = E1120

        def _map_file_name(dp: JsonDict) -> JsonDict:
            path, file_name = os.path.split(dp["file_name"])
            path = os.path.join(os.path.split(path)[0], "images")
            dp["json"]["filename"] = os.path.join(path, file_name.replace(".xml", ".jpg"))
            return dp["json"]

        df = MapData(df, _map_file_name)
        df = MapData(
            df,
            pascal_voc_dict_to_image(  # pylint: disable = E1120
                self.categories.get_categories(init=True, name_as_key=True),
                load_image,
                filter_empty_image=True,
                fake_score=fake_score,
                category_name_mapping={"table": LayoutType.TABLE, "table rotated": LayoutType.TABLE_ROTATED},
            ),
        )

        return df


_NAME_STRUCT = "pubtables1m_struct"
_ANNOTATION_FILES_STRUCT: Mapping[str, str] = {
    "train": "PubTables-1M-Structure_Annotations_Train",
    "val": "PubTables-1M-Structure_Annotations_Val",
    "test": "PubTables-1M-Structure_Annotations_Test",
}

_INIT_CATEGORIES_STRUCT = [
    LayoutType.TABLE,
    LayoutType.ROW,
    LayoutType.COLUMN,
    CellType.SPANNING,
    CellType.ROW_HEADER,
    CellType.COLUMN_HEADER,
    CellType.PROJECTED_ROW_HEADER,
]

_IMAGES: Mapping[str, str] = {
    "train": "PubTables-1M-Structure_Images_Train",
    "val": "PubTables-1M-Structure_Images_Val",
    "test": "PubTables-1M-Structure_Images_Test",
}


@dataset_registry.register("pubtables1m_struct")
class Pubtables1MStruct(_BuiltInDataset):
    """
    Pubtables1MStruct
    """

    _name = _NAME_STRUCT

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(
            name=_NAME_STRUCT, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS, type=_TYPE
        )

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES_STRUCT)

    def _builder(self) -> Pubtables1MBuilderStruct:
        return Pubtables1MBuilderStruct(location=_LOCATION, annotation_files=_ANNOTATION_FILES_STRUCT)


class Pubtables1MBuilderStruct(DataFlowBaseBuilder):
    """
    Pubtables1M dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the return
        values of the dataflow:

        `split:` Split of the dataset. Can be `train`, `val` or `test`. Default: `val`
        `max_datapoints:` Will stop iterating after max_datapoints. Default: `None`
        `load_image:` Will load the image for each datapoint.  Default: `False`
        `fake_score:` Will add a fake score so that annotations look like predictions

        :return: dataflow
        """
        if not lxml_available():
            raise ModuleNotFoundError("Pubtables1MBuilderStruct.build requires lxml but it is not installed.")

        split = str(kwargs.get("split", "val"))
        load_image = kwargs.get("load_image", False)
        max_datapoints = kwargs.get("max_datapoints")
        fake_score = kwargs.get("fake_score", False)

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)

        # Load
        path_ann_files = self.get_workdir() / self.get_annotation_file(split)

        df = SerializerFiles.load(path_ann_files, ".xml", max_datapoints)
        utf8_parser = etree.XMLParser(encoding="utf-8")

        @curry
        def load_xml(path_ann: str, utf8_parser: etree.XMLParser) -> JsonDict:
            with open(path_ann, "r", encoding="utf-8") as xml_file:
                root = etree.fromstring(xml_file.read().encode("utf_8"), parser=utf8_parser)
            return {"file_name": path_ann, "xml": root}

        df = MapData(df, load_xml(utf8_parser))  # pylint: disable=E1120

        with open(
            os.path.join(get_package_path(), "deepdoctection/datasets/instances/xsl/pascal_voc.xsl"),
            "r",
            encoding="utf-8",
        ) as xsl_file:
            xslt_file = xsl_file.read().encode("utf-8")
        xml_obj = etree.XML(xslt_file, parser=etree.XMLParser(encoding="utf-8"))
        xslt_obj = etree.XSLT(xml_obj)

        df = MapData(df, xml_to_dict(xslt_obj))  # pylint: disable = E1120

        @curry
        def _map_file_name(dp: JsonDict, split: str) -> JsonDict:
            path, file_name = os.path.split(dp["file_name"])
            file_split = _IMAGES[split]
            path = os.path.join(os.path.split(path)[0], file_split)
            dp["json"]["filename"] = os.path.join(path, file_name.replace(".xml", ".jpg"))
            return dp["json"]

        df = MapData(df, _map_file_name(split))
        df = MapData(
            df,
            pascal_voc_dict_to_image(  # pylint: disable = E1120
                self.categories.get_categories(init=True, name_as_key=True),
                load_image,
                filter_empty_image=True,
                fake_score=fake_score,
                category_name_mapping={
                    "table": LayoutType.TABLE,
                    "table spanning cell": CellType.SPANNING,
                    "table row": LayoutType.ROW,
                    "table row header": CellType.ROW_HEADER,
                    "table projected row header": CellType.PROJECTED_ROW_HEADER,
                    "table column": LayoutType.COLUMN,
                    "table column header": CellType.COLUMN_HEADER,
                },
            ),
        )

        assert self.categories is not None  # avoid many typing issues
        if self.categories.is_filtered():
            df = MapData(
                df,
                filter_cat(
                    self.categories.get_categories(as_dict=False, filtered=True),
                    self.categories.get_categories(as_dict=False, filtered=False),
                ),
            )

        return df
