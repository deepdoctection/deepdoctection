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

|    PubTables1M-Detection-PASCAL-VOC
|    ├── images
|    │ ├── PMC5700015_4.jpg
|    │ ├── PMC5700016_4.jpg
|    ├── test
|    │ ├── PMC512281_8.xml
|    │ ├── PMC512281_9.xml
|    ├── train
|    │ ├── ...
|    ├── val
|    │ ├── ...
|    ├── images_filelist.txt
|    ├── test_filelist.txt
|    ├── train_filelist.txt
|    ├── val_filelist.txt
"""

import os
from typing import Dict, List, Union

from lxml import etree  # type: ignore

from ...dataflow import DataFlow, MapData, SerializerFiles  # type: ignore
from ...datasets.info import DatasetInfo
from ...mapper.maputils import cur
from ...mapper.misc import xml_to_dict
from ...mapper.pascalstruct import pascal_voc_dict_to_image
from ...utils.detection_types import JsonDict
from ...utils.settings import names
from ...utils.systools import get_package_path
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories

_NAME = "pubtables1m"

_DESCRIPTION = (
    "[excerpt from Ajoy Mondal et. all. IIIT-AR-13K: A New Dataset for Graphical Object Detection in "
    "Documents] ...we release PubTables1M, a dataset of nearly one million tables from PubMed Central Open Access"
    " scientific articles, with complete bounding box annotations for both table detection and structure recognition."
    " In addition to being the largest dataset of its kind, PubTables1M addresses issues such as inherent ambiguity"
    " and lack of consistency in the source annotations, attempting to provide definitive ground truth labels through"
    " a thorough canonicalization and quality control process. "
)

_LICENSE = "Community Data License Agreement – Permissive, Version 1.0"

_URL = "https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3"

_SPLITS = {"train": "train", "val": "val", "test": "test"}
_LOCATION = "/PubTables1M-Detection-PASCAL-VOC"
_ANNOTATION_FILES: Dict[str, Union[str, List[str]]] = {
    "train": "train",
    "val": "val",
    "test": "test",
}

_INIT_CATEGORIES = [names.C.TAB]


class Pubtables1M(_BuiltInDataset):
    """
    Pubtables1M
    """

    _name = _NAME

    def _info(self) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES)

    def _builder(self) -> "Pubtables1MBuilder":
        return Pubtables1MBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class Pubtables1MBuilder(DataFlowBaseBuilder):
    """
    Pubtables1M dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the return
        values of the dataflow:

        :param split: Split of the dataset. Can be "train","val" or "test". Default: "val"
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None
        :param load_image: Will load the image for each datapoint.  Default: False
        :param fake_score: Will add a fake score so that annotations look like predictions

        :return: dataflow
        """

        split = str(kwargs.get("split", "val"))
        load_image = kwargs.get("load_image", False)
        max_datapoints = kwargs.get("max_datapoints")
        fake_score = kwargs.get("fake_score", False)

        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)

        # Load
        path_ann_files = os.path.join(self.get_workdir(), self.annotation_files[split])  # type: ignore

        df = SerializerFiles.load(path_ann_files, ".xml", max_datapoints)
        utf8_parser = etree.XMLParser(encoding="utf-8")

        @cur  # type: ignore
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
            pascal_voc_dict_to_image(  # type: ignore # pylint: disable = E1120
                self.categories.get_categories(init=True, name_as_key=True),  # type: ignore
                load_image,
                filter_empty_image=True,
                fake_score=fake_score,
                category_name_mapping={"table": names.C.TAB},
            ),
        )

        return df
