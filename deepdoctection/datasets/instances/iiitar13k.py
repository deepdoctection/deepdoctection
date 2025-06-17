# -*- coding: utf-8 -*-
# File: IIITAR13K.py

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
Module for IIITar13K dataset. Install the dataset following the folder structure

    iiitar13K
    ├── test_images
    │ ├── ar_alphabet_2004_eng_38.jpg
    │ ├── ar_alphabet_2004_eng_39.jpg
    ├── test_xml
    │ ├── ar_alphabet_2004_eng_38.xml
    │ ├── ar_alphabet_2004_eng_39.xml
    ├── training_images
    │ ├── ...
    ├── training_xml
    │ ├── ...
    ├── validation_images
    │ ├── ...
    ├── validation_xml
    │ ├── ...

"""
from __future__ import annotations

import os
from typing import Mapping, Union

from lazy_imports import try_import

from ...dataflow import DataFlow, MapData, SerializerFiles
from ...datasets.info import DatasetInfo
from ...mapper.maputils import curry
from ...mapper.misc import xml_to_dict
from ...mapper.pascalstruct import pascal_voc_dict_to_image
from ...utils.file_utils import lxml_available
from ...utils.fs import get_package_path
from ...utils.settings import DatasetType, LayoutType
from ...utils.types import JsonDict
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories
from ..registry import dataset_registry

with try_import() as import_guard:
    from lxml import etree

_NAME = "iiitar13k"
_SHORT_DESCRIPTION = "IIIT-AR-13K: A New Dataset for Graphical Object Detection in Documents"
_DESCRIPTION = (
    "[excerpt from Ajoy Mondal et. all. IIIT-AR-13K: A New Dataset for Graphical Object Detection in \n"
    "Documents] ...This dataset, IIIT-AR-13K, is created by manually annotating the bounding boxes of \n"
    "graphical or page objects in publicly available annual reports. This dataset contains a total of 13K \n"
    "annotated page images with objects in five different popular categories — table, figure, natural \n"
    "image, logo, and signature. This is the largest manually annotated dataset for graphical object \n"
    "detection. Annual reports created in multiple languages for several years from various companies \n"
    "bring high diversity into this dataset."
)

_LICENSE = "NN"

_URL = "http://cvit.iiit.ac.in/usodi/iiitar13k.php"

_SPLITS: Mapping[str, str] = {"train": "training_images", "val": "validation_images", "test": "test_images"}
_TYPE = DatasetType.OBJECT_DETECTION
_LOCATION = "iiitar13k"
_ANNOTATION_FILES: Mapping[str, str] = {
    "train": "training_xml",
    "val": "validation_xml",
    "test": "test_xml",
}

_INIT_CATEGORIES = [LayoutType.TABLE, LayoutType.LOGO, LayoutType.FIGURE, LayoutType.SIGNATURE]


@dataset_registry.register("iiitar13k")
class IIITar13K(_BuiltInDataset):
    """
    IIITar13K
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
        return DatasetCategories(init_categories=_INIT_CATEGORIES)

    def _builder(self) -> IIITar13KBuilder:
        return IIITar13KBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class IIITar13KBuilder(DataFlowBaseBuilder):
    """
    IIITar13K dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the return
        values of the dataflow:

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
            raise ModuleNotFoundError("IIITar13KBuilder.build requires lxml but it is not installed.")

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
            dp["json"]["filename"] = dp["file_name"]
            return dp["json"]

        df = MapData(df, _map_file_name)
        df = MapData(
            df,
            pascal_voc_dict_to_image(  # pylint: disable = E1120
                self.categories.get_categories(init=True, name_as_key=True),
                load_image,
                filter_empty_image=True,
                fake_score=fake_score,
                category_name_mapping={
                    "natural_image": LayoutType.FIGURE,
                    "figure": LayoutType.FIGURE,
                    "logo": LayoutType.LOGO,
                    "signature": LayoutType.SIGNATURE,
                    "table": LayoutType.FIGURE,
                },
            ),
        )

        return df
