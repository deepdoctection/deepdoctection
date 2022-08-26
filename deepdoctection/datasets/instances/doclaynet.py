# -*- coding: utf-8 -*-
# File: doclaynet.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Module for DocLayNet dataset. Place the dataset as follows

|    DocLayNet_core
|    ├── COCO
|    │ ├── test.json
|    │ ├── val.json
|    ├── PNG
|    │ ├── 0a0d43e301facee9e99cc33b9b16e732dd207135f4027e75f6aea2bf117535a2.png
"""

import os
from typing import Mapping, Union

from ...dataflow import DataFlow, MapData, MapDataComponent, SerializerCoco
from ...datapoint.annotation import CategoryAnnotation, SummaryAnnotation
from ...datapoint.image import Image
from ...mapper.cats import cat_to_sub_cat, filter_cat, filter_summary
from ...mapper.cocostruct import coco_to_image
from ...mapper.maputils import curry
from ...utils.detection_types import JsonDict
from ...utils.fs import load_image_from_file
from ...utils.settings import names
from ..base import DatasetBase
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories, DatasetInfo
from ..registry import dataset_registry

_NAME = "doclaynet"
_DESCRIPTION = (
    "DocLayNet is a human-annotated document layout segmentation dataset containing 80863 pages from a broad "
    "variety of document sources. \n"
    "DocLayNet provides page-by-page layout segmentation ground-truth using bounding-boxes for 11 distinct class"
    " labels on 80863 unique pages from 6 document categories. It provides several unique features compared"
    " to related work such as PubLayNet or DocBank: \n"
    "Humman Annotation: DocLayNet is hand-annotated by well-trained experts, providing a gold-standard in layout"
    " segmentation through human recognition and interpretation of each page layout \n"
    "Large layout variability: DocLayNet includes diverse and complex layouts from a large variety of public"
    " sources in Finance, Science, Patents, Tenders, Law texts and Manuals \n"
    "Detailed label set: DocLayNet defines 11 class labels to distinguish layout features in high detail. \n"
    "Redundant annotations: A fraction of the pages in DocLayNet are double- or triple-annotated, allowing to"
    " estimate annotation uncertainty and an upper-bound of achievable prediction accuracy with ML models "
    "Pre-defined train- test- and validation-sets: DocLayNet provides fixed sets for each to ensure proportional"
    " representation of the class-labels and avoid leakage of unique layout styles across the sets."
)
_LICENSE = "CDLA-Permissive"
_URL = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
_SPLITS: Mapping[str, str] = {"train": "train", "val": "val", "test": "test"}
_TYPE = names.DS.TYPE.OBJ

_LOCATION = "DocLayNet_core"

_ANNOTATION_FILES: Mapping[str, str] = {"train": "COCO/train.json", "val": "COCO/val.json", "test": "COCO/test.json"}
_INIT_CATEGORIES = [
    names.C.CAP,
    names.C.FOOT,
    names.C.FORMULA,
    names.C.LIST,
    names.C.PFOOT,
    names.C.PHEAD,
    names.C.FIG,
    names.C.SECH,
    names.C.TAB,
    names.C.TEXT,
    names.C.TITLE,
]
_SUB_CATEGORIES = {
    names.C.CAP: {"publaynet": [names.C.TEXT]},
    names.C.FOOT: {"publaynet": [names.C.TEXT]},
    names.C.FORMULA: {"publaynet": [names.C.TEXT]},
    names.C.LIST: {"publaynet": [names.C.LIST]},
    names.C.PFOOT: {"publaynet": [names.C.TEXT]},
    names.C.PHEAD: {"publaynet": [names.C.TITLE]},
    names.C.FIG: {"publaynet": [names.C.FIG]},
    names.C.SECH: {"publaynet": [names.C.TITLE]},
    names.C.TAB: {"publaynet": [names.C.TAB]},
    names.C.TEXT: {"publaynet": [names.C.TEXT]},
    names.C.TITLE: {"publaynet": [names.C.TITLE]},
}


@dataset_registry.register("doclaynet")
class DocLayNet(DatasetBase):
    """
    DocLayNetSeq
    """

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, splits=_SPLITS, type=_TYPE)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> "DocLayNetBuilder":
        return DocLayNetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class DocLayNetBuilder(DataFlowBaseBuilder):
    """
    DocLayNetBuilder dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
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
        df: DataFlow
        dataset_split = self.annotation_files[split]
        path = self.get_workdir() / dataset_split  # type: ignore
        df = SerializerCoco.load(path, max_datapoints=max_datapoints)

        # Map
        df = MapDataComponent(df, lambda dp: self.get_workdir() / "PNG" / dp, "file_name")
        df = MapData(
            df,
            coco_to_image(
                self.categories.get_categories(init=True),
                load_image,
                filter_empty_image=True,
                fake_score=fake_score,
                coarse_mapping={1: 10, 2: 10, 3: 10, 4: 4, 5: 10, 6: 11, 7: 7, 8: 11, 9: 9, 10: 10, 11: 11},
                coarse_sub_cat_name="publaynet",
            ),
        )

        if self.categories.is_cat_to_sub_cat():
            df = MapData(
                df,
                cat_to_sub_cat(self.categories.get_categories(name_as_key=True), self.categories.cat_to_sub_cat),
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


_NAME_SEQ = "doclaynet-seq"
_TYPE_SEQ = names.DS.TYPE.SEQ
_INIT_CATEGORIES_SEQ = [names.C.FR, names.C.SP, names.C.LR, names.C.GT, names.C.MAN, names.C.PAT]


@dataset_registry.register("doclaynet-seq")
class DocLayNetSeq(DatasetBase):
    """
    DocLayNetSeq is the DocLayNet dataset where the dataflow has been prepared to perform sequence classification
    """

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(name=_NAME_SEQ, description=_DESCRIPTION, license=_LICENSE, splits=_SPLITS, type=_TYPE_SEQ)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES_SEQ)

    def _builder(self) -> "DocLayNetSeqBuilder":
        return DocLayNetSeqBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class DocLayNetSeqBuilder(DataFlowBaseBuilder):
    """
    DocLayNetSeqBuilder dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        :param split: Split of the dataset. Can be "train","val" or "test". Default: "val"
        :param max_datapoints: Will stop iterating after max_datapoints. Default: None
        :param load_image: Will load the image for each datapoint.  Default: False

        :return: dataflow
        """
        split = str(kwargs.get("split", "val"))
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        load_image = kwargs.get("load_image", False)

        # Load
        dataset_split = self.annotation_files[split]
        path = self.get_workdir() / dataset_split  # type: ignore
        df = SerializerCoco.load(path, max_datapoints=max_datapoints)

        # Map
        df = MapDataComponent(df, lambda dp: self.get_workdir() / "PNG" / dp, "file_name")

        @curry
        def _map_to_image(dp: JsonDict, load_img: bool) -> Image:
            image = Image(location=dp["file_name"], file_name=os.path.split(dp["file_name"])[1])
            image.image = load_image_from_file(image.location)
            summary = SummaryAnnotation()
            label_to_category_name = {
                "financial_reports": names.C.FR,
                "scientific_articles": names.C.SP,
                "laws_and_regulations": names.C.LR,
                "government_tenders": names.C.GT,
                "manuals": names.C.MAN,
                "patents": names.C.PAT,
            }
            categories_dict = self.categories.get_categories(init=True, name_as_key=True)
            category_name = label_to_category_name[dp["doc_category"]]
            summary.dump_sub_category(
                names.C.DOC, CategoryAnnotation(category_name=category_name, category_id=categories_dict[category_name])
            )
            image.summary = summary
            if not load_img:
                image.clear_image()
            return image

        df = MapData(df, _map_to_image(load_image))

        assert self.categories is not None
        if self.categories.is_filtered():
            df = MapData(
                df,
                filter_summary({names.C.DOC: self.categories.get_categories(as_dict=False, filtered=True)}),
            )

        return df
