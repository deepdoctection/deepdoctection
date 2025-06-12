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

    DocLayNet_core
    ├── COCO
    │ ├── test.json
    │ ├── val.json
    ├── PNG
    │ ├── 0a0d43e301facee9e99cc33b9b16e732dd207135f4027e75f6aea2bf117535a2.png
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence, Union

from ...dataflow import DataFlow, MapData, MapDataComponent, SerializerCoco
from ...datapoint.annotation import CategoryAnnotation
from ...datapoint.image import Image
from ...mapper.cats import add_summary, cat_to_sub_cat, filter_cat, filter_summary
from ...mapper.cocostruct import coco_to_image
from ...mapper.maputils import curry
from ...utils.fs import load_image_from_file
from ...utils.settings import DatasetType, DocumentType, LayoutType, ObjectTypes, PageType, SummaryType, TypeOrStr
from ...utils.types import CocoDatapointDict
from ..base import DatasetBase
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories, DatasetInfo
from ..registry import dataset_registry

_NAME = "doclaynet"
_SHORT_DESCRIPTION = "DocLayNet is a human-annotated document layout segmentation dataset containing 80863 pages."
_DESCRIPTION = (
    "DocLayNet is a human-annotated document layout segmentation dataset containing 80863 pages from a broad "
    "variety of document sources. \n"
    "DocLayNet provides page-by-page layout segmentation ground-truth using bounding-boxes for 11 distinct class"
    " labels on 80863 unique pages from 6 document categories. It provides several unique features compared"
    " to related work such as PubLayNet or DocBank: \n"
    "Human Annotation: DocLayNet is hand-annotated by well-trained experts, providing a gold-standard in layout"
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
_TYPE = DatasetType.OBJECT_DETECTION

_LOCATION = "DocLayNet_core"

_ANNOTATION_FILES: Mapping[str, str] = {"train": "COCO/train.json", "val": "COCO/val.json", "test": "COCO/test.json"}
_INIT_CATEGORIES = [
    LayoutType.CAPTION,
    LayoutType.FOOTNOTE,
    LayoutType.FORMULA,
    LayoutType.LIST,
    LayoutType.PAGE_FOOTER,
    LayoutType.PAGE_HEADER,
    LayoutType.FIGURE,
    LayoutType.SECTION_HEADER,
    LayoutType.TABLE,
    LayoutType.TEXT,
    LayoutType.TITLE,
]
_SUB_CATEGORIES: Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]] = {
    LayoutType.CAPTION: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.FOOTNOTE: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.FORMULA: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.LIST: {DatasetType.PUBLAYNET: [LayoutType.LIST]},
    LayoutType.PAGE_FOOTER: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.PAGE_HEADER: {DatasetType.PUBLAYNET: [LayoutType.TITLE]},
    LayoutType.FIGURE: {DatasetType.PUBLAYNET: [LayoutType.FIGURE]},
    LayoutType.SECTION_HEADER: {DatasetType.PUBLAYNET: [LayoutType.TITLE]},
    LayoutType.TABLE: {DatasetType.PUBLAYNET: [LayoutType.TABLE]},
    LayoutType.TEXT: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.TITLE: {DatasetType.PUBLAYNET: [LayoutType.TITLE]},
}


@dataset_registry.register("doclaynet")
class DocLayNet(DatasetBase):
    """
    `DocLayNet`
    """

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(
            name=_NAME,
            short_description=_SHORT_DESCRIPTION,
            description=_DESCRIPTION,
            license=_LICENSE,
            splits=_SPLITS,
            type=_TYPE,
            url=_URL,
        )

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES, init_sub_categories=_SUB_CATEGORIES)

    def _builder(self) -> DocLayNetBuilder:
        return DocLayNetBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class DocLayNetBuilder(DataFlowBaseBuilder):
    """
    `DocLayNetBuilder` dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        Args:
            kwargs: (split) Split of the dataset. Can be `train`,`val` or `test`. Default: `val`
                    (max_datapoints) Will stop iterating after `max_datapoints`. Default: `None`
                    (load_image) Will load the image for each datapoint.  Default: `False`
                    (fake_score) Will add a fake score so that annotations look like predictions

        Returns:
            Dataflow
        """
        split = str(kwargs.get("split", "val"))
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        load_image = kwargs.get("load_image", False)
        fake_score = kwargs.get("fake_score", False)
        resized = kwargs.get("resized", False)

        # Load
        df: DataFlow
        path = self.get_workdir() / self.get_annotation_file(split)
        df = SerializerCoco.load(path, max_datapoints=max_datapoints)

        # Map
        if resized:
            png_folder = "PNG_resized"
        else:
            png_folder = "PNG"

        df = MapDataComponent(df, lambda dp: self.get_workdir() / png_folder / dp, "file_name")
        df = MapData(
            df,
            coco_to_image(
                self.categories.get_categories(init=True),
                load_image,
                filter_empty_image=True,
                fake_score=fake_score,
                coarse_mapping={1: 10, 2: 10, 3: 10, 4: 4, 5: 10, 6: 11, 7: 7, 8: 11, 9: 9, 10: 10, 11: 11},
                coarse_sub_cat_name=DatasetType.PUBLAYNET,
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
        df = MapData(df, add_summary(self.categories.get_categories(filtered=True)))
        return df


_NAME_SEQ = "doclaynet-seq"
_TYPE_SEQ = DatasetType.SEQUENCE_CLASSIFICATION
_INIT_CATEGORIES_SEQ = [
    DocumentType.FINANCIAL_REPORT,
    DocumentType.SCIENTIFIC_PUBLICATION,
    DocumentType.LAWS_AND_REGULATIONS,
    DocumentType.GOVERNMENT_TENDERS,
    DocumentType.MANUALS,
    DocumentType.PATENTS,
]


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

    def _builder(self) -> DocLayNetSeqBuilder:
        return DocLayNetSeqBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class DocLayNetSeqBuilder(DataFlowBaseBuilder):
    """
    DocLayNetSeqBuilder dataflow builder
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

        Returns:
            Dataflow
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
        def _map_to_image(dp: CocoDatapointDict, load_img: bool) -> Image:
            image = Image(location=dp["file_name"], file_name=os.path.split(dp["file_name"])[1])
            image.image = load_image_from_file(image.location)
            summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)
            label_to_category_name = {
                "financial_reports": DocumentType.FINANCIAL_REPORT,
                "scientific_articles": DocumentType.SCIENTIFIC_PUBLICATION,
                "laws_and_regulations": DocumentType.LAWS_AND_REGULATIONS,
                "government_tenders": DocumentType.GOVERNMENT_TENDERS,
                "manuals": DocumentType.MANUALS,
                "patents": DocumentType.PATENTS,
            }
            categories_dict = self.categories.get_categories(init=True, name_as_key=True)
            category_name = label_to_category_name[dp["doc_category"]]
            summary.dump_sub_category(
                PageType.DOCUMENT_TYPE,
                CategoryAnnotation(category_name=category_name, category_id=categories_dict[category_name]),
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
                filter_summary({PageType.DOCUMENT_TYPE: self.categories.get_categories(as_dict=False, filtered=True)}),
            )

            @curry
            def _re_map_cat_ids(dp: Image, filtered_categories_name_as_key: Mapping[TypeOrStr, int]) -> Image:
                if PageType.DOCUMENT_TYPE in dp.summary.sub_categories:
                    summary_cat = dp.summary.get_sub_category(PageType.DOCUMENT_TYPE)
                    summary_cat.category_id = filtered_categories_name_as_key[summary_cat.category_name]
                return dp

            df = MapData(df, _re_map_cat_ids(self.categories.get_categories(filtered=True, name_as_key=True)))

        return df
