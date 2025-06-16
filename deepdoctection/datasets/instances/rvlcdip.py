# -*- coding: utf-8 -*-
# File: rvldcip.py

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

    rvl-cdip
    ├── images
    │ ├── imagesa
    │ ├── ...
    │ ├── imagesa
    ├── label
    │ ├── test.txt
    │ ├── train.txt
    │ ├── val.txt
"""
from __future__ import annotations

import os
from typing import Mapping, Union

from ...dataflow import DataFlow, MapData
from ...dataflow.custom_serialize import SerializerTabsepFiles
from ...datapoint.annotation import CategoryAnnotation
from ...datapoint.image import Image
from ...mapper.cats import filter_summary
from ...mapper.maputils import curry
from ...utils.fs import load_image_from_file
from ...utils.settings import DatasetType, DocumentType, PageType, SummaryType, TypeOrStr
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories, DatasetInfo
from ..registry import dataset_registry

_NAME = "rvl-cdip"
_SHORT_DESCRIPTION = "RVL-CDIP is a dataset for document classification."
_DESCRIPTION = (
    "The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400, 000 gray- \n"
    "scale images in 16 classes, with 25, 000 images per class . There are 320, 000 training images, 40, 000  \n"
    "validation images, and 40, 000 test images.The images are sized so their largest dimension does not exceed 1000 \n"
    "pixels. This dataset is a subset of the IIT-CDIP Test Collection 1.0 [1], which is publicly available here. The \n"
    "file structure of this dataset is the same as in the IIT collection, so it is possible to refer to that dataset \n"
    "for OCR and additional metadata. The IIT-CDIP dataset is itself a subset of the Legacy Tobacco Document Library"
)
_LICENSE = (
    "RVL-CDIP is a subset of IIT-CDIP, which came from the Legacy Tobacco Document Library, for which \n"
    "license information can be found at https://www.industrydocuments.ucsf.edu/help/copyright/ ."
)

_URL = "https://www.cs.cmu.edu/~aharley/rvl-cdip/"

_SPLITS: Mapping[str, str] = {"train": "train", "val": "val", "test": "test"}
_TYPE = DatasetType.SEQUENCE_CLASSIFICATION
_LOCATION = "rvl-cdip"

_ANNOTATION_FILES: Mapping[str, str] = {"train": "labels/train.txt", "val": "labels/val.txt", "test": "labels/test.txt"}
_INIT_CATEGORIES = [
    DocumentType.LETTER,
    DocumentType.FORM,
    DocumentType.EMAIL,
    DocumentType.HANDWRITTEN,
    DocumentType.ADVERTISEMENT,
    DocumentType.SCIENTIFIC_REPORT,
    DocumentType.SCIENTIFIC_PUBLICATION,
    DocumentType.SPECIFICATION,
    DocumentType.FILE_FOLDER,
    DocumentType.NEWS_ARTICLE,
    DocumentType.BUDGET,
    DocumentType.INVOICE,
    DocumentType.PRESENTATION,
    DocumentType.QUESTIONNAIRE,
    DocumentType.RESUME,
    DocumentType.MEMO,
]


@dataset_registry.register("rvl-cdip")
class Rvlcdip(_BuiltInDataset):
    """
    RVLCDIP
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

    def _builder(self) -> RvlcdipBuilder:
        return RvlcdipBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class RvlcdipBuilder(DataFlowBaseBuilder):
    """
    Rvlcdip dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images.

        Args:
            kwargs:
                split (str): Split of the dataset. Can be `train`, `val` or `test`. Default: `val`
                max_datapoints (int): Will stop iterating after max_datapoints. Default: `None`
                load_image (bool): Will load the image for each datapoint. Default: `False`

        Returns:
            Dataflow
        """

        split = str(kwargs.get("split", "val"))
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        load_image = kwargs.get("load_image", False)

        # Load
        df: DataFlow
        path = self.get_workdir() / self.get_annotation_file(split)
        df = SerializerTabsepFiles.load(path, max_datapoints)

        @curry
        def _map_str_to_image(dp: str, load_img: bool) -> Image:
            location, label_str = dp.split()[0], dp.split()[1]
            label = int(label_str) + 1
            file_name = os.path.split(location)[1]
            image = Image(location=(self.get_workdir() / "images" / location).as_posix(), file_name=file_name)
            image.image = load_image_from_file(image.location)
            summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)
            categories_dict = self.categories.get_categories(init=True)
            summary.dump_sub_category(
                PageType.DOCUMENT_TYPE, CategoryAnnotation(category_name=categories_dict[label], category_id=label)
            )
            image.summary = summary
            if not load_img:
                image.clear_image()
            return image

        df = MapData(df, _map_str_to_image(load_image))  # pylint: disable=E1120

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
