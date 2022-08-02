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

|    rvl-cdip
|    ├── images
|    │ ├── imagesa
|    │ ├── ...
|    │ ├── imagesa
|    ├── label
|    │ ├── test.txt
|    │ ├── train.txt
|    │ ├── val.txt
"""

import os

from typing import Mapping, Union

from ...utils.settings import names
from ...utils.fs import load_image_from_file
from ...dataflow.custom_serialize import SerializerTabsepFiles
from ...dataflow import MapData, DataFlow
from ...datapoint.image import Image
from ...datapoint.annotation import SummaryAnnotation, CategoryAnnotation
from ...mapper.maputils import curry
from ...mapper.cats import filter_summary
from ..registry import dataset_registry
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories, DatasetInfo


_NAME = "rvl-cdip"
_DESCRIPTION = (
    "The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400, 000 grayscale \n"
    "images in 16 classes, with 25, 000 images per class . There are 320, 000 training images, 40, 000 validation \n"
    " images, and 40, 000 test images.The images are sized so their largest dimension does not exceed 1000 pixels. \n" 
    "This dataset is a subset of the IIT-CDIP Test Collection 1.0 [1], which is publicly available here. The file \n"
    "structure of this dataset is the same as in the IIT collection, so it is possible to refer to that dataset for \n"
    " OCR and additional metadata. The IIT-CDIP dataset is itself a subset of the Legacy Tobacco Document Library"
)
_LICENSE = ("RVL-CDIP is a subset of IIT-CDIP, which came from the Legacy Tobacco Document Library, for which \n"
            "license information can be found at https://www.industrydocuments.ucsf.edu/help/copyright/ .")

_URL = (
    "https://www.cs.cmu.edu/~aharley/rvl-cdip/"
)

_SPLITS: Mapping[str, str] = {"train": "train", "val": "val", "test": "test"}
_TYPE = names.DS.TYPE.SEQ
_LOCATION = "rvl-cdip"

_ANNOTATION_FILES: Mapping[str, str] = {"train": "labels/train.txt", "val": "labels/val.txt", "test": "labels/test.txt"}
_INIT_CATEGORIES = [names.C.LET, names.C.FORM, names.C.EM, names.C.HW, names.C.AD, names.C.SR, names.C.SP,
                    names.C.SPEC,names.C.FF, names.C.NA,names.C.BU,names.C.INV,names.C.PRES,names.C.QUEST,
                    names.C.RES,names.C.MEM]


@dataset_registry.register("rvl-cdip")
class Rvlcdip(_BuiltInDataset):
    """
    RVLCDIP
    """

    _name = _NAME

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, license=_LICENSE, url=_URL, splits=_SPLITS,type=_TYPE)

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES)

    def _builder(self) -> "RvlcdipBuilder":
        return RvlcdipBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class RvlcdipBuilder(DataFlowBaseBuilder):
    """
    Rvlcdip dataflow builder
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
        assert isinstance(dataset_split, str)
        path = self.get_workdir() / dataset_split
        df = SerializerTabsepFiles.load(path, max_datapoints)

        @curry
        def _map_str_to_image(dp: str, load_image: bool) -> Image:
            location, label = dp.split()[0], dp.split()[1]
            label = str(int(label) + 1)
            file_name = os.path.split(location)[1]
            image = Image(location=(self.get_workdir()/"images"/location).as_posix(),file_name=file_name)
            image.image = load_image_from_file(image.location)
            summary = SummaryAnnotation()
            categories_dict = self.categories.get_categories(init=True)
            summary.dump_sub_category(names.C.DOC,
                                      CategoryAnnotation(category_name=categories_dict[label], category_id=str(label)))
            image.summary = summary
            if not load_image:
                image.clear_image()
            return image

        df = MapData(df,_map_str_to_image(load_image))

        assert self.categories is not None
        if self.categories.is_filtered():
            df = MapData(
                df, filter_summary({names.C.DOC: self.categories.get_categories(as_dict=False, filtered=True)}))

        return df





