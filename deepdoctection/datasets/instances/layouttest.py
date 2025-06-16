# -*- coding: utf-8 -*-
# File: layouttest.py

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
Module for Testlayout dataset. Install the dataset following the folder structure

    testlayout
    ├── predict
    │ ├── xrf_layout_test_predict.jsonl
    ├── test
    │ ├── xrf_layout_test.jsonl
"""
from __future__ import annotations

from typing import Mapping, Union

from ...dataflow import DataFlow, MapData
from ...dataflow.custom_serialize import SerializerJsonlines
from ...datasets.info import DatasetInfo
from ...mapper.prodigystruct import prodigy_to_image
from ...utils.settings import DatasetType, LayoutType
from ..base import _BuiltInDataset
from ..dataflow_builder import DataFlowBaseBuilder
from ..info import DatasetCategories
from ..registry import dataset_registry

_NAME = "testlayout"
_DESCRIPTION = (
    "A small test set consisting of a subset of six images with layout annotations. \n"
    "The set is meant for debugging purposes and nothing else"
)

_LICENSE = (
    "The annotations in this dataset belong to Dr. Janis Meyer and are licensed \n"
    " under a Community Data License Agreement \n"
    " – Permissive – Version 1.0 License. Dr. Janis Meyer does not own the copyright of the images. \n"
    " Use of the images must abide by the PMC Open Access Subset Terms of Use."
)

_SPLITS: Mapping[str, str] = {"test": "test", "predict": "predict"}
_TYPE = DatasetType.OBJECT_DETECTION
_LOCATION = "testlayout"

_ANNOTATION_FILES: Mapping[str, str] = {
    "test": "xrf_layout_test.jsonl",
    "predict": "xrf_layout_test_predict.jsonl",
}

_INIT_CATEGORIES = [LayoutType.TEXT, LayoutType.TITLE, LayoutType.LIST, LayoutType.TABLE, LayoutType.FIGURE]


@dataset_registry.register("testlayout")
class LayoutTest(_BuiltInDataset):
    """
    `LayoutTest`
    """

    _name = _NAME

    @classmethod
    def _info(cls) -> DatasetInfo:
        return DatasetInfo(
            name=_NAME,
            short_description=_DESCRIPTION,
            description=_DESCRIPTION,
            license=_LICENSE,
            splits=_SPLITS,
            type=_TYPE,
        )

    def _categories(self) -> DatasetCategories:
        return DatasetCategories(init_categories=_INIT_CATEGORIES)

    def _builder(self) -> LayoutTestBuilder:
        return LayoutTestBuilder(location=_LOCATION, annotation_files=_ANNOTATION_FILES)


class LayoutTestBuilder(DataFlowBaseBuilder):
    """
    LayoutTest dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        Returns a dataflow from which you can stream datapoints of images. The following arguments affect the returns
        of the dataflow:

        Args:
            kwargs:
                (split) Split of the dataset. Only `test` is available for this small sample. Default: `test`
                (max_datapoints) Will stop iterating after `max_datapoints`. Default: `None`
                (load_image) Will load the image for each datapoint. Default: `False`
                (fake_score) Will add a fake score so that annotations look like predictions. Default: `False`

        Returns:
            Dataflow
        """
        split = str(kwargs.get("split", "test"))
        max_datapoints = kwargs.get("max_datapoints")
        if max_datapoints is not None:
            max_datapoints = int(max_datapoints)
        load_image = kwargs.get("load_image", False)
        fake_score = kwargs.get("fake_score", False)

        # Load
        df: DataFlow
        path = self.get_workdir() / self.splits[split] / self.get_annotation_file(split)

        df = SerializerJsonlines.load(path, max_datapoints=max_datapoints)

        # Map
        df = MapData(df, lambda dp: dp if dp["answer"] == "accept" else None)
        prodigy_mapper = prodigy_to_image(  # pylint: disable=E1120
            categories_name_as_key=self.categories.get_categories(init=True, name_as_key=True),
            load_image=load_image,
            fake_score=fake_score,
        )
        df = MapData(df, prodigy_mapper)

        return df
