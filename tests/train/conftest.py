# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixture module for train
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import numpy as np
from pytest import fixture

from deepdoctection.dataflow import DataFlow, DataFromList  # type: ignore
from deepdoctection.datapoint import BoundingBox, Image, ImageAnnotation
from deepdoctection.datasets.base import DatasetBase
from deepdoctection.datasets.dataflow_builder import DataFlowBaseBuilder
from deepdoctection.datasets.info import DatasetCategories, DatasetInfo

_NAME = "test_set"
_LOCATION = "/path/to/dir"
_INIT_CATEGORIES = ["TABLE", "TEXT"]
_DESCRIPTION = "test description"
_URL = "test url"
_LICENSE = "test license"


@dataclass
class Datapoint:
    """
    A dataclass for generating an Image datapoint with annotations
    """

    image = Image(location="/test/to/path", file_name="test_name")
    np_image = np.ones([400, 600, 3], dtype=np.float32)
    anns = [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=100.0, lrx=200.0, lry=400.0, absolute_coords=True),
            category_name="TABLE",
            category_id="1",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=50.0, uly=70.0, lrx=70.0, lry=90.0, absolute_coords=True),
            category_name="TEXT",
            category_id="2",
        ),
    ]

    def get_datapoint(self) -> Image:
        """
        datapoint
        """
        image = deepcopy(self.image)
        image.image = self.np_image
        for ann in self.anns:
            image.dump(ann)
        return image


class TestDataset(DatasetBase):
    """
    Class for a dataset fixture
    """

    def _info(self) -> DatasetInfo:
        """
        _info
        """
        return DatasetInfo(name=_NAME, description=_DESCRIPTION, url=_URL, license=_LICENSE)

    def _categories(self) -> DatasetCategories:
        """
        _categories
        """
        return DatasetCategories(init_categories=_INIT_CATEGORIES)

    def _builder(self) -> "TestBuilder":
        """
        _builder
        """
        return TestBuilder(location=_LOCATION)


class TestBuilder(DataFlowBaseBuilder):
    """
    test dataflow builder
    """

    def build(self, **kwargs: Union[str, int]) -> DataFlow:
        """
        build
        """
        return DataFromList([Datapoint().get_datapoint(), Datapoint().get_datapoint()])


@fixture(name="test_dataset")
def fixture_test_dataset() -> DatasetBase:
    """
    fixture for test dataset
    """
    return TestDataset()
