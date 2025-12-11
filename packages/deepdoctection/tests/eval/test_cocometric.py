# -*- coding: utf-8 -*-
# File: test_cocometric.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dd_core.dataflow import DataFromList
from dd_core.datapoint import BoundingBox, ImageAnnotation
from dd_core.utils.object_types import get_type
from deepdoctection.eval.cocometric import CocoMetric

try:
    from dd_datasets.base import DatasetCategories
except ImportError:
    DatasetCategories = None


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestCocoMetric:
    """
    Test CocoMetric returns correct result when evaluating gt against itself
    """

    @pytest.fixture(autouse=True)
    def _setup(self, dp_image):
        dp_image = deepcopy(dp_image)
        box = BoundingBox(ulx=2.6, uly=3.7, lrx=4.6, lry=5.7, absolute_coords=True)
        ann = ImageAnnotation(category_name="test_cat_1", bounding_box=box, score=0.53, category_id=1)
        dp_image.dump(ann)
        box = BoundingBox(ulx=16.6, uly=26.6, height=4.0, width=14.0, absolute_coords=True)
        ann = ImageAnnotation(category_name="test_cat_2", bounding_box=box, score=0.99, category_id=2)
        dp_image.dump(ann)
        dp_list = [dp_image]
        self.dataflow_gt = DataFromList(dp_list)
        self.dataflow_pr = DataFromList(dp_list)
        self.categories = DatasetCategories(init_categories=[get_type("test_cat_1"), get_type("test_cat_2")])

    def test_coco_metric_returns_correct_distance(self) -> None:
        """
        when testing datapoint against itself, evaluation returns full score except when some areas do not exist
        """

        # Act
        output = CocoMetric.get_distance(self.dataflow_gt, self.dataflow_pr, self.categories)

        # Assert
        output_list = []
        for res in output:
            output_list.append(res["val"])
        output = np.asarray(output_list)  # type: ignore

        expected_output = np.asarray([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1])

        assert_allclose(output, expected_output, atol=1e-10)  # type: ignore

    def test_when_params_change_coco_metric_returns_correct_distance(self) -> None:
        """
        when parameters are changed then coco metric return correct distance
        """

        # Arrange
        CocoMetric.set_params(
            area_range=[[0**2, 1e5**2], [0**2, 1**2], [1**2, 100**2], [100**2, 1e5**2]],  # type: ignore
            max_detections=[1, 2, 3],
        )

        # Act
        output = CocoMetric.get_distance(self.dataflow_gt, self.dataflow_pr, self.categories)

        # Assert
        output_list = []
        for res in output:
            output_list.append(res["val"])
        output = np.asarray(output_list)  # type: ignore

        expected_output = np.asarray([-1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1])

        assert_allclose(output, expected_output, atol=1e-10)  # type: ignore

        # Clean-up
        CocoMetric._params = {}  # pylint: disable=W0212

    def test_when_f1_score_coco_metric_returns_correct_distance(self) -> None:
        """
        when f1_score = True is set then coco metric returns correct distance
        """

        # Arrange
        CocoMetric.set_params(f1_score=True)

        # Act
        output = CocoMetric.get_distance(self.dataflow_gt, self.dataflow_pr, self.categories)

        # Assert
        output_list = []
        for res in output:
            output_list.append(res["val"])
            output = np.asarray(output_list)  # type: ignore

        expected_output = np.asarray([-1, -1])

        assert_allclose(output, expected_output, atol=1e-10)  # type: ignore

        # Clean-up
        CocoMetric._f1_score = None  # pylint: disable=W0212
        CocoMetric._f1_iou = None  # pylint: disable=W0212
