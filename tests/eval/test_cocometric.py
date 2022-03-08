# -*- coding: utf-8 -*-
# File: test_cocometric.py

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
Testing the module eval.cocometric
"""
import numpy as np
from numpy.testing import assert_allclose

from deepdoctection.dataflow import DataFromList  # type: ignore
from deepdoctection.eval.cocometric import CocoMetric

from ..conftest import get_image_results


class TestCocoMetric:
    """
    Test CocoMetric returns correct result when evaluating gt against itself
    """

    def setup(self) -> None:
        """
        setup necessary components
        """

        dp_list = [get_image_results().image]
        self.dataflow_gt = DataFromList(dp_list)
        self.dataflow_pr = DataFromList(dp_list)
        self.categories = get_image_results().get_dataset_categories()

    def test_coco_metric_returns_correct_distance(self) -> None:
        """
        when testing datapoint against itself, evaluation returns full score except when some areas do not exist
        """

        self.setup()

        # Act
        output = CocoMetric.get_distance(self.dataflow_gt, self.dataflow_pr, self.categories)

        # Assert
        output_list = []
        for res in output:
            output_list.append(res["val"])  # type: ignore
        output = np.asarray(output_list)  # type: ignore

        expected_output = np.asarray([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1])

        assert_allclose(output, expected_output, atol=1e-10)  # type: ignore

    def test_when_params_change_coco_metric_returns_correct_distance(self) -> None:
        """
        when parameters are changed then coco metric return correct distance
        """

        self.setup()

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
            output_list.append(res["val"])  # type: ignore
        output = np.asarray(output_list)  # type: ignore

        expected_output = np.asarray([-1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1])

        assert_allclose(output, expected_output, atol=1e-10)  # type: ignore

        # Clean-up
        CocoMetric._params = {}  # pylint: disable=W0212

    def test_when_f1_score_coco_metric_returns_correct_distance(self) -> None:
        """
        when f1_score = True is set then coco metric returns correct distance
        """
        self.setup()

        # Arrange
        CocoMetric.set_params(f1_score=True)

        # Act
        output = CocoMetric.get_distance(self.dataflow_gt, self.dataflow_pr, self.categories)

        # Assert
        output_list = []
        for res in output:
            output_list.append(res["val"])  # type: ignore
            output = np.asarray(output_list)  # type: ignore

        expected_output = np.asarray([-1, -1])

        assert_allclose(output, expected_output, atol=1e-10)  # type: ignore

        # Clean-up
        CocoMetric._f1_score = None  # pylint: disable=W0212
        CocoMetric._f1_iou = None  # pylint: disable=W0212
