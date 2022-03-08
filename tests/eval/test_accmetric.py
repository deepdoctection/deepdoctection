# -*- coding: utf-8 -*-
# File: xxx.py

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
Testing module eval.accmetric
"""

from deepdoctection.dataflow import DataFromList  # type: ignore
from deepdoctection.datapoint.image import Image
from deepdoctection.datasets.info import DatasetCategories
from deepdoctection.eval.accmetric import AccuracyMetric
from deepdoctection.utils.settings import names


class TestAccuracyMetric:
    """
    Test AccMetric returns correct when evaluating gt against itself
    """

    @staticmethod
    def test_accuracy_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        when testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = AccuracyMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert output[0] == {"key": names.C.TAB, "val": 1.0, "num_samples": 1}
        assert output[1] == {"key": names.C.CELL, "val": 1.0, "num_samples": 5}
        assert output[2] == {"key": names.C.ROW, "val": 1.0, "num_samples": 2}
        assert output[3] == {"key": names.C.COL, "val": 1.0, "num_samples": 2}

        # Clean-up
        AccuracyMetric._cats = None  # pylint: disable=W0212
        AccuracyMetric._sub_cats = None  # pylint: disable=W0212

    @staticmethod
    def test_accuracy_metric_for_sub_cat_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        when testing datapoint against itself for sub categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)
        AccuracyMetric.set_categories(sub_category_names={names.C.CELL: [names.C.RN, names.C.CS]})

        # Arrange
        output = AccuracyMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)
        # Assert
        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0] == {"key": names.C.RN, "val": 1.0, "num_samples": 5}
        assert output[1] == {"key": names.C.CS, "val": 1.0, "num_samples": 5}

    @staticmethod
    def test_accuracy_metric_for_sub_cat_returns_correct_distance_as_dict(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        when testing datapoint against itself for sub categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)
        AccuracyMetric.set_categories(sub_category_names={names.C.CELL: [names.C.RN, names.C.CS]})

        # Arrange
        output = AccuracyMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories, True)

        # Assert
        assert output == {"key/ROW_NUMBER/num_samples/5": 1.0, "key/COLUMN_SPAN/num_samples/5": 1.0}
