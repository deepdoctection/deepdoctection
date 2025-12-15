# -*- coding: utf-8 -*-
# File: test_accmetric.py

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

"""
This module contains unit tests for the evaluation metrics used in the deepdoctection framework. The metrics
are tested for their correctness in evaluating ground truth data against predictions.

The tests include evaluation for accuracy, confusion matrices, precision, and recall metrics under various
contexts such as categories and subcategories. Dummy test cases verify trivial correctness of the implemented
metric definitions.
"""

from __future__ import annotations

import pytest

from dd_core.dataflow import DataFromList
from dd_core.datapoint.image import Image
from dd_core.utils.object_types import CellType
from deepdoctection.eval.accmetric import (
    AccuracyMetric,
    ConfusionMetric,
    F1Metric,
    F1MetricMicro,
    PrecisionMetric,
    PrecisionMetricMicro,
    RecallMetric,
    RecallMetricMicro,
)

try:
    from dd_datasets.info import DatasetCategories
except ImportError:
    DatasetCategories = None  # type: ignore


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestAccuracyMetric:
    """
    Test AccMetric returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_accuracy_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = AccuracyMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert output[0] == {"key": "table", "val": 1.0, "num_samples": 1}
        assert output[1] == {"key": "cell", "val": 1.0, "num_samples": 5}
        assert output[2] == {"key": "row", "val": 1.0, "num_samples": 2}
        assert output[3] == {"key": "column", "val": 1.0, "num_samples": 2}

        # Clean-up
        AccuracyMetric._cats = None  # pylint: disable=W0212
        AccuracyMetric._sub_cats = None  # pylint: disable=W0212

    @staticmethod
    def test_accuracy_metric_for_sub_cat_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for sub categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)
        accuracy_metric = AccuracyMetric()
        accuracy_metric.set_categories(sub_category_names={"cell": [CellType.ROW_NUMBER, CellType.COLUMN_SPAN]})

        # Arrange
        output = accuracy_metric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)
        # Assert
        assert len(output) == 2
        assert output[0] == {"key": "row_number", "val": 1.0, "num_samples": 5}
        assert output[1] == {"key": "column_span", "val": 1.0, "num_samples": 5}

        # Clean-up
        AccuracyMetric._cats = None  # pylint: disable=W0212
        AccuracyMetric._sub_cats = None  # pylint: disable=W0212

    @staticmethod
    def test_accuracy_metric_for_sub_cat_returns_correct_distance_as_dict(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for sub categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)
        accuracy_metric = AccuracyMetric()
        accuracy_metric.set_categories(sub_category_names={"cell": ["row_number", "column_span"]})

        # Arrange
        result = accuracy_metric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)
        output = accuracy_metric.result_list_to_dict(result)

        # Assert
        assert output == {"column_span/num_samples/5": 1.0, "row_number/num_samples/5": 1.0}


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestConfusionMetric:
    """
    Test ConfusionMetric returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_confusion_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = ConfusionMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 98
        assert output[3] == {
            "key": "table",
            "category_id": 2,
            "category_id_pr": 2,
            "val": 1.0,
            "num_samples": 1,
        }
        assert output[12] == {
            "key": "cell",
            "category_id": 3,
            "category_id_pr": 3,
            "val": 5.0,
            "num_samples": 5,
        }
        assert output[48] == {
            "key": "row",
            "category_id": 6,
            "category_id_pr": 6,
            "val": 2.0,
            "num_samples": 2,
        }
        assert output[97] == {
            "key": "column",
            "category_id": 7,
            "category_id_pr": 7,
            "val": 2.0,
            "num_samples": 2,
        }


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestPrecisionMetric:
    """
    Test PrecisionMetric returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_precision_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = PrecisionMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 18
        assert output[1] == {"key": "table", "category_id": 2, "val": 1.0, "num_samples": 1}
        assert output[4] == {"key": "cell", "category_id": 3, "val": 1.0, "num_samples": 5}
        assert output[10] == {"key": "row", "category_id": 6, "val": 1.0, "num_samples": 2}
        assert output[17] == {"key": "column", "category_id": 7, "val": 1.0, "num_samples": 2}


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestRecallMetric:
    """
    Test RecallMetric returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_recall_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = RecallMetric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 18
        assert output[1] == {"key": "table", "category_id": 2, "val": 1.0, "num_samples": 1}
        assert output[4] == {"key": "cell", "category_id": 3, "val": 1.0, "num_samples": 5}
        assert output[10] == {"key": "row", "category_id": 6, "val": 1.0, "num_samples": 2}
        assert output[17] == {"key": "column", "category_id": 7, "val": 1.0, "num_samples": 2}


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestF1Metric:
    """
    Test F1Metric returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_f1_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = F1Metric.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 18
        assert output[1] == {"key": "table", "category_id": 2, "val": 1.0, "num_samples": 1}
        assert output[4] == {"key": "cell", "category_id": 3, "val": 1.0, "num_samples": 5}
        assert output[10] == {"key": "row", "category_id": 6, "val": 1.0, "num_samples": 2}
        assert output[17] == {"key": "column", "category_id": 7, "val": 1.0, "num_samples": 2}


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestPrecisionMetricMicro:
    """
    Test PrecisionMetricMicro returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_precision_micro_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = PrecisionMetricMicro.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 4
        assert output[0] == {"key": "table", "val": 1.0, "num_samples": 1}
        assert output[1] == {"key": "cell", "val": 1.0, "num_samples": 5}
        assert output[2] == {"key": "row", "val": 1.0, "num_samples": 2}
        assert output[3] == {"key": "column", "val": 1.0, "num_samples": 2}


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestRecallMetricMicro:
    """
    Test RecallMetricMicro returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_recall_micro_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = RecallMetricMicro.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 4
        assert output[0] == {"key": "table", "val": 1.0, "num_samples": 1}
        assert output[1] == {"key": "cell", "val": 1.0, "num_samples": 5}
        assert output[2] == {"key": "row", "val": 1.0, "num_samples": 2}
        assert output[3] == {"key": "column", "val": 1.0, "num_samples": 2}


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestF1MetricMicro:
    """
    Test F1MetricMicro returns correct values when evaluating gt against itself
    """

    @staticmethod
    def test_f1_micro_metric_returns_correct_distance(
        dp_image_fully_segmented: Image, dataset_categories: DatasetCategories
    ) -> None:
        """
        When testing datapoint against itself for categories, evaluation returns full score (trivial test)
        """

        # Arrange
        dp_list = [dp_image_fully_segmented]
        dataflow_gt = DataFromList(dp_list)
        dataflow_pr = DataFromList(dp_list)

        # Act
        output = F1MetricMicro.get_distance(dataflow_gt, dataflow_pr, dataset_categories)

        # Assert
        assert isinstance(output, list)
        assert len(output) == 4
        assert output[0] == {"key": "table", "val": 1.0, "num_samples": 1}
        assert output[1] == {"key": "cell", "val": 1.0, "num_samples": 5}
        assert output[2] == {"key": "row", "val": 1.0, "num_samples": 2}
        assert output[3] == {"key": "column", "val": 1.0, "num_samples": 2}
