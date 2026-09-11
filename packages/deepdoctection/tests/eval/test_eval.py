# -*- coding: utf-8 -*-
# File: test_eval.py

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
Unit tests for evaluation processes using the `Evaluator` class.

This module contains tests to ensure the `Evaluator` processes input images
and returns metrics correctly. It makes use of mock objects to simulate
dependencies and test various evaluation scenarios.

The tests verify proper handling of detection results, category assignments,
and integration of COCO metrics through the evaluation pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

import shared_test_utils as stu
from dd_core.dataflow import DataFromList
from dd_core.datapoint import BoundingBox, Image, ImageAnnotation
from dd_core.utils import DatasetKind, get_type
from dd_core.utils.object_types import LayoutLabel
from dd_datasets.base import DatasetCategories
from dd_datasets.doc_factory import DocumentDatasetFactory
from deepdoctection.eval import CocoMetric, Evaluator
from deepdoctection.eval.record_align import (
    RecordAlignF1Metric,
    RecordAlignF1MetricMicro,
    RecordAlignMetric,
    RecordAlignPrecisionMetric,
    RecordAlignPrecisionMetricMicro,
    RecordAlignRecallMetric,
    RecordAlignRecallMetricMicro,
)
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.hfdetr import HFDetrDerivedDetector
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager
from deepdoctection.pipe.layout import ImageLayoutService


@pytest.mark.skipif(DatasetCategories is None, reason="dd_datasets is not installed; DatasetCategories unavailable")
class TestEvaluator:
    """
    Test Evaluator processes correctly
    """

    @pytest.fixture
    @patch("deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_model", MagicMock(return_value=MagicMock()))
    @patch("deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_pre_processor", MagicMock())
    @patch("deepdoctection.extern.hfdetr.PretrainedConfig.from_pretrained", MagicMock())
    def setup_method(
        self,
        dp_image: Image,
    ) -> None:
        """
        setup the necessary requirements
        """
        detection_results = [
            DetectionResult(box=[15.0, 100.0, 60.0, 150.0], score=0.9, class_id=1, class_name=get_type("row")),
            DetectionResult(box=[15.0, 200.0, 70.0, 240.0], score=0.8, class_id=1, class_name=get_type("row")),
            DetectionResult(box=[10.0, 50.0, 20.0, 250.0], score=0.7, class_id=2, class_name=get_type("column")),
        ]
        categories = DatasetCategories(init_categories=[get_type("row"), get_type("column")])
        detr_categories = {
            1: get_type("table"),
            2: get_type("column"),
            3: get_type("row"),
            4: get_type("column_header"),
            5: get_type("projected_row_header"),
            6: get_type("spanning"),
        }
        row_anns = [
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=15.0, uly=100.0, lrx=60.0, lry=150.0, absolute_coords=True),
                category_name="row",
                category_id=1,
            ),
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=15.0, uly=200.0, lrx=70.0, lry=240.0, absolute_coords=True),
                category_name="row",
                category_id=1,
            ),
        ]

        col_anns = [
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=10.0, uly=50.0, lrx=20.0, lry=250.0, absolute_coords=True),
                category_name="column",
                category_id=2,
            ),
            ImageAnnotation(
                bounding_box=BoundingBox(ulx=40.0, uly=20.0, lrx=50.0, lry=240.0, absolute_coords=True),
                category_name="column",
                category_id=2,
            ),
        ]
        anns = row_anns + col_anns
        for ann in anns:
            dp_image.dump(ann)

        self._dataset = MagicMock()
        self._dataset.dataflow = MagicMock()
        self._dataset.dataset_info = MagicMock()
        self._dataset.dataflow.build = MagicMock(return_value=DataFromList([dp_image]))
        self._dataset.dataflow.categories = categories
        self._dataset.dataset_info.type = DatasetKind.OBJECT_DETECTION
        ModelDownloadManager.maybe_download_weights_and_configs("Aryn/deformable-detr-DocLayNet/model.safetensors")
        path_config = ModelCatalog.get_full_path_configs("Aryn/deformable-detr-DocLayNet/model.safetensors")
        path_weights = ModelCatalog.get_full_path_weights("Aryn/deformable-detr-DocLayNet/model.safetensors")
        preprocessor_config = ModelCatalog.get_full_path_preprocessor_configs(
            "Aryn/deformable-detr-DocLayNet/model.safetensors"
        )
        self._layout_detector = HFDetrDerivedDetector(
            path_config_json=path_config,
            path_weights=path_weights,
            path_feature_extractor_config_json=preprocessor_config,
            categories=detr_categories,
            device="cpu",
        )
        self._pipe_component = ImageLayoutService(self._layout_detector)
        self._pipe_component.predictor.predict = MagicMock(return_value=detection_results)  # type: ignore
        self._metric = CocoMetric

        self.evaluator = Evaluator(self._dataset, self._pipe_component, self._metric, 1)

    def test_evaluator_runs_and_returns_distance(self, setup_method) -> None:  #  type: ignore  # pylint: disable=W0613
        """
        Testing evaluator runs and returns metric distance
        """

        # Act
        out = self.evaluator.run()

        # Assert
        assert len(out) == 12


class TestEvaluatorWithPredictionsDataset:
    """
    Test Evaluator evaluates a ground truth dataset against a precomputed predictions dataset holding
    `structured_output` extractions, without running any pipeline component. Ground truth is the
    `eval_doc` fixture document, predictions is `eval_doc_pred`, a copy with two `structured_output`
    values changed: `accountHolder.company` on page 1 and `balances.closingBalance` on page 2.

    The real `RecordAlignMetric` family is used unmodified, configured via `set_categories` to compare
    the `structured_output` summary sub category by value: `mapper` stays `image_or_docs_to_cat_id`, only
    `_summary_sub_cats` and the new `_id_name_or_value` attribute are set.
    """

    @pytest.fixture
    def setup_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        setup a ground truth dataset (eval_doc) and a predictions dataset (eval_doc_pred)
        """
        monkeypatch.setenv("ENABLE_DYNAMIC_OBJECT_TYPES", "True")

        self._gt_dataset = DocumentDatasetFactory.make(
            name="eval_doc",
            location=str(stu.asset_path("eval_doc").parent),
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=[LayoutLabel.TABLE],
        )
        self._predictions_dataset = DocumentDatasetFactory.make(
            name="eval_doc_pred",
            location=str(stu.asset_path("eval_doc_pred").parent),
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=[LayoutLabel.TABLE],
        )
        self._metric = RecordAlignF1Metric
        self._pipe_component = MagicMock()

    @staticmethod
    def _reset(metric_cls: type[RecordAlignMetric]) -> None:
        metric_cls._cats = None  # pylint: disable=W0212
        metric_cls._sub_cats = None  # pylint: disable=W0212
        metric_cls._summary_sub_cats = None  # pylint: disable=W0212
        metric_cls._id_name_or_value = "id"  # pylint: disable=W0212

    @pytest.mark.parametrize("metric_cls", [RecordAlignPrecisionMetric, RecordAlignRecallMetric, RecordAlignF1Metric])
    def test_evaluator_detects_structured_output_differences(
        self, setup_method: None, metric_cls: type[RecordAlignMetric]  # pylint: disable=W0613
    ) -> None:
        """
        Passing predictions_dataset skips the pipeline entirely; the real RecordAlignMetric family picks
        up the two fields changed in eval_doc_pred and reports every other field as a perfect match
        """

        # Arrange
        metric_cls.set_categories(
            category_names=[], summary_sub_category_names="structured_output", id_name_or_value="value"
        )
        evaluator = Evaluator(self._gt_dataset, metric=metric_cls, predictions_dataset=self._predictions_dataset)

        # Act
        out = evaluator.run(mode="image")

        # Assert
        assert evaluator.pipe_component is None
        assert evaluator.pipe is None
        by_key = {str(row["key"]): row["val"] for row in out}
        changed = {"structured_output.accountHolder.company", "structured_output.balances.closingBalance"}
        assert by_key["structured_output.balances.closingBalance"] == 0.0
        # company differs on page 1 only, matches on page 2: exactly half right
        assert by_key["structured_output.accountHolder.company"] == 0.5
        unaffected = {key: val for key, val in by_key.items() if key not in changed}
        assert unaffected  # sanity: there are other fields besides the two changed ones
        assert all(val == 1.0 for val in unaffected.values())

        # Clean-up
        self._reset(metric_cls)

    @pytest.mark.parametrize(
        "metric_cls", [RecordAlignPrecisionMetricMicro, RecordAlignRecallMetricMicro, RecordAlignF1MetricMicro]
    )
    def test_evaluator_micro_variants_detect_structured_output_differences(
        self, setup_method: None, metric_cls: type[RecordAlignMetric]  # pylint: disable=W0613
    ) -> None:
        """
        The micro variants also honor the structured_output configuration and report a single row below
        1.0, since two out of many field values differ
        """

        # Arrange
        metric_cls.set_categories(
            category_names=[], summary_sub_category_names="structured_output", id_name_or_value="value"
        )
        evaluator = Evaluator(self._gt_dataset, metric=metric_cls, predictions_dataset=self._predictions_dataset)

        # Act
        out = evaluator.run(mode="image")

        # Assert
        assert len(out) == 1
        assert out[0]["key"] == "total"
        assert 0.0 < out[0]["val"] < 1.0

        # Clean-up
        self._reset(metric_cls)

    def test_evaluator_raises_when_both_component_and_predictions_dataset_given(
        self, setup_method: None  # pylint: disable=W0613
    ) -> None:
        """
        component_or_pipeline and predictions_dataset are mutually exclusive
        """

        # Act & Assert
        with pytest.raises(ValueError):
            Evaluator(
                self._gt_dataset,
                self._pipe_component,
                self._metric,
                predictions_dataset=self._predictions_dataset,
            )

    def test_evaluator_raises_when_neither_component_nor_predictions_dataset_given(
        self, setup_method: None  # pylint: disable=W0613
    ) -> None:
        """
        One of component_or_pipeline or predictions_dataset must be given
        """

        # Act & Assert
        with pytest.raises(ValueError):
            Evaluator(self._gt_dataset, metric=self._metric)

    def test_compare_raises_when_predictions_dataset_used(self, setup_method: None) -> None:  # pylint: disable=W0613
        """
        compare() requires a live pipeline and is not supported in predictions_dataset mode
        """

        # Arrange
        evaluator = Evaluator(self._gt_dataset, metric=self._metric, predictions_dataset=self._predictions_dataset)

        # Act & Assert
        with pytest.raises(ValueError):
            next(evaluator.compare())
