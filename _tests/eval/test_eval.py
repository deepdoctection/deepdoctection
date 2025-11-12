# -*- coding: utf-8 -*-
# File: test_eval.py

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
Testing the module eval.eval
"""
from typing import List
from unittest.mock import MagicMock, patch

from pytest import fixture, mark

from deepdoctection.dataflow import DataFromList
from deepdoctection.datapoint import Image
from deepdoctection.datasets import DatasetCategories
from deepdoctection.eval import CocoMetric, Evaluator
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.hfdetr import HFDetrDerivedDetector

from deepdoctection.pipe.layout import ImageLayoutService
from deepdoctection.utils import DatasetType, ObjectTypes


class TestEvaluator:
    """
    Test Evaluator processes correctly
    """

    @fixture
    @patch("deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_model", MagicMock(return_value=MagicMock()))
    @patch("deepdoctection.extern.hfdetr.HFDetrDerivedDetector.get_pre_processor", MagicMock())
    @patch("deepdoctection.extern.hfdetr.PretrainedConfig.from_pretrained", MagicMock())
    def setup_method(
        self,
        detr_categories: dict[int, ObjectTypes],
        image_with_anns: Image,
        categories: DatasetCategories,
        detection_results: List[DetectionResult],
    ) -> None:
        """
        setup the necessary requirements
        """

        self._dataset = MagicMock()
        self._dataset.dataflow = MagicMock()
        self._dataset.dataset_info = MagicMock()
        self._dataset.dataflow.build = MagicMock(return_value=DataFromList([image_with_anns]))
        self._dataset.dataflow.categories = categories
        self._dataset.dataset_info.type = DatasetType.OBJECT_DETECTION

        self._layout_detector = HFDetrDerivedDetector(
            path_config_json="",
            path_weights="",
            path_feature_extractor_config_json="",
            categories=detr_categories,
            device="cpu",
        )
        self._pipe_component = ImageLayoutService(self._layout_detector)
        self._pipe_component.predictor.predict = MagicMock(return_value=detection_results)  # type: ignore
        self._metric = CocoMetric

        self.evaluator = Evaluator(self._dataset, self._pipe_component, self._metric, 1)

    @mark.pt_deps
    def test_evaluator_runs_and_returns_distance(self, setup_method) -> None:  #  type: ignore  # pylint: disable=W0613
        """
        Testing evaluator runs and returns metric distance
        """

        # Act
        out = self.evaluator.run()

        # Assert
        assert len(out) == 12
