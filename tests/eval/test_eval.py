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
from deepdoctection.pipe.layout import ImageLayoutService
from deepdoctection.utils import DatasetType, tensorpack_available

from ..test_utils import set_num_gpu_to_one

if tensorpack_available():
    from deepdoctection.extern.tpdetect import TPFrcnnDetector


class TestEvaluator:
    """
    Test Evaluator processes correctly
    """

    @fixture
    @patch("deepdoctection.extern.tp.tpcompat.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_one))
    def setup_method(
        self,
        path_to_tp_frcnn_yaml: str,
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
        self._dataset.dataset_info.type = DatasetType.object_detection

        self._layout_detector = TPFrcnnDetector(
            path_yaml=path_to_tp_frcnn_yaml,
            path_weights="",
            categories=categories.get_categories(),
        )
        self._pipe_component = ImageLayoutService(self._layout_detector)
        self._pipe_component.predictor.predict = MagicMock(return_value=detection_results)  # type: ignore
        self._metric = CocoMetric

        self.evaluator = Evaluator(self._dataset, self._pipe_component, self._metric, 1)

    @mark.requires_tf
    @mark.full
    def test_evaluator_runs_and_returns_distance(self, setup_method) -> None:  #  type: ignore  # pylint: disable=W0613
        """
        Testing evaluator runs and returns metric distance
        """

        # Act
        out = self.evaluator.run()

        # Assert
        assert len(out) == 12
