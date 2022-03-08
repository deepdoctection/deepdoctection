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
from unittest.mock import MagicMock

from pytest import fixture

from deepdoctection.dataflow import DataFromList  # type: ignore
from deepdoctection.datapoint import Image
from deepdoctection.datasets import DatasetCategories
from deepdoctection.eval import CocoMetric, Evaluator
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.tpdetect import TPFrcnnDetector
from deepdoctection.pipe.layout import ImageLayoutService


class TestEvaluator:
    """
    Test Evaluator processes correctly
    """

    @fixture
    def setup_method(
        self, image_with_anns: Image, categories: DatasetCategories, detection_results: List[DetectionResult]
    ) -> None:
        """
        setup the necessary requirements
        """

        self._dataset = MagicMock()
        self._dataset.dataflow = MagicMock()
        self._dataset.dataflow.build = MagicMock(return_value=DataFromList([image_with_anns]))
        self._dataset.dataflow.categories = categories

        self._layout_detector = MagicMock(spec=TPFrcnnDetector)
        self._layout_detector.clone = MagicMock(return_value=MagicMock(spec=TPFrcnnDetector))
        self._layout_detector.tp_predictor = MagicMock()
        self._pipe_component = ImageLayoutService(self._layout_detector)
        self._layout_detector.predict = MagicMock(return_value=detection_results)
        self._metric = CocoMetric

        self.evaluator = Evaluator(self._dataset, self._pipe_component, self._metric)

    def test_evaluator_runs_and_returns_distance(self, setup_method) -> None:  #  type: ignore  # pylint: disable=W0613
        """
        Testing evaluator runs and returns metric distance
        """

        # Act
        cat_list = self._dataset.dataflow.categories.get_categories(as_dict=False, name_as_key=True)
        out = self.evaluator.run(category_names=cat_list)

        # Assert
        assert len(out) == 12
