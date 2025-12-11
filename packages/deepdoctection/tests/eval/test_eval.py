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


from typing import List
from unittest.mock import MagicMock, patch

import pytest

from dd_core.dataflow import DataFromList
from dd_core.datapoint import BoundingBox, Image, ImageAnnotation
from dd_core.utils import DatasetType, get_type
from deepdoctection.eval import CocoMetric, Evaluator
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.hfdetr import HFDetrDerivedDetector
from deepdoctection.pipe.layout import ImageLayoutService

try:
    from dd_datasets.base import DatasetCategories
except ImportError:
    DatasetCategories = None


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

    def test_evaluator_runs_and_returns_distance(self, setup_method) -> None:  #  type: ignore  # pylint: disable=W0613
        """
        Testing evaluator runs and returns metric distance
        """

        # Act
        out = self.evaluator.run()

        # Assert
        assert len(out) == 12
