# -*- coding: utf-8 -*-
# File: test_layout.py

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


import pytest
from typing import cast, List
from unittest.mock import create_autospec

from dd_core.datapoint.view import Image
from dd_core.utils.object_types import get_type

from deepdoctection.pipe.layout import ImageLayoutService, skip_if_category_or_service_extracted
from deepdoctection.extern.base import DetectionResult, ObjectDetector




def test_skip_if_category_or_service_extracted_by_category(image: Image):
    result = skip_if_category_or_service_extracted(category_names=get_type("word"))(image)
    assert result is True


def test_skip_if_category_or_service_extracted_by_service_id(image: Image):
    result = skip_if_category_or_service_extracted(service_ids="01a15bff")(image)
    assert result is True


def test_image_layout_service_rebuilds_annotations(image_without_anns: Image, anns: List):
    det_results: List[DetectionResult] = []
    for ann in anns:
        det_results.append(
            DetectionResult(
                box=ann.bounding_box.to_list("xyxy"),
                class_id=ann.category_id,
                score=ann.score if ann.score is not None else 1.0,
                class_name=ann.category_name.value,
                absolute_coords=ann.bounding_box.absolute_coords
            )
        )

    mock_detector = create_autospec(ObjectDetector, instance=True)
    mock_detector.name = "fake_layout"
    mock_detector.model_id = "fake_model_id"
    mock_detector.predict.return_value = det_results
    mock_detector.clone.return_value = mock_detector

    detector = cast(ObjectDetector, mock_detector)
    layout_service = ImageLayoutService(layout_detector=detector)

    result_image = layout_service.pass_datapoint(image_without_anns)

    expected_anns = result_image.get_annotation()
    assert len(expected_anns) == len(anns)

    def sort_key(a):
        return (a.category_id, a.bounding_box.to_list("xyxy"))

    anns_sorted = sorted(anns, key=sort_key)
    expected_sorted = sorted(expected_anns, key=sort_key)

    for orig, recreated in zip(anns_sorted, expected_sorted):
        assert recreated.category_id == orig.category_id
        assert recreated.category_name == orig.category_name
        assert recreated.score == pytest.approx(orig.score if orig.score is not None else 1.0, rel=1e-6)
        assert recreated.bounding_box.to_list("xyxy") == orig.bounding_box.to_list("xyxy")
