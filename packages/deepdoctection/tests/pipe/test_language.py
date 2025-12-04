# -*- coding: utf-8 -*-
# File: test_language.py

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


import numpy as np
from typing import cast
from unittest.mock import create_autospec, Mock

import shared_test_utils as stu
import pytest

from dd_core.datapoint.view import Image
from dd_core.utils.object_types import PageType, get_type

from deepdoctection.extern.hflm import LanguageDetector
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.language import LanguageDetectionService



def test_language_detection_service_sets_summary(image: Image):
    score = np.array([0.95])
    detection = DetectionResult(class_name=get_type("deu"), score=float(score.item()))

    mock_ld = create_autospec(LanguageDetector, instance=True)
    mock_ld.name = "fake_lang"
    mock_ld.predict.return_value = detection
    mock_ld.clone.return_value = mock_ld

    language_detector = cast(LanguageDetector, mock_ld)

    service = LanguageDetectionService(language_detector=language_detector)

    service.dp_manager.set_summary_annotation = Mock()

    service.pass_datapoint(image)

    service.dp_manager.set_summary_annotation.assert_called_once()
    called_args = service.dp_manager.set_summary_annotation.call_args[0]

    assert called_args[0] == PageType.LANGUAGE
    assert called_args[1] == PageType.LANGUAGE
    assert called_args[2] == 1
    assert called_args[3] == "deu"
    assert pytest.approx(called_args[4], rel=1e-6) == float(score.item())
