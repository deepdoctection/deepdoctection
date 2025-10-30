# -*- coding: utf-8 -*-
# File: xxx.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Testing module pipe.transform
"""

from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_array_equal
from pytest import mark

from deepdoctection.datapoint.image import Image
from deepdoctection.extern.base import DetectionResult, ImageTransformer
from deepdoctection.pipe.transform import SimpleTransformService
from deepdoctection.utils.identifier import get_uuid_from_str
from deepdoctection.utils.object_types import PageType


class TestSimpleTransformService:
    """
    Test SimpleTransformService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._transform_predictor = MagicMock(spec=ImageTransformer)
        self._transform_predictor.get_category_names = MagicMock(return_value=(PageType.ANGLE,))
        self._transform_predictor.name = "mock_transform"
        self._transform_predictor.model_id = get_uuid_from_str(self._transform_predictor.name)[:8]
        detect_result = DetectionResult()
        detect_result.new_h = 794  # type: ignore
        detect_result.new_w = 596  # type: ignore
        self._transform_predictor.predict = MagicMock(return_value=detect_result)
        self.simple_transform = SimpleTransformService(self._transform_predictor)

    @mark.basic
    def test_pass_datapoint(self, dp_image: Image) -> None:
        """
        test pass_datapoint
        """

        # Arrange
        np_output_img = np.ones((794, 596, 3), dtype=np.uint8) * 255
        self._transform_predictor.transform_image = MagicMock(return_value=np_output_img)

        # Act
        dp = self.simple_transform.pass_datapoint(dp_image)

        # Assert
        assert_array_equal(dp.image, np_output_img)  # type: ignore
        assert dp.width == 596
        assert dp.height == 794
