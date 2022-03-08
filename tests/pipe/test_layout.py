# -*- coding: utf-8 -*-
# File: test_layout.py

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
Testing module pipe.layout
"""

from unittest.mock import MagicMock

from deepdoctection.datapoint import Image, ImageAnnotation
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.layout import ImageLayoutService


class TestImageLayoutService:
    """
    Test ImageLayoutService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._layout_detector = MagicMock()
        self.image_layout_service = ImageLayoutService(self._layout_detector, to_image=True)

    def test_pass_datapoint(
        self, dp_image: Image, layout_detect_results: DetectionResult, layout_annotations: ImageAnnotation
    ) -> None:
        """
        test pass_datapoint
        """

        # Arrange
        self._layout_detector.predict = MagicMock(return_value=layout_detect_results)

        # Act
        dp = self.image_layout_service.pass_datapoint(dp_image)
        anns = dp.get_annotation()

        # Assert
        assert len(anns) == 4
        for ann in anns:
            assert isinstance(ann.image, Image)
            ann.image = None
        assert anns == layout_annotations
