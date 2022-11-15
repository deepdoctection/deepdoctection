# -*- coding: utf-8 -*-
# File: test_language.py

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
Testing module pipe.language
"""

from unittest.mock import MagicMock

from pytest import mark

from deepdoctection.datapoint import ContainerAnnotation, Image
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.language import LanguageDetectionService
from deepdoctection.pipe.text import TextOrderService
from deepdoctection.utils import CellType, LayoutType, PageType


class TestLanguageDetectionService:
    """
    Test LanguageDetectionService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._language_detector = MagicMock()
        self._text_order_service = TextOrderService(
            text_container=LayoutType.word,
            floating_text_block_names=[LayoutType.title, LayoutType.text, LayoutType.list],
            text_block_names=[
                LayoutType.title,
                LayoutType.text,
                LayoutType.list,
                LayoutType.cell,
                CellType.header,
                CellType.body,
            ],
        )
        self.language_detection_service = LanguageDetectionService(
            self._language_detector,
            text_container=LayoutType.word,
            text_block_names=[
                LayoutType.title,
                LayoutType.text,
                LayoutType.list,
                LayoutType.cell,
                CellType.header,
                CellType.body,
            ],
        )

    @mark.basic
    def test_pass_datapoint(
        self, dp_image_with_layout_and_word_annotations: Image, language_detect_result: DetectionResult
    ) -> None:
        """
        test pass datapoint
        """
        # Arrange
        dp_image = dp_image_with_layout_and_word_annotations
        self._language_detector.predict = MagicMock(return_value=language_detect_result)
        dp_with_text_ordered = self._text_order_service.pass_datapoint(dp_image)  # need to setup a reading order

        # Act
        dp = self.language_detection_service.pass_datapoint(dp_with_text_ordered)

        # Assert
        assert dp.summary is not None
        assert dp.summary.get_sub_category(PageType.language).category_name == "language"
        container_ann = dp.summary.get_sub_category(PageType.language)
        assert isinstance(container_ann, ContainerAnnotation)
        assert container_ann.value == "eng"
