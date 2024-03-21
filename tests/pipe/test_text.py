# -*- coding: utf-8 -*-
# File: test_text.py

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
Testing module pipe.text
"""

from typing import List
from unittest.mock import MagicMock

from pytest import mark, raises

from deepdoctection.datapoint import BoundingBox, Image, ImageAnnotation
from deepdoctection.extern.base import DetectionResult, ObjectDetector, PdfMiner
from deepdoctection.pipe.text import TextExtractionService
from deepdoctection.utils.settings import LayoutType


class TestTextExtractionService:
    """
    Test TextExtractionService2. In this setting, extraction will be tested when extract_from_category is None.
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._text_extract_detector = MagicMock(spec=ObjectDetector)
        self._text_extract_detector.name = "mock_text_extractor"
        self.text_extraction_service = TextExtractionService(self._text_extract_detector)

    @mark.basic
    def test_integration_pipeline_component(
        self, dp_image_fully_segmented_fully_tiled: Image, word_detect_result: List[DetectionResult]
    ) -> None:
        """
        Integration test through calling `pass_datapoint` of pipeline component
        """

        # Arrange
        self._text_extract_detector.predict = MagicMock(return_value=word_detect_result)

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_fully_segmented_fully_tiled)
        anns = dp.get_annotation(category_names=LayoutType.word)

        # Assert
        first_text_ann = anns[0]

        embedding_bbox = first_text_ann.get_bounding_box(dp.image_id)
        assert embedding_bbox == first_text_ann.bounding_box


class TestTextExtractionServiceWithPdfPlumberDetector:
    """
    Test TextExtractionServiceWithPdfPlumberDetector with PdfPlumberTextDetector
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._text_extract_detector = MagicMock(spec=PdfMiner)
        self._text_extract_detector.name = "mock_pdfminer"
        self.text_extraction_service = TextExtractionService(self._text_extract_detector)

    @mark.basic
    def test_integration_pipeline_component(
        self, dp_image_fully_segmented_fully_tiled: Image, word_detect_result: List[DetectionResult]
    ) -> None:
        """
        Integration test through calling `pass_datapoint` of pipeline component
        """

        # Arrange
        dp_image_fully_segmented_fully_tiled.pdf_bytes = b"some bytes"
        self._text_extract_detector.predict = MagicMock(return_value=word_detect_result)
        self._text_extract_detector.get_width_height = MagicMock(return_value=(600, 400))

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_fully_segmented_fully_tiled)
        anns = dp.get_annotation(category_names=LayoutType.word)

        # Assert
        first_text_ann = anns[0]

        embedding_bbox = first_text_ann.get_bounding_box(dp.image_id)
        assert embedding_bbox == first_text_ann.bounding_box


@mark.basic
def test_text_extraction_service_raises_error_with_inconsistent_attributes() -> None:
    """
    Testing TextExtractionService does not build when instantiating with a PdfMiner and passing some ROI
    """

    # Arrange
    text_extract_detector = MagicMock(spec=PdfMiner)
    text_extract_detector.name = "mock_pdfminer"

    # Act and Assert
    with raises(TypeError):
        TextExtractionService(text_extract_detector, extract_from_roi=LayoutType.table)


class TestTextExtractionServiceWithSubImage:
    """
    Test TextExtractionService where extract_from_category = "table". As the unit test will not differ from test
    class above only an integration test will be executed.
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._text_extract_detector = MagicMock(spec=ObjectDetector, accepts_batch=False)
        self._text_extract_detector.name = "mock_text_extractor"
        self.text_extraction_service = TextExtractionService(
            self._text_extract_detector, extract_from_roi=LayoutType.table
        )

    @mark.basic
    def test_integration_pipeline_component(
        self,
        dp_image_with_layout_anns: Image,
        double_word_detect_results: List[List[DetectionResult]],
        word_box_global: List[BoundingBox],
    ) -> None:
        """
        integration test through calling  serve of pipeline component
        """

        # Arrange
        self._text_extract_detector.predict = MagicMock(side_effect=double_word_detect_results)

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_with_layout_anns)
        word_anns = dp.get_annotation(category_names=LayoutType.word)
        table_anns = dp.get_annotation(category_names=LayoutType.table)

        assert len(word_anns) == 4
        assert len(table_anns) == 2

        assert isinstance(table_anns, list) and isinstance(word_anns, list)
        first_table_ann = table_anns[0]
        second_table_ann = table_anns[1]
        first_word_ann = word_anns[0]
        second_word_ann = word_anns[1]
        third_word_ann = word_anns[2]
        fourth_word_ann = word_anns[3]

        global_box_fta = first_word_ann.get_bounding_box(dp.image_id)
        assert global_box_fta == word_box_global[0]
        local_box_fta = first_word_ann.get_bounding_box(first_table_ann.annotation_id)
        assert local_box_fta == first_word_ann.bounding_box
        ft_text_ann = first_table_ann.image.get_annotation(annotation_ids=first_word_ann.annotation_id)[  # type: ignore
            0
        ]

        assert isinstance(ft_text_ann, ImageAnnotation)

        global_box_sta = second_word_ann.get_bounding_box(dp.image_id)
        assert global_box_sta == word_box_global[1]
        local_box_sta = second_word_ann.get_bounding_box(first_table_ann.annotation_id)
        assert local_box_sta == second_word_ann.bounding_box
        ft_text_ann = first_table_ann.image.get_annotation(  # type: ignore
            annotation_ids=second_word_ann.annotation_id
        )[0]
        assert isinstance(ft_text_ann, ImageAnnotation)

        global_box_tta = third_word_ann.get_bounding_box(dp.image_id)
        assert global_box_tta == word_box_global[2]
        local_box_tta = third_word_ann.get_bounding_box(second_table_ann.annotation_id)
        assert local_box_tta == third_word_ann.bounding_box
        st_text_ann = second_table_ann.image.get_annotation(  # type: ignore
            annotation_ids=third_word_ann.annotation_id
        )[0]
        assert isinstance(st_text_ann, ImageAnnotation)

        global_box_fta = fourth_word_ann.get_bounding_box(dp.image_id)
        assert global_box_fta == word_box_global[3]
        local_box_fta = fourth_word_ann.get_bounding_box(second_table_ann.annotation_id)
        assert local_box_fta == fourth_word_ann.bounding_box
        st_text_ann = second_table_ann.image.get_annotation(  # type: ignore
            annotation_ids=fourth_word_ann.annotation_id
        )[0]
        assert isinstance(st_text_ann, ImageAnnotation)