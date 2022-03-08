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

from pytest import raises

from deepdoctection.datapoint import BoundingBox, Image, ImageAnnotation
from deepdoctection.extern.base import DetectionResult, ObjectDetector, PdfMiner
from deepdoctection.pipe.text import TextExtractionService, TextOrderService
from deepdoctection.utils.settings import names


class TestTextExtractionService:
    """
    Test TextExtractionService2. In this setting, extraction will be tested when extract_from_category is None.
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._text_extract_detector = MagicMock(spec=ObjectDetector)
        self.text_extraction_service = TextExtractionService(self._text_extract_detector)

    def test_integration_pipeline_component(
        self, dp_image_fully_segmented_fully_tiled: Image, word_detect_result: List[DetectionResult]
    ) -> None:
        """
        Integration test through calling :meth:`pass_datapoint` of pipeline component
        """

        # Arrange
        self._text_extract_detector.predict = MagicMock(return_value=word_detect_result)

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_fully_segmented_fully_tiled)
        anns = dp.get_annotation(category_names=names.C.WORD)

        # Assert
        first_text_ann = anns[0]

        embedding_bbox = first_text_ann.image.get_embedding(dp.image_id)  # type: ignore
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
        self.text_extraction_service = TextExtractionService(self._text_extract_detector)

    def test_integration_pipeline_component(
        self, dp_image_fully_segmented_fully_tiled: Image, word_detect_result: List[DetectionResult]
    ) -> None:
        """
        Integration test through calling :meth:`pass_datapoint` of pipeline component
        """

        # Arrange
        dp_image_fully_segmented_fully_tiled.pdf_bytes = b"some bytes"
        self._text_extract_detector.predict = MagicMock(return_value=word_detect_result)
        self._text_extract_detector.get_width_height = MagicMock(return_value=(600, 400))

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_fully_segmented_fully_tiled)
        anns = dp.get_annotation(category_names=names.C.WORD)

        # Assert
        first_text_ann = anns[0]

        embedding_bbox = first_text_ann.image.get_embedding(dp.image_id)  # type: ignore
        assert embedding_bbox == first_text_ann.bounding_box


def test_text_extraction_service_raises_error_with_inconsistent_attributes() -> None:
    """
    Testing TextExtractionService does not build when instantiating with a PdfMiner and passing some ROI
    """

    # Arrange
    text_extract_detector = MagicMock(spec=PdfMiner)

    # Act and Assert
    with raises(AssertionError):
        TextExtractionService(text_extract_detector, extract_from_roi=names.C.TAB)


class TestTextExtractionServiceWithSubImage:
    """
    Test TextExtractionService where extract_from_category = "table". As the unit test will not differ from test
    class above only an integration test will be executed.
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._text_extract_detector = MagicMock(spec=ObjectDetector)
        self.text_extraction_service = TextExtractionService(self._text_extract_detector, extract_from_roi=names.C.TAB)

    def test_integration_pipeline_component(
        self,
        dp_image_with_layout_anns: Image,
        double_word_detect_results: List[List[DetectionResult]],
        word_box_global: List[BoundingBox],
    ) -> None:
        """
        integration test through calling meth: serve of pipeline component
        """

        # Arrange
        self._text_extract_detector.predict = MagicMock(side_effect=double_word_detect_results)

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_with_layout_anns)
        word_anns = dp.get_annotation(category_names=names.C.WORD)
        table_anns = dp.get_annotation(category_names=names.C.TAB)

        assert len(word_anns) == 4
        assert len(table_anns) == 2

        assert isinstance(table_anns, list) and isinstance(word_anns, list)
        first_table_ann = table_anns[0]
        second_table_ann = table_anns[1]
        first_word_ann = word_anns[0]
        second_word_ann = word_anns[1]
        third_word_ann = word_anns[2]
        fourth_word_ann = word_anns[3]

        global_box_fta = first_word_ann.image.get_embedding(dp.image_id)  # type: ignore
        assert global_box_fta == word_box_global[0]
        local_box_fta = first_word_ann.image.get_embedding(first_table_ann.annotation_id)  # type: ignore
        assert local_box_fta == first_word_ann.bounding_box
        ft_text_ann = first_table_ann.image.get_annotation(annotation_ids=first_word_ann.annotation_id)[  # type: ignore
            0
        ]

        global_box_sta = second_word_ann.image.get_embedding(dp.image_id)  # type: ignore
        assert global_box_sta == word_box_global[1]
        local_box_sta = second_word_ann.image.get_embedding(first_table_ann.annotation_id)  # type: ignore
        assert local_box_sta == second_word_ann.bounding_box
        ft_text_ann = first_table_ann.image.get_annotation(  # type: ignore
            annotation_ids=second_word_ann.annotation_id
        )[0]
        assert isinstance(ft_text_ann, ImageAnnotation)

        global_box_tta = third_word_ann.image.get_embedding(dp.image_id)  # type: ignore
        assert global_box_tta == word_box_global[2]
        local_box_tta = third_word_ann.image.get_embedding(second_table_ann.annotation_id)  # type: ignore
        assert local_box_tta == third_word_ann.bounding_box
        st_text_ann = second_table_ann.image.get_annotation(  # type: ignore
            annotation_ids=third_word_ann.annotation_id
        )[0]
        assert isinstance(st_text_ann, ImageAnnotation)

        global_box_fta = fourth_word_ann.image.get_embedding(dp.image_id)  # type: ignore
        assert global_box_fta == word_box_global[3]
        local_box_fta = fourth_word_ann.image.get_embedding(second_table_ann.annotation_id)  # type: ignore
        assert local_box_fta == fourth_word_ann.bounding_box
        st_text_ann = second_table_ann.image.get_annotation(  # type: ignore
            annotation_ids=fourth_word_ann.annotation_id
        )[0]
        assert isinstance(st_text_ann, ImageAnnotation)


class TestTextOrderService:  # pylint: disable=R0903
    """
    Test TextOrderService
    """

    @staticmethod
    def test_integration_pipeline_component(dp_image_with_layout_and_word_annotations: Image) -> None:
        """
        test integration_pipeline_component
        """

        # Arrange
        text_order_service = TextOrderService()
        dp_image = dp_image_with_layout_and_word_annotations

        # Act
        text_order_service.pass_datapoint(dp_image)

        # Assert
        layout_anns = dp_image.get_annotation(category_names=[names.C.TITLE, names.C.TEXT])
        word_anns = dp_image.get_annotation(category_names=names.C.WORD)

        # only need to check on layout_anns and word_anns, if sub cats have been added
        # and numbers are correctly assigned

        sub_cat = layout_anns[0].get_sub_category(names.C.RO)
        assert sub_cat.category_id == "1"
        sub_cat = layout_anns[1].get_sub_category(names.C.RO)
        assert sub_cat.category_id == "2"

        sub_cat = word_anns[0].get_sub_category(names.C.RO)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[1].get_sub_category(names.C.RO)
        assert sub_cat.category_id == "2"
        sub_cat = word_anns[2].get_sub_category(names.C.RO)
        assert sub_cat.category_id == "1"
        sub_cat = word_anns[3].get_sub_category(names.C.RO)
        assert sub_cat.category_id == "2"
