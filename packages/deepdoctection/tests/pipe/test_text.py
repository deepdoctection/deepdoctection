# -*- coding: utf-8 -*-
# File: test_text.py

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

"""
Testing module pipe.text
"""
from copy import deepcopy
from typing import List
from unittest.mock import MagicMock

from pytest import mark, raises

from dd_core.datapoint import BoundingBox, Image, ImageAnnotation
from dd_core.utils.object_types import get_type
from deepdoctection.extern.base import DetectionResult, ObjectDetector, PdfMiner
from deepdoctection.pipe.text import TextExtractionService


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
        self._text_extract_detector.model_id = "test_model"
        self.text_extraction_service = TextExtractionService(self._text_extract_detector)

    def test_integration_pipeline_component(self, dp_image_fully_segmented_fully_tiled: Image) -> None:
        """
        Integration test through calling `pass_datapoint` of pipeline component
        """

        word_detect_result = [
            DetectionResult(
                box=[10.0, 10.0, 24.0, 23.0],
                score=0.8,
                text="foo",
                block="1",
                line="2",
                class_id=1,
                class_name=get_type("word"),
            ),
            DetectionResult(
                box=[30.0, 20.0, 38.0, 25.0],
                score=0.2,
                text="bak",
                block="4",
                line="5",
                class_id=1,
                class_name=get_type("word"),
            ),
        ]

        self._text_extract_detector.predict = MagicMock(return_value=word_detect_result)

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image_fully_segmented_fully_tiled)
        anns = dp.get_annotation(category_names=get_type("word"))

        # Assert
        first_text_ann = anns[0]

        embedding_bbox = first_text_ann.get_bounding_box(dp.image_id)
        assert embedding_bbox == BoundingBox(absolute_coords=False, ulx=0.01666667, uly=0.025, lrx=0.04, lry=0.0575)


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
        self._text_extract_detector.model_id = "test_model"
        self.text_extraction_service = TextExtractionService(self._text_extract_detector)

    def test_integration_pipeline_component(self, dp_image_fully_segmented_fully_tiled: Image) -> None:
        """
        Integration test through calling `pass_datapoint` of pipeline component
        """

        word_detect_result = [
            DetectionResult(
                box=[10.0, 10.0, 24.0, 23.0],
                score=0.8,
                text="foo",
                block="1",
                line="2",
                class_id=1,
                class_name=get_type("word"),
            ),
            DetectionResult(
                box=[30.0, 20.0, 38.0, 25.0],
                score=0.2,
                text="bak",
                block="4",
                line="5",
                class_id=1,
                class_name=get_type("word"),
            ),
        ]

        dp_image_fully_segmented_fully_tiled.pdf_bytes = b"some bytes"
        self._text_extract_detector.predict = MagicMock(return_value=word_detect_result)
        self._text_extract_detector.get_width_height = MagicMock(return_value=(600, 400))

        dp = self.text_extraction_service.pass_datapoint(dp_image_fully_segmented_fully_tiled)
        anns = dp.get_annotation(category_names=get_type("word"))

        first_text_ann = anns[0]

        embedding_bbox = first_text_ann.get_bounding_box(dp.image_id)
        assert embedding_bbox == BoundingBox(absolute_coords=False, ulx=0.01666667, uly=0.025, lrx=0.04, lry=0.0575)


def test_text_extraction_service_raises_error_with_inconsistent_attributes() -> None:
    """
    Testing TextExtractionService does not build when instantiating with a PdfMiner and passing some ROI
    """

    # Arrange
    text_extract_detector = MagicMock(spec=PdfMiner)
    text_extract_detector.name = "mock_pdfminer"
    text_extract_detector.model_id = "test_model"

    # Act and Assert
    with raises(TypeError):
        TextExtractionService(text_extract_detector, extract_from_roi=get_type("table"))


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
        self._text_extract_detector.model_id = "test_model"
        self.text_extraction_service = TextExtractionService(
            self._text_extract_detector, extract_from_roi=get_type("table")
        )

    @mark.basic
    def test_integration_pipeline_component(self, dp_image: Image, layout_annotations) -> None:  # type: ignore
        """
        integration test through calling serve of pipeline component
        """
        dp_image = deepcopy(dp_image)
        for img_ann in layout_annotations():
            dp_image.dump(img_ann)
            dp_image.image_ann_to_image(img_ann.annotation_id, True)

        word_detect_result = [
            DetectionResult(
                box=[10.0, 10.0, 24.0, 23.0],
                score=0.8,
                text="foo",
                block="1",
                line="2",
                class_id=1,
                class_name=get_type("word"),
            ),
            DetectionResult(
                box=[30.0, 20.0, 38.0, 25.0],
                score=0.2,
                text="bak",
                block="4",
                line="5",
                class_id=1,
                class_name=get_type("word"),
            ),
        ]

        word_box_global = [
            BoundingBox(absolute_coords=False, ulx=0.1, uly=0.15, lrx=0.12333333, lry=0.1825),
            BoundingBox(absolute_coords=False, ulx=0.13333333, uly=0.175, lrx=0.14666667, lry=0.1875),
            BoundingBox(absolute_coords=False, ulx=0.35, uly=0.15, lrx=0.37333333, lry=0.1825),
            BoundingBox(absolute_coords=False, ulx=0.38333333, uly=0.175, lrx=0.39666667, lry=0.1875),
        ]

        double_word_detect_results = [word_detect_result, word_detect_result]

        # Arrange
        self._text_extract_detector.predict = MagicMock(side_effect=double_word_detect_results)

        # Act
        dp = self.text_extraction_service.pass_datapoint(dp_image)
        word_anns = dp.get_annotation(category_names=get_type("word"))
        table_anns = dp.get_annotation(category_names=get_type("table"))

        assert len(word_anns) == 4
        assert len(table_anns) == 2

        assert isinstance(table_anns, list) and isinstance(word_anns, list)
        first_table_ann = table_anns[0]
        assert first_table_ann.image is not None
        second_table_ann = table_anns[1]
        assert  second_table_ann.image is not None
        first_word_ann = word_anns[0]
        second_word_ann = word_anns[1]
        third_word_ann = word_anns[2]
        fourth_word_ann = word_anns[3]

        global_box_fta = first_word_ann.get_bounding_box(dp.image_id)
        assert global_box_fta == word_box_global[0]
        local_box_fta = first_word_ann.get_bounding_box(first_table_ann.annotation_id)
        assert first_word_ann.bounding_box is not None
        assert local_box_fta == first_word_ann.bounding_box.transform(
            first_table_ann.image.width, first_table_ann.image.height
        )
        ft_text_ann = first_table_ann.image.get_annotation(annotation_ids=first_word_ann.annotation_id)[
            0
        ]

        assert isinstance(ft_text_ann, ImageAnnotation)

        global_box_sta = second_word_ann.get_bounding_box(dp.image_id)
        assert global_box_sta == word_box_global[1]
        local_box_sta = second_word_ann.get_bounding_box(first_table_ann.annotation_id)
        assert second_word_ann.bounding_box is not None
        assert local_box_sta == second_word_ann.bounding_box.transform(
            first_table_ann.image.width, first_table_ann.image.height
        )
        ft_text_ann = first_table_ann.image.get_annotation(
            annotation_ids=second_word_ann.annotation_id
        )[0]
        assert isinstance(ft_text_ann, ImageAnnotation)

        global_box_tta = third_word_ann.get_bounding_box(dp.image_id)
        assert global_box_tta == word_box_global[2]
        local_box_tta = third_word_ann.get_bounding_box(second_table_ann.annotation_id)
        assert third_word_ann.bounding_box is not None
        assert local_box_tta == third_word_ann.bounding_box.transform(
            second_table_ann.image.width, second_table_ann.image.height
        )
        st_text_ann = second_table_ann.image.get_annotation(
            annotation_ids=third_word_ann.annotation_id
        )[0]
        assert isinstance(st_text_ann, ImageAnnotation)

        global_box_fta = fourth_word_ann.get_bounding_box(dp.image_id)
        assert global_box_fta == word_box_global[3]
        local_box_fta = fourth_word_ann.get_bounding_box(second_table_ann.annotation_id)
        assert fourth_word_ann.bounding_box is not None
        assert local_box_fta == fourth_word_ann.bounding_box.transform(
            second_table_ann.image.width, second_table_ann.image.height
        )
        st_text_ann = second_table_ann.image.get_annotation(
            annotation_ids=fourth_word_ann.annotation_id
        )[0]
        assert isinstance(st_text_ann, ImageAnnotation)
