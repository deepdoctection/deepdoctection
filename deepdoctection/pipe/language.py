# -*- coding: utf-8 -*-
# File: language.py

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
Module for language detection pipeline component
"""
from typing import List, Optional

from ..datapoint.image import Image
from ..datapoint.page import Page
from ..extern.base import LanguageDetector, ObjectDetector
from ..utils.detection_types import JsonDict
from ..utils.logger import logger
from ..utils.settings import LayoutType, PageType
from .base import PipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("LanguageDetectionService")
class LanguageDetectionService(PipelineComponent):
    """
    Pipeline Component for identifying the language in an image.

    There are two ways to use this component:

    - By analyzing the already extracted and ordered text. For this purpose, a page object is parsed internally and
      the full text is passed to the language_detector. This approach provides the greatest precision.

    - By previous text extraction with an object detector and subsequent transfer of concatenated word elements to the
      language_detector. Only one OCR detector can be used here. This method can be used, for example, to select an OCR
      detector that specializes in a language using. Although the word recognition is less accurate
      when choosing any detector, the results are confident enough to rely on the results, especially when extracting
      longer text passages. So, a TextExtractionService, for example, can be selected as the subsequent pipeline
      component. The words determined by the OCR detector are not transferred to the image object.

      .. code-block:: python

          lang_detector = FasttextLangDetector(path_weights,profile.categories)
          component = LanguageDetectionService(lang_detector, text_container="WORD",
                                               floating_text_block_names=["TEXT","TITLE"],
                                               text_block_names=["TEXT","TITLE",TABLE"])

    """

    def __init__(
        self,
        language_detector: LanguageDetector,
        text_container: str,
        text_detector: Optional[ObjectDetector] = None,
        floating_text_block_names: Optional[List[str]] = None,
        text_block_names: Optional[List[str]] = None,
    ):
        """
        :param language_detector: Detector to determine text
        :param text_container: text container, needed for generating the reading order. Not necessary when passing a
                               text detector.
        :param text_detector: Object detector to extract text. You cannot use a Pdfminer here.

        :param floating_text_block_names: floating text blocks, needed for generating the reading order. Not necessary
                                          when passing a text detector.
        :param text_block_names: text blocks, needed for generating the reading order. Not necessary
                                 when passing a text detector.
        """

        self.predictor = language_detector
        self.text_detector = text_detector
        self._text_container = text_container
        self._floating_text_block_names = floating_text_block_names
        self._text_block_names = text_block_names
        self._init_sanity_checks()
        super().__init__(self._get_name(self.predictor.name))  # cannot use PredictorPipelineComponent class because of return type of predict meth

    def serve(self, dp: Image) -> None:
        if self.text_detector is None:
            page = Page.from_image(dp, self._text_container, self._floating_text_block_names, self._text_block_names)
            text = page.get_text()
        else:
            if dp.image is None:
                raise ValueError("dp.image cannot be None")
            detect_result_list = self.text_detector.predict(dp.image)
            # this is a concatenation of all detection result. No reading order
            text = " ".join([result.text for result in detect_result_list if result.text is not None])
        predict_result = self.predictor.predict(text)
        self.dp_manager.set_summary_annotation(
            PageType.language, PageType.language, 1, predict_result.text, predict_result.score
        )

    def _init_sanity_checks(self) -> None:
        assert (
            self.text_detector or self._text_container
        ), "if no text_detector is provided a text container must be specified"
        if not self.text_detector:
            assert self._text_container in [LayoutType.word, LayoutType.line], (
                f"text_container must be either {LayoutType.word} or " f"{LayoutType.line}"
            )
            assert set(self._floating_text_block_names) <= set(  # type: ignore
                self._text_block_names  # type: ignore
            ), "floating_text_block_names must be a subset of text_block_names"
            if not self._floating_text_block_names and not self._text_block_names:
                logger.info(
                    "floating_text_block_names and text_block_names are set to None and. "
                    "This setting will return no reading order!"
                )

    def clone(self) -> PipelineComponent:
        return self.__class__(
            self.predictor,
            self._text_container,
            self.text_detector,
            self._floating_text_block_names,
            self._text_block_names,
        )

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", [PageType.language]),
            ]
        )

    @staticmethod
    def _get_name(predictor_name: str) -> str:
        return f"language_detection_{predictor_name}"
