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
from typing import Optional, Sequence

from ..datapoint.image import Image
from ..datapoint.view import Page
from ..extern.base import LanguageDetector, ObjectDetector
from ..utils.error import ImageError
from ..utils.settings import PageType, TypeOrStr, get_type
from .base import MetaAnnotation, PipelineComponent
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

          lang_detector = FasttextLangDetector(path_weights,profile.categories)
          component = LanguageDetectionService(lang_detector, text_container="word",
                                               text_block_names=["text","title","table"])

    """

    def __init__(
        self,
        language_detector: LanguageDetector,
        text_container: Optional[TypeOrStr] = None,
        text_detector: Optional[ObjectDetector] = None,
        floating_text_block_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        :param language_detector: Detector to determine text
        :param text_container: text container, needed to generate the reading order. Not necessary when passing a
                               text detector.
        :param text_detector: Object detector to extract text. You cannot use a Pdfminer here.

        :param floating_text_block_categories: text blocks, needed for generating the reading order. Not necessary
                                 when passing a text detector.
        """

        self.predictor = language_detector
        self.text_detector = text_detector
        self.text_container = get_type(text_container) if text_container is not None else text_container
        self.floating_text_block_categories = (
            tuple(get_type(text_block) for text_block in floating_text_block_categories)
            if (floating_text_block_categories is not None)
            else ()
        )

        super().__init__(self._get_name(self.predictor.name))

    def serve(self, dp: Image) -> None:
        if self.text_detector is None:
            page = Page.from_image(dp, self.text_container, self.floating_text_block_categories)
            text = page.text_no_line_break
        else:
            if dp.image is None:
                raise ImageError("image cannot be None")
            detect_result_list = self.text_detector.predict(dp.image)
            # this is a concatenation of all detection result. No reading order
            text = " ".join((result.text for result in detect_result_list if result.text is not None))
        predict_result = self.predictor.predict(text)
        self.dp_manager.set_summary_annotation(
            PageType.LANGUAGE, PageType.LANGUAGE, 1, predict_result.text, predict_result.score
        )

    def clone(self) -> PipelineComponent:
        predictor = self.predictor.clone()
        if not isinstance(predictor, LanguageDetector):
            raise TypeError(f"Predictor must be of type LanguageDetector, but is of type {type(predictor)}")
        return self.__class__(
            language_detector=predictor,
            text_container=self.text_container,
            text_detector=self.text_detector.clone() if self.text_detector is not None else None,
            floating_text_block_categories=self.floating_text_block_categories,
        )

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(image_annotations=(), sub_categories={}, relationships={}, summaries=(PageType.LANGUAGE,))

    @staticmethod
    def _get_name(predictor_name: str) -> str:
        return f"language_detection_{predictor_name}"

    def clear_predictor(self) -> None:
        self.predictor.clear_model()
