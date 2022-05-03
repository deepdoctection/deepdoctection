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
from typing import Dict, Optional, Union, List

from ..datapoint.image import Image
from ..extern.base import LanguageDetector, ObjectDetector
from ..utils.settings import names
from ..utils.logger import logger
from ..mapper.pagestruct import to_page
from .base import PredictorPipelineComponent


class LanguageDetectionService(PredictorPipelineComponent):
    """
    Pipeline component for determining the language given by the text.
    """

    def __init__(self, language_detector: LanguageDetector,
                 text_detector: Optional[ObjectDetector]= None,
                 text_container: Optional[str]= None,
                 floating_text_block_names: Optional[Union[str, List[str]]] = None,
                 text_block_names: Optional[Union[str, List[str]]] = None,
    ):

        super().__init__(language_detector,None)
        self.text_detector = text_detector
        self._text_container = text_container
        self._floating_text_block_names = floating_text_block_names
        self._text_block_names = text_block_names
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        if not self.text_detector:
            page = to_page(dp,self._text_container,self._floating_text_block_names,self._text_block_names)
            text = page.get_text()
        else:
            detect_result_list = self.text_detector.predict(dp.image)
            # this is a concatenation of all detection result. No reading order
            text = " ".join([result.text for result in detect_result_list])
        predict_result = self.predictor.predict(text)
        self.dp_manager.set_summary_annotation(names.NLP.LANG, 1, predict_result.text, predict_result.score)

    def _init_sanity_checks(self) -> None:
        assert self.text_detector or self._text_container, "if no text_detector is provided a text container must be " \
                                                          "specified"
        if not self.text_detector:
            assert self._text_container in [names.C.WORD, names.C.LINE], (
                f"text_container must be either {names.C.WORD} or " f"{names.C.LINE}"
            )
            assert set(self._floating_text_block_names) <= set(
                self._text_block_names
            ), "floating_text_block_names must be a subset of text_block_names"
            if (
                    not self._floating_text_block_names
                    and not self._text_block_names
            ):
                logger.info(
                    "floating_text_block_names and text_block_names are set to None and. "
                    "This setting will return no reading order!"
                )




