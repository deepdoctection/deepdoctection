# -*- coding: utf-8 -*-
# File: fastlang.py

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
Deepdoctection wrappers for fasttext language detection models
"""
from abc import ABC
from copy import copy
from typing import Any, List, Mapping, Tuple, Union

from ..utils.file_utils import Requirement, fasttext_available, get_fasttext_requirement
from ..utils.settings import TypeOrStr, get_type
from .base import DetectionResult, LanguageDetector, PredictorBase

if fasttext_available():
    from fasttext import load_model  # type: ignore


class FasttextLangDetectorMixin(LanguageDetector, ABC):
    """
    Base class for Fasttext language detection implementation. This class only implements the basic wrapper functions.
    """

    def __init__(self, categories: Mapping[str, TypeOrStr]) -> None:
        """
        :param categories: A dict with the model output label and value. We use as convention the ISO 639-2 language
        """
        self.categories = copy({idx: get_type(cat) for idx, cat in categories.items()})

    def output_to_detection_result(self, output: Union[Tuple[Any, Any]]) -> DetectionResult:
        """
        Generating `DetectionResult` from model output
        :param output: FastText model output
        :return: `DetectionResult` filled with `text` and `score`
        """
        return DetectionResult(text=self.categories[output[0][0]], score=output[1][0])


class FasttextLangDetector(FasttextLangDetectorMixin):
    """
    Fasttext language detector wrapper. Two models provided in the fasttext library can be used to identify languages.
    The background to the models can be found in the works:

    [1] Joulin A, Grave E, Bojanowski P, Mikolov T, Bag of Tricks for Efficient Text Classification

    [2] Joulin A, Grave E, Bojanowski P, Douze M, Jégou H, Mikolov T, FastText.zip: Compressing text classification
        models

    The models are distributed under the Creative Commons Attribution-Share-Alike License 3.0.
    (<https://creativecommons.org/licenses/by-sa/3.0/>)

    When loading the models via the ModelCatalog, the original and unmodified models are used.

        path_weights = ModelCatalog.get_full_path_weights("fasttext/lid.176.bin")
        profile = ModelCatalog.get_profile("fasttext/lid.176.bin")
        lang_detector = FasttextLangDetector(path_weights,profile.categories)
        detection_result = lang_detector.predict("some text in some language")

    """

    def __init__(self, path_weights: str, categories: Mapping[str, TypeOrStr]):
        """
        :param path_weights: path to model weights
        :param categories: A dict with the model output label and value. We use as convention the ISO 639-2 language
                           code.
        """
        super().__init__(categories)
        self.name = "fasttest_lang_detector"
        self.path_weights = path_weights
        self.model = self.get_wrapped_model(self.path_weights)

    def predict(self, text_string: str) -> DetectionResult:
        output = self.model.predict(text_string)
        return self.output_to_detection_result(output)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_fasttext_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_weights, self.categories)

    @staticmethod
    def get_wrapped_model(path_weights: str) -> Any:
        """
        Get the wrapped model
        :param path_weights: path to model weights
        """
        return load_model(path_weights)
