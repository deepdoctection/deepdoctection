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

from typing import List
from ..utils.file_utils import fasttext_available, get_fasttext_requirement, Requirement
from .base import LanguageDetector, DetectionResult, PredictorBase

if fasttext_available():
    from fasttext import load_model


class FasttextLangDetector(LanguageDetector):
    """
    Fasttext language detector
    """

    def __init__(self, path_weights: str):
        super().__init__()
        self.path_weights = path_weights
        self.model = load_model(self.path_weights)

    def predict(self, text_string: str) -> DetectionResult:
        output = self.model.predict(text_string)
        return DetectionResult(text=output[0][0].split("_")[-1],
                               score=output[1][0])

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_fasttext_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_weights)




