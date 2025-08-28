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
Wrappers for fasttext language detection models
"""

from __future__ import annotations

import os
from abc import ABC
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Union

from lazy_imports import try_import

from ..utils.develop import deprecated
from ..utils.file_utils import Requirement, get_fasttext_requirement, get_numpy_v1_requirement
from ..utils.settings import TypeOrStr, get_type
from ..utils.types import PathLikeOrStr
from .base import DetectionResult, LanguageDetector, ModelCategories

with try_import() as import_guard:
    from fasttext import load_model  # type: ignore # pylint: disable=E0401


class FasttextLangDetectorMixin(LanguageDetector, ABC):
    """
    Base class for `Fasttext` language detection implementation. This class only implements the basic wrapper functions.
    """

    def __init__(self, categories: Mapping[int, TypeOrStr], categories_orig: Mapping[str, TypeOrStr]) -> None:
        """
        Args:
            categories: A `dict` with the model output label and value. We use as convention the `ISO 639-2` language
        """
        self.categories = ModelCategories(init_categories=categories)
        self.categories_orig = MappingProxyType({cat_orig: get_type(cat) for cat_orig, cat in categories_orig.items()})

    def output_to_detection_result(self, output: Union[tuple[Any, Any]]) -> DetectionResult:
        """
        Generating `DetectionResult` from model output

        Args:
            output: `FastText` model output

        Returns:
            `DetectionResult` filled with `text` and `score`
        """
        return DetectionResult(class_name=self.categories_orig[output[0][0]], score=output[1][0])

    @staticmethod
    def get_name(path_weights: PathLikeOrStr) -> str:
        """Returns the name of the model"""
        return "fasttext_" + "_".join(Path(path_weights).parts[-2:])


@deprecated("As FastText archived, it will be deprecated in the near future.", "2025-08-17")
class FasttextLangDetector(FasttextLangDetectorMixin):
    """
    Fasttext language detector wrapper. Two models provided in the fasttext library can be used to identify languages.
    The background to the models can be found in the works:

    Info:
        [1] Joulin A, Grave E, Bojanowski P, Mikolov T, Bag of Tricks for Efficient Text Classification
        [2] Joulin A, Grave E, Bojanowski P, Douze M, Jégou H, Mikolov T, FastText.zip: Compressing text classification
        models

    When loading the models via the `ModelCatalog`, the original and unmodified models are used.

    Example:
        ```python
        path_weights = ModelCatalog.get_full_path_weights("fasttext/lid.176.bin")
        profile = ModelCatalog.get_profile("fasttext/lid.176.bin")
        lang_detector = FasttextLangDetector(path_weights,profile.categories)
        detection_result = lang_detector.predict("some text in some language")
        ```
    """

    def __init__(
        self, path_weights: PathLikeOrStr, categories: Mapping[int, TypeOrStr], categories_orig: Mapping[str, TypeOrStr]
    ):
        """
        Args:
            path_weights: path to model weights
            categories: A dict with the model output label and value. We use as convention the ISO 639-2 language
                        code.
        """
        super().__init__(categories, categories_orig)

        self.path_weights = Path(path_weights)

        self.name = self.get_name(self.path_weights)
        self.model_id = self.get_model_id()

        self.model = self.get_wrapped_model(self.path_weights)

    def predict(self, text_string: str) -> DetectionResult:
        output = self.model.predict(text_string)
        return self.output_to_detection_result(output)

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_numpy_v1_requirement(), get_fasttext_requirement()]

    def clone(self) -> FasttextLangDetector:
        return self.__class__(self.path_weights, self.categories.get_categories(), self.categories_orig)

    @staticmethod
    def get_wrapped_model(path_weights: PathLikeOrStr) -> Any:
        """
        Get the wrapped model

        Args:
            path_weights: path to model weights
        """
        return load_model(os.fspath(path_weights))
