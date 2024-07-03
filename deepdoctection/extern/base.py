# -*- coding: utf-8 -*-
# File: base.py

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
Abstract classes for unifying external base- and Doctection predictors
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union, TYPE_CHECKING
from lazy_imports import try_import

from ..utils.types import JsonDict, PixelValues, Requirement
from ..utils.identifier import get_uuid_from_str
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type

if TYPE_CHECKING:
    with try_import() as import_guard:
        import torch


class PredictorBase(ABC):
    """
    Abstract base class for all types of predictors (e.g. object detectors language models, ...)
    """

    name: str
    model_id: str

    def __new__(cls, *args, **kwargs):  # type: ignore # pylint: disable=W0613
        requirements = cls.get_requirements()
        name = cls.__name__ if hasattr(cls, "__name__") else cls.__class__.__name__
        if not all(requirement[1] for requirement in requirements):
            raise ImportError(
                "\n".join(
                    [f"{name} has the following dependencies:"]
                    + [requirement[2] for requirement in requirements if not requirement[1]]
                )
            )
        return super().__new__(cls)

    @classmethod
    @abstractmethod
    def get_requirements(cls) -> list[Requirement]:
        """
        Get a list of requirements for running the detector
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> PredictorBase:
        """
        Clone an instance
        """
        raise NotImplementedError()

    def get_model_id(self) -> str:
        """
        Get the generating model
        """
        if self.name is not None:
            return get_uuid_from_str(self.name)[:8]
        raise ValueError("name must be set before calling get_model_id")


@dataclass
class DetectionResult:
    """
    Simple mutable storage for detection results.

    `box`: [ulx,uly,lrx,lry]

    `class_id`: category id

    `score`: prediction score

    `mask`: binary mask

    `absolute_coords` : absolute coordinates

    `class_name`: category name

    `text`: text string. Used for OCR predictors

    `block`: block number. For reading order from some ocr predictors

    `line`: line number. For reading order from some ocr predictors

    `uuid`: uuid. For assigning detection result (e.g. text to image annotations)


    """

    box: Optional[list[float]] = None
    class_id: Optional[int] = None
    score: Optional[float] = None
    mask: Optional[list[float]] = None
    absolute_coords: bool = True
    class_name: ObjectTypes = DefaultType.default_type
    text: Optional[Union[str, ObjectTypes]] = None
    block: Optional[str] = None
    line: Optional[str] = None
    uuid: Optional[str] = None
    relationships: Optional[dict[str, Any]] = None
    angle: Optional[float] = None


class ObjectDetector(PredictorBase):
    """
    Abstract base class for object detection. This can be anything ranging from layout detection to OCR.
    Use this to connect external detectors with deepdoctection predictors on images.

    **Example:**

            MyFancyTensorpackPredictor(TensorpackPredictor,ObjectDetector)

    and implement the `predict`.
    """

    _categories: Mapping[str, ObjectTypes]

    @property
    def categories(self) -> Mapping[str, ObjectTypes]:
        """categories"""
        return self._categories

    @categories.setter
    def categories(self, categories: Mapping[str, TypeOrStr]) -> None:
        """categories setter"""
        self._categories = {key: get_type(value) for key, value in categories.items()}

    @abstractmethod
    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in `predict`
        """
        return False

    def get_category_names(self) -> list[ObjectTypes]:
        """
        Returns a list with the full range of detectable categories
        """
        return list(self.categories.values())


class PdfMiner(PredictorBase):
    """
    Abstract base class for mining information from PDF documents. Reads in a bytes stream from a PDF document page.
    Use this to connect external pdf miners and wrap them into Deep-Doctection predictors.
    """

    _categories: Mapping[str, ObjectTypes]
    _pdf_bytes: Optional[bytes] = None

    @property
    def categories(self) -> Mapping[str, ObjectTypes]:
        """categories"""
        return self._categories

    @categories.setter
    def categories(self, categories: Mapping[str, TypeOrStr]) -> None:
        self._categories = {key: get_type(value) for key, value in categories.items()}

    @abstractmethod
    def predict(self, pdf_bytes: bytes) -> list[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @abstractmethod
    def get_width_height(self, pdf_bytes: bytes) -> tuple[float, float]:
        """
        Abstract method get_width_height
        """
        raise NotImplementedError()

    def clone(self) -> PredictorBase:
        return self.__class__()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in `predict`
        """
        return False

    def get_category_names(self) -> list[ObjectTypes]:
        """
        Returns a list of possible detectable categories
        """
        return list(self.categories.values())


class TextRecognizer(PredictorBase):
    """
    Abstract base class for text recognition. In contrast to ObjectDetector one assumes that `predict` accepts
    batches of numpy arrays. More precisely, when using `predict` pass a list of tuples with uuids (e.g. image_id,
    or annotation_id) or numpy arrays.
    """

    @abstractmethod
    def predict(self, images: list[tuple[str, PixelValues]]) -> list[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in `predict`
        """
        return True


@dataclass
class TokenClassResult:
    """
    Simple mutable storage for token classification results

     `id`: uuid of token (not unique)

     `token_id`: token id

     `token`: token

     `class_id`: category id

     `class_name`: category name

     `semantic_name`: semantic name

     `bio_tag`: bio tag

     `score`: prediction score
    """

    uuid: str
    token: str
    class_id: int
    class_name: ObjectTypes = DefaultType.default_type
    semantic_name: ObjectTypes = DefaultType.default_type
    bio_tag: ObjectTypes = DefaultType.default_type
    score: Optional[float] = None
    token_id: Optional[int] = None


@dataclass
class SequenceClassResult:
    """
    Storage for sequence classification results

    `class_id`: category id
    `class_name`: category name
    `score`: prediction score
    `class_name_orig`: original class name
    """

    class_id: int
    class_name: ObjectTypes = DefaultType.default_type
    score: Optional[float] = None
    class_name_orig: Optional[str] = None


class LMTokenClassifier(PredictorBase):
    """
    Abstract base class for token classifiers. If you want to connect external token classifiers with Deepdoctection
    predictors wrap them into a class derived from this class. Note, that this class is still DL library agnostic.
    """

    _categories: Mapping[str, ObjectTypes]

    @property
    def categories(self) -> Mapping[str, ObjectTypes]:
        """categories"""
        return self._categories

    @categories.setter
    def categories(self, categories: Mapping[str, TypeOrStr]) -> None:
        """categories setter"""
        self._categories = {key: get_type(value) for key, value in categories.items()}

    @abstractmethod
    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    def get_category_names(self) -> list[ObjectTypes]:
        """
        Returns a list of possible detectable tokens
        """
        return list(self.categories.values())

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Some models require that their inputs must be pre-processed in a specific way. Responsible for converting
        an `Image` datapoint into the input format in inference mode is a mapper function which is called
        in a pipeline component. The mapper function's name, which has to be used must be specified in the returned
        value of `image_to_features_mapping`.
        This mapper function is often implemented for various models and can therefore have various parameters.
        Some parameters can be inferred from the config file of the model parametrization. Some other might not be
        in the parametrization and therefore have to be specified here.

        This method therefore returns a dictionary that contains as keys some arguments of the function
        `image_to_features_mapping` and as values the values necessary for providing the model with the required input.
        """
        return {}

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_raw_features_mapping` is used for training and transforms an image into raw features that can be
        further processed through augmentation or batching. It should not be used when running inference, i.e. when
        running the model in a pipeline component.
        """
        return ""

    @staticmethod
    def image_to_features_mapping() -> str:
        """Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_features_mapping` is a mapping function that converts a single image into ready features that can
        be directly fed into the model. We use this function to determine the input format of the model in a pipeline
        component. Note that this function will also require specific parameters, which can be specified in
        `default_kwargs_for_image_to_features_mapping`.

        """
        return ""


class LMSequenceClassifier(PredictorBase):
    """
    Abstract base class for sequence classification. If you want to connect external sequence classifiers with
    deepdoctection predictors, wrap them into a class derived from this class.
    """

    _categories: Mapping[str, ObjectTypes]

    @property
    def categories(self) -> Mapping[str, ObjectTypes]:
        """categories"""
        return self._categories

    @categories.setter
    def categories(self, categories: Mapping[str, TypeOrStr]) -> None:
        """categories setter"""
        self._categories = {key: get_type(value) for key, value in categories.items()}

    @abstractmethod
    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    def get_category_names(self) -> list[ObjectTypes]:
        """
        Returns a list of possible detectable categories for a sequence
        """
        return list(self.categories.values())

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Some models require that their inputs must be pre-processed in a specific way. Responsible for converting
        an `Image` datapoint into the input format in inference mode is a mapper function which is called
        in a pipeline component. The mapper function's name, which has to be used must be specified in the returned
        value of `image_to_features_mapping`.
        This mapper function is often implemented for various models and can therefore have various parameters.
        Some parameters can be inferred from the config file of the model parametrization. Some other might not be
        in the parametrization and therefore have to be specified here.

        This method therefore returns a dictionary that contains as keys some arguments of the function
        `image_to_features_mapping` and as values the values necessary for providing the model with the required input.
        """
        return {}

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_raw_features_mapping` is used for training and transforms an image into raw features that can be
        further processed through augmentation or batching. It should not be used when running inference, i.e. when
        running the model in a pipeline component.
        """
        return ""

    @staticmethod
    def image_to_features_mapping() -> str:
        """Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_features_mapping` is a mapping function that converts a single image into ready features that can
        be directly fed into the model. We use this function to determine the input format of the model in a pipeline
        component. Note that this function will also require specific parameters, which can be specified in
        `default_kwargs_for_image_to_features_mapping`.

        """
        return ""


class LanguageDetector(PredictorBase):
    """
    Abstract base class for language detectors. The `predict` accepts a string of arbitrary length and returns an
    ISO-639 code for the detected language.
    """

    _categories: Mapping[str, ObjectTypes]

    @property
    def categories(self) -> Mapping[str, ObjectTypes]:
        """categories"""
        return self._categories

    @categories.setter
    def categories(self, categories: Mapping[str, TypeOrStr]) -> None:
        """categories setter"""
        self._categories = {key: get_type(value) for key, value in categories.items()}

    @abstractmethod
    def predict(self, text_string: str) -> DetectionResult:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    def get_category_names(self) -> list[ObjectTypes]:
        """
        Returns a list of possible detectable languages
        """
        return list(self.categories.values())


class ImageTransformer(PredictorBase):
    """
    Abstract base class for transforming an image. The `transform` accepts and returns a numpy array
    """

    @abstractmethod
    def transform(self, np_img: PixelValues, specification: DetectionResult) -> PixelValues:
        """
        Abstract method transform
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, np_img: PixelValues) -> DetectionResult:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    def clone(self) -> PredictorBase:
        return self.__class__()

    @staticmethod
    @abstractmethod
    def get_category_name() -> ObjectTypes:
        """
        Returns a (single) category the `ImageTransformer` can predict
        """
        raise NotImplementedError()
