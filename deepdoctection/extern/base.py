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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple, Union

from ..utils.detection_types import ImageType, Requirement
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type


class PredictorBase(ABC):
    """
    Abstract base class for all types of predictors (e.g. object detectors language models, ...)
    """

    name: str

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
    def get_requirements(cls) -> List[Requirement]:
        """
        Get a list of requirements for running the detector
        """
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> "PredictorBase":
        """
        Clone an instance
        """
        raise NotImplementedError


@dataclass
class DetectionResult:
    """
    Simple mutable storage for detection results.

    :attr:`box`: [ulx,uly,lrx,lry]

    :attr:`class_id`: category id

    :attr:`score`: prediction score

    :attr:`mask`: binary mask

    :attr:`absolute_coords` : absolute coordinates

    :attr:`class_name`: category name

    :attr:`text`: text string. Used for OCR predictors

    :attr:`block`: block number. For reading order from some ocr predictors

    :attr:`line`: line number. For reading order from some ocr predictors

    :attr:`uuid`: uuid. For assigning detection result (e.g. text to image annotations)
    """

    box: Optional[List[float]] = None
    class_id: Optional[int] = None
    score: Optional[float] = None
    mask: Optional[List[float]] = None
    absolute_coords: bool = True
    class_name: ObjectTypes = DefaultType.default_type
    text: Optional[Union[str, ObjectTypes]] = None
    block: Optional[str] = None
    line: Optional[str] = None
    uuid: Optional[str] = None


class ObjectDetector(PredictorBase):
    """
    Abstract base class for object detection. This can be anything ranging from layout detection to OCR.
    Use this to connect external detectors with Deep-Doctection predictors on images.

    **Example:**

        .. code-block:: python

            MyFancyTensorpackPredictor(TensorpackPredictor,ObjectDetector)

    and implement the :meth:`predict`.
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
    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in :meth:`predict`
        """
        return False

    def possible_categories(self) -> List[ObjectTypes]:
        """
        Abstract method possible_categories. Must implement a method that returns a list of possible detectable
        categories
        """
        return list(self.categories.values())


class PdfMiner(PredictorBase):
    """
    Abstract base class for mining information from PDF documents. Reads in a bytes stream from a PDF document page.
    Use this to connect external pdf miners and wrap them into Deep-Doctection predictors.
    """

    _categories: Mapping[str, ObjectTypes]
    _pdf_bytes: Optional[bytes] = None
    _page: Any = None

    @property
    def categories(self) -> Mapping[str, ObjectTypes]:
        """categories"""
        return self._categories

    @categories.setter
    def categories(self, categories: Mapping[str, TypeOrStr]) -> None:
        self._categories = {key: get_type(value) for key, value in categories.items()}

    @abstractmethod
    def predict(self, pdf_bytes: bytes) -> List[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError

    @abstractmethod
    def get_width_height(self, pdf_bytes: bytes) -> Tuple[float, float]:
        """
        Abstract method get_width_height
        """
        raise NotImplementedError

    def clone(self) -> PredictorBase:
        return self.__class__()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in :meth:`predict`
        """
        return False

    def possible_categories(self) -> List[ObjectTypes]:
        """
        Returns a list of possible detectable categories
        """
        return list(self.categories.values())


class TextRecognizer(PredictorBase):
    """
    Abstract base class for text recognition. In contrast to ObjectDetector one assumes that :meth:`predict` accepts
    batches of numpy arrays. More precisely, when using :meth:`predict` pass a list of tuples with uuids (e.g. image_id,
    or annotation_id) or numpy arrays.
    """

    @abstractmethod
    def predict(self, images: List[Tuple[str, ImageType]]) -> List[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in :meth:`predict`
        """
        return True


@dataclass
class TokenClassResult:
    """
    Simple mutable storage for token classification results

     :attr:`id`: uuid of token (not unique)

     :attr:`token_id`: token id

     :attr:`token`: token

     :attr:`class_id`: category id

     :attr:`class_name`: category name

     :attr:`semantic_name`: semantic name

     :attr:`bio_tag`: bio tag

     :attr:`score`: prediction score
    """

    uuid: str
    token_id: int
    token: str
    class_id: int
    class_name: ObjectTypes = DefaultType.default_type
    semantic_name: ObjectTypes = DefaultType.default_type
    bio_tag: ObjectTypes = DefaultType.default_type
    score: Optional[float] = None


@dataclass
class SequenceClassResult:
    """
    Storage for sequence classification results

    :attr:`class_id`: category id
    :attr:`class_name`: category name
    :attr:`score`: prediction score
    """

    class_id: int
    class_name: ObjectTypes = DefaultType.default_type
    score: Optional[float] = None


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
    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[TokenClassResult]:  # type: ignore
        """
        Abstract method predict
        """
        raise NotImplementedError

    def possible_tokens(self) -> List[ObjectTypes]:
        """
        Returns a list of possible detectable tokens
        """
        return list(self.categories.values())

    @abstractmethod
    def clone(self) -> "LMTokenClassifier":
        """
        Clone an instance
        """
        raise NotImplementedError


class LMSequenceClassifier(PredictorBase):
    """
    Abstract base class for sequence classification. If you want to connect external sequence classifiers with
    Deepdoctection predictors, wrap them into a class derived from this class.
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
    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> SequenceClassResult:  # type: ignore
        """
        Abstract method predict
        """
        raise NotImplementedError

    def possible_categories(self) -> List[ObjectTypes]:
        """
        Returns a list of possible detectable categories for a sequence
        """
        return list(self.categories.values())

    @abstractmethod
    def clone(self) -> "LMSequenceClassifier":
        """
        Clone an instance
        """
        raise NotImplementedError


class LanguageDetector(PredictorBase):
    """
    Abstract base class for language detectors. The :meth:`predict` accepts a string of arbitrary length and returns an
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
        raise NotImplementedError

    def possible_languages(self) -> List[ObjectTypes]:
        """
        Returns a list of possible detectable languages
        """
        return list(self.categories.values())


class ImageTransformer(PredictorBase):
    """
    Abstract base class for transforming an image. The :meth:`transform` accepts a numpy array and returns the same.
    """

    @abstractmethod
    def transform(self, np_img: ImageType) -> ImageType:
        """
        Abstract method transform
        """
        raise NotImplementedError

    def clone(self) -> PredictorBase:
        return self.__class__()
