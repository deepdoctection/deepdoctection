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
Abstract classes for unifying external base- and DDoctection predictors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..utils.detection_types import ImageType, Requirement


class PredictorBase(ABC):  # pylint: disable=R0903
    """
    Abstract base class for all types of predictors (e.g. object detectors language models, ...)
    """

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

    :attr:`class_name`: category name

    :attr:`text`: text string. Used for OCR predictors

    :attr:`block`: block number. For reading order from some ocr predictors

    :attr:`line`: line number. For reading order from some ocr predictors
    """

    box: List[float]
    class_id: int
    score: Optional[float] = None
    mask: Optional[List[float]] = None
    class_name: str = ""
    text: Optional[str] = None
    block: Optional[str] = None
    line: Optional[str] = None


class ObjectDetector(PredictorBase):  # pylint: disable=R0903
    """
    Abstract base class for object detection. This can be anything ranging from layout detection to OCR.
    Use this to connect external detectors with Deep-Doctection predictors on images.

    **Example:**

        .. code-block:: python

            MyFancyTensorpackPredictor(TensorpackPredictor,ObjectDetector)

    and implement the :meth:`predict`.
    """

    @abstractmethod
    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError


class PdfMiner(PredictorBase):
    """
    Abstract base class for mining information from PDF documents. Reads in a bytes stream from a PDF document page.
    Use this to connect external pdf miners and wrap them into Deep-Doctection predictors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pdf_bytes: Optional[bytes] = None
        self._page: Any = None

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
    """

    uuid: str
    token_id: int
    token: str
    class_id: int
    class_name: str = ""
    semantic_name: str = ""
    bio_tag: str = ""


class LMTokenClassifier(PredictorBase):
    """
    Abstract base class for token classifiers. If you want to connect external token classifiers with Deep-Doctection
    predictors wrap them into a class derived from this class. Note, that this class is still DL library agnostic.
    """

    @abstractmethod
    def predict(self, **encodings: str) -> List[TokenClassResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError
