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
from typing import List, Optional

from ..utils.detection_types import ImageType, Requirement

__all__ = ["PredictorBase", "ObjectDetector", "DetectionResult"]


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


@dataclass
class DetectionResult:
    """
    Simple mutable storage for detection results.

    :attr:`box`: [ulx,uly,lrx,lry]

    :attr:`score`: prediction score

    :attr:`class_id`: category id

    :attr:`mask`: binary mask

    :attr:`class_name`: category name

    :attr:`text`: text string. Used for OCR predictors

    :attr:`block`: block number. For reading order from some ocr predictors

    :attr:`line`: line number. For reading order from some ocr predictors
    """

    box: List[float]
    score: float
    class_id: int
    mask: Optional[List[float]] = None
    class_name: str = ""
    text: Optional[str] = None
    block: Optional[str] = None
    line: Optional[str] = None


class ObjectDetector(PredictorBase):  # pylint: disable=R0903
    """
    Abstract base class. Use this to connect external detectors with Deep-Doctection predictors on images.

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
