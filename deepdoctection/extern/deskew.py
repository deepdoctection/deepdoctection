# -*- coding: utf-8 -*-
# File: deskew.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
jdeskew estimator and rotator to deskew images: <https://github.com/phamquiluan/jdeskew>
"""

from typing import List

from lazy_imports import try_import

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import get_jdeskew_requirement
from ..utils.settings import PageType
from ..utils.viz import viz_handler
from .base import DetectionResult, ImageTransformer

with try_import() as import_guard:
    from jdeskew.estimator import get_angle


class Jdeskewer(ImageTransformer):
    """
    Deskew an image following <https://phamquiluan.github.io/files/paper2.pdf>. It allows to determine that deskew angle
    up to 45 degrees and provides the corresponding rotation so that text lines range horizontally.
    """

    def __init__(self, min_angle_rotation: float = 2.0):
        self.name = "jdeskewer"
        self.model_id = self.get_model_id()
        self.min_angle_rotation = min_angle_rotation

    def transform(self, np_img: ImageType, specification: DetectionResult) -> ImageType:
        """
        Rotation of the image according to the angle determined by the jdeskew estimator.

        **Example**:
                    jdeskew_predictor = Jdeskewer()
                    detection_result = jdeskew_predictor.predict(np_image)
                    jdeskew_predictor.transform(np_image, DetectionResult(angle=5.0))

        :param np_img: image as numpy array
        :param specification: DetectionResult with angle value
        :return: image rotated by the angle
        """
        if abs(specification.angle) > self.min_angle_rotation:  # type: ignore
            return viz_handler.rotate_image(np_img, specification.angle)  # type: ignore
        return np_img

    def predict(self, np_img: ImageType) -> DetectionResult:
        """
        Predict the angle of the image to deskew it.

        :param np_img: image as numpy array
        :return: DetectionResult with angle value
        """
        return DetectionResult(angle=round(float(get_angle(np_img)), 4))

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        """
        Get a list of requirements for running the detector
        """
        return [get_jdeskew_requirement()]

    @staticmethod
    def possible_category() -> PageType:
        return PageType.angle
