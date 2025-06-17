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
Jdeskew estimator and rotator: <https://github.com/phamquiluan/jdeskew>
"""

from __future__ import annotations

from lazy_imports import try_import

from ..utils.file_utils import get_jdeskew_requirement
from ..utils.settings import ObjectTypes, PageType
from ..utils.types import PixelValues, Requirement
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

    def transform_image(self, np_img: PixelValues, specification: DetectionResult) -> PixelValues:
        """
        Rotation of the image according to the angle determined by the jdeskew estimator.

        Example:
            ```python
            jdeskew_predictor = Jdeskewer()
            detection_result = jdeskew_predictor.predict(np_image)
            jdeskew_predictor.transform(np_image, DetectionResult(angle=5.0))
            ```

        Args:
            np_img: image as `np.array`
            specification: `DetectionResult` with angle value

        Returns:
            image rotated by the angle
        """
        if abs(specification.angle) > self.min_angle_rotation:  # type: ignore
            return viz_handler.rotate_image(np_img, specification.angle)  # type: ignore
        return np_img

    def predict(self, np_img: PixelValues) -> DetectionResult:
        """
        Predict the angle of the image to deskew it.

        Args:
            np_img: image as `np.array`

        Returns:
            `DetectionResult` with angle value
        """
        return DetectionResult(angle=round(float(get_angle(np_img)), 4))

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        """
        Get a list of requirements for running the detector
        """
        return [get_jdeskew_requirement()]

    def clone(self) -> Jdeskewer:
        return self.__class__(self.min_angle_rotation)

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return (PageType.ANGLE,)
