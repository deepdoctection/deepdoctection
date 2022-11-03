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
jdeskew estimator and rotator to deskew images: https://github.com/phamquiluan/jdeskew
"""

from typing import List

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import get_jdeskew_requirement, jdeskew_available
from .base import ImageTransformer

if jdeskew_available():
    from jdeskew.estimator import get_angle
    from jdeskew.utility import rotate


class Jdeskewer(ImageTransformer):
    """
    Deskew an image following https://phamquiluan.github.io/files/paper2.pdf . It allows to determine that deskew angle
    up to 45 degrees and provides the corresponding rotation so that text lines range horizontally.
    """

    def __init__(self, min_angle_rotation: float = 2.0):
        self.name = "jdeskew_transform"
        self.min_angle_rotation = min_angle_rotation

    def transform(self, np_img: ImageType) -> ImageType:
        angle = get_angle(np_img)

        if angle > self.min_angle_rotation:
            return rotate(np_img, angle)
        return np_img

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        """
        Get a list of requirements for running the detector
        """
        return [get_jdeskew_requirement()]
