# -*- coding: utf-8 -*-
# File: transform.py

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
Module for deterministic image transformations and the sometimes necessary recalculation
of coordinates. Most have the ideas have been taken from
https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/imgaug/transform.py .
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from numpy import float32
import cv2

from ..utils.detection_types import ImageType


class BaseTransform(ABC):
    """
    A deterministic image transformation. This class is also the place to provide a default implementation to any
    :meth:`apply_xxx` method. The current default is to raise NotImplementedError in any such methods.
    All subclasses should implement `apply_image`. The image should be of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255]. Some subclasses may implement `apply_coords`, when applicable.
    It should take and return a numpy array of Nx2, where each row is the (x, y) coordinate.
    The implementation of each method may choose to modify its input data in-place for efficient transformation.
    """

    @abstractmethod
    def apply_image(self, img: ImageType) -> ImageType:
        raise NotImplementedError


class ResizeTransform(BaseTransform):
    """
    Resize the image.
    """
    def __init__(self, h: int, w: int, new_h: int, new_w: int, interp):
        """
        :param h: height
        :param w: width
        :param new_h: target height
        :param new_w: target width
        :param: interp: cv2 interpolation method like cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                        cv2.INTER_AREA
        """
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, img: ImageType) -> ImageType:
        assert img.shape[:2] == (self.h, self.w)
        ret = cv2.resize(
            img, (self.new_w, self.new_h),
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords
