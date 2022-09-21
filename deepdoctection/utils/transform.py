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
from typing import Union

import cv2
import numpy as np
import numpy.typing as npt
from numpy import float32

from .detection_types import ImageType

__all__ = ["ResizeTransform", "InferenceResize"]


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
        """The transformation that should be applied to the image"""
        raise NotImplementedError


class ResizeTransform(BaseTransform):
    """
    Resize the image.
    """

    def __init__(
        self,
        h: Union[int, float],
        w: Union[int, float],
        new_h: Union[int, float],
        new_w: Union[int, float],
        interp: str,
    ):
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
        ret = cv2.resize(img, (self.new_w, self.new_h), interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """Transformation that should be applied to coordinates"""
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


class InferenceResize:
    """
    Try resizing the shortest edge to a certain number while avoiding the longest edge to exceed max_size. This is
    the inference version of :class:`extern.tp.frcnn.common.CustomResize` .
    """

    def __init__(self, short_edge_length: int, max_size: int, interp: str = cv2.INTER_LINEAR) -> None:
        """
        :param short_edge_length ([int, int]): a [min, max] interval from which to sample the shortest edge length.
        :param max_size (int): maximum allowed longest edge length.
        """
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp

    def get_transform(self, img: ImageType) -> ResizeTransform:
        """
        get transform
        """
        h, w = img.shape[:2]
        new_w: Union[int, float]
        new_h: Union[int, float]

        scale = self.short_edge_length * 1.0 / min(h, w)

        if h < w:
            new_h, new_w = self.short_edge_length, scale * w
        else:
            new_h, new_w = scale * h, self.short_edge_length
        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h = new_h * scale
            new_w = new_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return ResizeTransform(h, w, new_h, new_w, self.interp)
