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
<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/imgaug/transform.py> .
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
from numpy import float32

from .detection_types import ImageType

__all__ = ["ResizeTransform", "InferenceResize", "PadTransform"]


class BaseTransform(ABC):
    """
    A deterministic image transformation. This class is also the place to provide a default implementation to any
    `apply_xxx` method. The current default is to raise NotImplementedError in any such methods.
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
        :param interp: cv2 interpolation method like cv2.INTER_NEAREST, cv2.INTER_LINEAR,
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
    the inference version of `extern.tp.frcnn.common.CustomResize` .
    """

    def __init__(self, short_edge_length: int, max_size: int, interp: str = cv2.INTER_LINEAR) -> None:
        """
        :param short_edge_length: a [min, max] interval from which to sample the shortest edge length.
        :param max_size: maximum allowed longest edge length.
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


def normalize_image(image: ImageType, pixel_mean: npt.NDArray[float32], pixel_std: npt.NDArray[float32]) -> ImageType:
    """
    Preprocess pixel values of an image by rescaling.

    :param image: image as np.array
    :param pixel_mean: (3,) array
    :param pixel_std: (3,) array
    """
    return (image - pixel_mean) * (1.0 / pixel_std)


def pad_image(image: ImageType, top: int, right: int, bottom: int, left: int) -> ImageType:
    """Pad an image with white color and with given top/bottom/right/left pixel values. Only white padding is
    currently supported

    :param image: image as np.array
    :param top: Top pixel value to pad
    :param right: Right pixel value to pad
    :param bottom: Bottom pixel value to pad
    :param left: Left pixel value to pad
    """
    return np.pad(image, ((left, right), (top, bottom), (0, 0)), "constant", constant_values=(255))


class PadTransform(BaseTransform):
    """
    A transform for padding images left/right/top/bottom-wise.
    """

    def __init__(
        self,
        top: int,
        right: int,
        bottom: int,
        left: int,
        mode: Literal["xyxy", "xywh"] = "xyxy",
    ):
        """
        :param top: padding top image side
        :param right: padding right image side
        :param bottom: padding bottom image side
        :param left: padding left image side
        :param mode: bounding box mode. Needed for transforming coordinates.
        """
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.mode = mode

    def apply_image(self, img: ImageType) -> ImageType:
        """Apply padding to image"""
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        return pad_image(img, self.top, self.right, self.bottom, self.left)

    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """Transformation that should be applied to coordinates"""
        if self.mode == "xyxy":
            coords[:, 0] = coords[:, 0] + self.left
            coords[:, 1] = coords[:, 1] + self.top
            coords[:, 2] = coords[:, 2] + self.left
            coords[:, 3] = coords[:, 3] + self.top
        else:
            coords[:, 0] = coords[:, 0] + self.left
            coords[:, 1] = coords[:, 1] + self.top
        return coords

    def inverse_apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """Inverse transformation going back from coordinates of padded image to original image"""
        if self.image_height is None or self.image_width is None:
            raise ValueError("Initialize image_width and image_height first")

        if self.mode == "xyxy":
            coords[:, 0] = np.maximum(coords[:, 0] - self.left, np.zeros(coords[:, 0].shape))
            coords[:, 1] = np.maximum(coords[:, 1] - self.top, np.zeros(coords[:, 1].shape))
            coords[:, 2] = np.minimum(coords[:, 2] - self.left, np.ones(coords[:, 2].shape) * self.image_width)
            coords[:, 3] = np.minimum(coords[:, 3] - self.top, np.ones(coords[:, 3].shape) * self.image_height)
        else:
            coords[:, 0] = np.maximum(coords[:, 0] - self.left, np.zeros(coords[:, 0].shape))
            coords[:, 1] = np.maximum(coords[:, 1] - self.top, np.zeros(coords[:, 1].shape))
        return coords

    def clone(self) -> "PadTransform":
        """clone"""
        return self.__class__(self.top, self.right, self.bottom, self.left, self.mode)
