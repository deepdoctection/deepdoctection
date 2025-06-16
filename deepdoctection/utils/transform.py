# -*- coding: utf-8 -*-
# File: test_transform.py

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

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Literal, Optional, Set, Union

import numpy as np
import numpy.typing as npt
from numpy import float32

from .settings import ObjectTypes, PageType
from .types import PixelValues
from .viz import viz_handler

__all__ = [
    "point4_to_box",
    "box_to_point4",
    "ResizeTransform",
    "InferenceResize",
    "PadTransform",
    "normalize_image",
    "pad_image",
    "BaseTransform",
    "RotationTransform",
]


def box_to_point4(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    box = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    box = box.reshape((-1, 2))
    return box


def point4_to_box(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Args:
        points: (nx4)x2

    Returns:
        nx4 boxes (`x1y1x2y2`)
    """
    points = points.reshape((-1, 4, 2))
    min_xy = points.min(axis=1)  # nx2
    max_xy = points.max(axis=1)  # nx2
    return np.concatenate((min_xy, max_xy), axis=1)


class BaseTransform(ABC):
    """
    A deterministic image transformation. This class is also the place to provide a default implementation to any
    `apply_xxx` method. The current default is to raise `NotImplementedError` in any such methods.
    All subclasses should implement `apply_image`. The image should be of type `uint8` in range [0, 255], or
    floating point images in range [0, 1] or [0, 255]. Some subclasses may implement `apply_coords`, when applicable.
    It should take and return a numpy array of Nx2, where each row is the (x, y) coordinate.
    The implementation of each method may choose to modify its input data in-place for efficient transformation.

    Note:
        All subclasses should implement `apply_image`. Some may implement `apply_coords`.
    """

    @abstractmethod
    def apply_image(self, img: PixelValues) -> PixelValues:
        """
        The transformation that should be applied to the image.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Transformation that should be applied to coordinates. Coords are supposed to be passed as like

        ```python
        np.array([[ulx_0,uly_0,lrx_0,lry_0],[ulx_1,uly_1,lrx_1,lry_1],...])
        ```

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError()

    @abstractmethod
    def inverse_apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Inverse transformation going back from coordinates of transformed image to original image. Coords are
        supposed to be passed as like

        ```python
        np.array([[ulx_0,uly_0,lrx_0,lry_0],[ulx_1,uly_1,lrx_1,lry_1],...])
        ```

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """Get category names"""
        raise NotImplementedError()

    def get_init_args(self) -> Set[str]:
        """Return the names of the arguments of the constructor."""
        args = inspect.signature(self.__init__).parameters.keys()  # type: ignore
        return {arg for arg in args if arg != "self"}


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
        Args:
            h: Height.
            w: Width.
            new_h: Target height.
            new_w: Target width.
            interp: Interpolation method, that depends on the image processing library. Currently, it supports
                `NEAREST`, `BOX`, `BILINEAR`, `BICUBIC` and `VIZ` for PIL or `INTER_NEAREST`, `INTER_LINEAR`,
                `INTER_AREA` or `VIZ` for OpenCV.
        """
        self.h = h
        self.w = w
        self.new_h = int(new_h)
        self.new_w = int(new_w)
        self.interp = interp

    def apply_image(self, img: PixelValues) -> PixelValues:
        """
        Apply the resize transformation to the image.

        Args:
            img: Image to be resized.

        Returns:
            Resized image.

        Raises:
            AssertionError: If the input image shape does not match the expected height and width.
        """
        assert img.shape[:2] == (self.h, self.w)
        ret = viz_handler.resize(img, self.new_w, self.new_h, self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Transformation that should be applied to coordinates. Coords are supposed to be passed as
        numpy array of points.

        Args:
            coords: Coordinates to be transformed.

        Returns:
            Transformed coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def inverse_apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Inverse transformation going back from coordinates of resized image to original image.

        Args:
            coords: Coordinates to be inversely transformed.

        Returns:
            Inversely transformed coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.w * 1.0 / self.new_w)
        coords[:, 1] = coords[:, 1] * (self.h * 1.0 / self.new_h)
        return coords

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """Get category names"""
        return (PageType.SIZE,)


class InferenceResize:
    """
    Try resizing the shortest edge to a certain number while avoiding the longest edge to exceed max_size. This is
    the inference version of `extern.tp.frcnn.common.CustomResize` .
    """

    def __init__(self, short_edge_length: int, max_size: int, interp: str = "VIZ") -> None:
        """
        Args:
            short_edge_length: A [min, max] interval from which to sample the shortest edge length.
            max_size: Maximum allowed longest edge length.
            interp: Interpolation method.
        """
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp

    def get_transform(self, img: PixelValues) -> ResizeTransform:
        """
        Get the `ResizeTransform` for the image.

        Args:
            img: Image to be transformed.

        Returns:
            `ResizeTransform` object.
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


def normalize_image(
    image: PixelValues, pixel_mean: npt.NDArray[float32], pixel_std: npt.NDArray[float32]
) -> PixelValues:
    """
    Preprocess pixel values of an image by rescaling.

    Args:
        image: Image as numpy array.
        pixel_mean: (3,) array.
        pixel_std: (3,) array.

    Returns:
        Normalized image.
    """
    return (image - pixel_mean) * (1.0 / pixel_std)


def pad_image(image: PixelValues, top: int, right: int, bottom: int, left: int) -> PixelValues:
    """
    Pad an image with white color and with given top/bottom/right/left pixel values. Only white padding is
    currently supported.

    Args:
        image: Image as numpy array.
        top: Top pixel value to pad.
        right: Right pixel value to pad.
        bottom: Bottom pixel value to pad.
        left: Left pixel value to pad.

    Returns:
        Padded image.
    """
    return np.pad(image, ((top, bottom), (left, right), (0, 0)), "constant", constant_values=255)


class PadTransform(BaseTransform):
    """
    A transform for padding images left/right/top/bottom-wise.
    """

    def __init__(
        self,
        pad_top: int,
        pad_right: int,
        pad_bottom: int,
        pad_left: int,
    ):
        """
        A transform for padding images left/right/top/bottom-wise.

        Args:
            pad_top: Padding top image side.
            pad_right: Padding right image side.
            pad_bottom: Padding bottom image side.
            pad_left: Padding left image side.
        """

        self.pad_top = pad_top
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None

    def apply_image(self, img: PixelValues) -> PixelValues:
        """
        Apply padding to image.

        Args:
            img: Image to be padded.

        Returns:
            Padded image.
        """
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        return pad_image(img, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left)

    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Transformation that should be applied to coordinates.

        Args:
            coords: Coordinates to be transformed.

        Returns:
            Transformed coordinates.
        """
        coords[:, 0] = coords[:, 0] + self.pad_left
        coords[:, 1] = coords[:, 1] + self.pad_top
        coords[:, 2] = coords[:, 2] + self.pad_left
        coords[:, 3] = coords[:, 3] + self.pad_top
        return coords

    def inverse_apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Inverse transformation going back from coordinates of padded image to original image.

        Args:
            coords: Coordinates to be inversely transformed.

        Returns:
            Inversely transformed coordinates.

        Raises:
            ValueError: If `image_width` and `image_height` are not initialized.
        """
        if self.image_height is None or self.image_width is None:
            raise ValueError("Initialize image_width and image_height first")
        coords[:, 0] = np.maximum(coords[:, 0] - self.pad_left, np.zeros(coords[:, 0].shape))
        coords[:, 1] = np.maximum(coords[:, 1] - self.pad_top, np.zeros(coords[:, 1].shape))
        coords[:, 2] = np.minimum(coords[:, 2] - self.pad_left, np.ones(coords[:, 2].shape) * self.image_width)
        coords[:, 3] = np.minimum(coords[:, 3] - self.pad_top, np.ones(coords[:, 3].shape) * self.image_height)
        return coords

    def clone(self) -> PadTransform:
        """clone"""
        return self.__class__(self.pad_top, self.pad_right, self.pad_bottom, self.pad_left)

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """Get category names"""
        return (
            PageType.PAD_TOP,
            PageType.PAD_RIGHT,
            PageType.PAD_LEFT,
            PageType.PAD_BOTTOM,
        )


class RotationTransform(BaseTransform):
    """
    A transform for rotating images by 90, 180, 270, or 360 degrees.
    """

    def __init__(self, angle: Literal[90, 180, 270, 360]):
        """
        Args:
            angle: Angle to rotate the image. Must be one of 90, 180, 270, or 360 degrees.
        """
        self.angle = angle
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None

    def apply_image(self, img: PixelValues) -> PixelValues:
        """
        Apply rotation to image.

        Args:
            img: Image to be rotated.

        Returns:
            Rotated image.
        """
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        return viz_handler.rotate_image(img, self.angle)

    def apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Transformation that should be applied to coordinates.

        Args:
            coords: Coordinates to be transformed.

        Returns:
            Transformed coordinates.

        Raises:
            ValueError: If `image_width` and `image_height` are not initialized.
        """
        if self.image_width is None or self.image_height is None:
            raise ValueError("Initialize image_width and image_height first")

        if self.angle == 90:
            coords[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
            coords[:, [1, 3]] = self.image_width - coords[:, [1, 3]]
            coords[:, [0, 1, 2, 3]] = coords[:, [0, 3, 2, 1]]
        elif self.angle == 180:
            coords[:, [0, 2]] = self.image_width - coords[:, [0, 2]]
            coords[:, [1, 3]] = self.image_height - coords[:, [1, 3]]
            coords[:, [0, 1, 2, 3]] = coords[:, [2, 3, 0, 1]]
        elif self.angle == 270:
            coords[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
            coords[:, [0, 2]] = self.image_height - coords[:, [0, 2]]
            coords[:, [0, 1, 2, 3]] = coords[:, [2, 1, 0, 3]]

        return coords

    def inverse_apply_coords(self, coords: npt.NDArray[float32]) -> npt.NDArray[float32]:
        """
        Inverse transformation going back from coordinates of rotated image to original image.

        Args:
            coords: Coordinates to be inversely transformed.

        Returns:
            Inversely transformed coordinates.

        Raises:
            ValueError: If `image_width` and `image_height` are not initialized.
        """
        if self.image_width is None or self.image_height is None:
            raise ValueError("Initialize image_width and image_height first")

        if self.angle == 90:
            coords[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
            coords[:, [0, 2]] = self.image_width - coords[:, [0, 2]]
            coords[:, [0, 1, 2, 3]] = coords[:, [2, 1, 0, 3]]
        elif self.angle == 180:
            coords[:, [0, 2]] = self.image_width - coords[:, [0, 2]]
            coords[:, [1, 3]] = self.image_height - coords[:, [1, 3]]
            coords[:, [0, 1, 2, 3]] = coords[:, [2, 3, 0, 1]]
        elif self.angle == 270:
            coords[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
            coords[:, [1, 3]] = self.image_height - coords[:, [1, 3]]
            coords[:, [0, 1, 2, 3]] = coords[:, [0, 3, 2, 1]]
        return coords

    def clone(self) -> RotationTransform:
        """clone"""
        return self.__class__(self.angle)

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """Get category names"""
        return (PageType.ANGLE,)
