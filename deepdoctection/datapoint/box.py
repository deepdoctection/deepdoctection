# -*- coding: utf-8 -*-
# File: box.py

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
`BoundingBox` class and methods for manipulating bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Optional, Sequence, Union, no_type_check

import numpy as np
import numpy.typing as npt
from lazy_imports import try_import
from numpy import float32

from ..utils.error import BoundingBoxError
from ..utils.file_utils import cocotools_available
from ..utils.logger import LoggingRecord, logger
from ..utils.types import BoxCoordinate, PixelValues

with try_import() as import_guard:
    import pycocotools.mask as coco_mask


# taken from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/common.py


def coco_iou(box_a: npt.NDArray[float32], box_b: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Calculate iou for two arrays of bounding boxes in `xyxy` format

    Args:
        box_a: Array of shape Nx4
        box_b: Array of shape Mx4

    Returns:
        Array of shape NxM
    """

    def to_xywh(box: npt.NDArray[float32]) -> npt.NDArray[float32]:
        box = box.copy()
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]
        return box

    ret = coco_mask.iou(to_xywh(box_a), to_xywh(box_b), np.zeros((len(box_b),), dtype=bool))
    # can accelerate even more, if using float32
    return ret.astype("float32")


# taken from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/np_box_ops.py


def area(boxes: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Computes area of boxes.

    Args:
        boxes: numpy array with shape [N, 4] holding N boxes in xyxy format

    Returns:
        A numpy array with shape `[N*1]` representing box areas
    """
    return np.array((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=float32)


# taken from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/np_box_ops.py


def intersection(boxes1: npt.NDArray[float32], boxes2: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Compute pairwise intersection areas between boxes.

    Args:
        boxes1: A `np.array` with shape `[N, 4]` holding `N` boxes in `xyxy` format
        boxes2: A `np.array` with shape `[M, 4]` holding `M` boxes in `xyxy` format

    Returns:
        A `np.array` with shape `[N*M]` representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)  # pylint: disable=W0632
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)  # pylint: disable=W0632

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape, dtype="f4"), all_pairs_min_ymax - all_pairs_max_ymin
    )
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape, dtype="f4"), all_pairs_min_xmax - all_pairs_max_xmin
    )
    return intersect_heights * intersect_widths


# taken from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/np_box_ops.py


def np_iou(boxes1: npt.NDArray[float32], boxes2: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes in xyxy format.
        boxes2: a numpy array with shape [M, 4] holding M boxes in xyxy format.

    Returns:
        A `np.array` with shape `[N, M]` representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect

    return intersect / union


def iou(boxes1: npt.NDArray[float32], boxes2: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Computes pairwise intersection-over-union between box collections.

    Note:
        The method will be chosen based on what is installed:

        - If `pycocotools` is installed it will choose `pycocotools.mask.iou` which is a `C++` implementation
          and much faster
        - Otherwise it will use the numpy implementation as fallback

    Args:
        boxes1:  A `np.array` with shape `[N, 4]` holding `N` boxes in `xyxy` format.
        boxes2:  A np.array with shape `[N, 4]` holding `N` boxes in `xyxy` format.

    Returns:
        A `np.array` with shape `[N, M]` representing pairwise iou scores.
    """

    if cocotools_available():
        return coco_iou(boxes1, boxes2)
    return np_iou(boxes1, boxes2)


RELATIVE_COORD_SCALE_FACTOR = 10**8


@dataclass
class BoundingBox:
    """
    Rectangular bounding box that stores coordinates and allows different representations.

    This implementation differs from the previous version by using internal integer storage with precision scaling
    for both absolute and relative coordinates. Coordinates are stored internally as integers `(_ulx, _uly, etc.)`
    with relative coordinates multiplied by `RELATIVE_COORD_CONVERTER` for precision. Properties `(ulx, uly, etc.)`
    handle the conversion between internal storage and exposed values.

    Note:
        You can define an instance by passing:

        - Upper left point `(ulx, uly) + width` and height, OR
        - Upper left point `(ulx, uly) + lower right point (lrx, lry)`

    Note:
        - When `absolute_coords=True`, coordinates will be rounded to integers
        - When `absolute_coords=False`, coordinates must be between 0 and 1
        - The box is validated on initialization to ensure coordinates are valid

    Attributes:
        absolute_coords: Whether the coordinates are absolute pixel values (`True`) or normalized
                                `[0,1]` values (`False`).
        _ulx: Upper-left x-coordinate, stored as an integer.
        _uly: Upper-left y-coordinate, stored as an integer.
        _lrx: Lower-right x-coordinate, stored as an integer.
        _lry: Lower-right y-coordinate, stored as an integer.
        _height: Height of the bounding box, stored as an integer.
        _width: Width of the bounding box, stored as an integer.

    """

    absolute_coords: bool
    _ulx: int = 0
    _uly: int = 0
    _lrx: int = 0
    _lry: int = 0
    _height: int = 0
    _width: int = 0

    def __init__(
        self,
        absolute_coords: bool,
        ulx: BoxCoordinate,
        uly: BoxCoordinate,
        lrx: BoxCoordinate = 0,
        lry: BoxCoordinate = 0,
        width: BoxCoordinate = 0,
        height: BoxCoordinate = 0,
    ):
        """
        Initialize a BoundingBox instance with specified coordinates.

        Note:
             This initializer supports two ways of defining a bounding box:
             - Using upper-left coordinates (ulx, uly) with width and height
             - Using upper-left (ulx, uly) and lower-right (lrx, lry) coordinates

             When `absolute_coords=True`, coordinates are stored as integers.
             When `absolute_coords=False`, coordinates are stored as scaled integers
             (original `float values * RELATIVE_COORD_SCALE_FACTOR`) for precision.

        Args:
            absolute_coords: Whether coordinates are absolute pixels (`True`) or normalized `[0,1]` values (`False`)
            ulx: Upper-left `x`-coordinate (`float` or `int`)
            uly: Upper-left `y`-coordinate (`float` or `int`)
            lrx: Lower-right `x`-coordinate (`float` or `int`), default 0
            lry: Lower-right `y`-coordinate (`float` or `int`), default 0
            width: Width of the bounding box (`float` or `int`), default 0
            height: Height of the bounding box (`float` or `int`), default 0
        """
        self.absolute_coords = absolute_coords
        if absolute_coords:
            self._ulx = round(ulx)
            self._uly = round(uly)
            if lrx and lry:
                self._lrx = round(lrx)
                self._lry = round(lry)
            if width and height:
                self._width = round(width)
                self._height = round(height)
        else:
            self._ulx = round(ulx * RELATIVE_COORD_SCALE_FACTOR)
            self._uly = round(uly * RELATIVE_COORD_SCALE_FACTOR)
            if lrx and lry:
                self._lrx = round(lrx * RELATIVE_COORD_SCALE_FACTOR)
                self._lry = round(lry * RELATIVE_COORD_SCALE_FACTOR)
            if width and height:
                self._width = round(width * RELATIVE_COORD_SCALE_FACTOR)
                self._height = round(height * RELATIVE_COORD_SCALE_FACTOR)
        if not self._width and not self._height:
            self._width = self._lrx - self._ulx
            self._height = self._lry - self._uly
        if not self._lrx and not self._lry:
            self._lrx = self._ulx + self._width
            self._lry = self._uly + self._height
        self.__post_init__()

    def __post_init__(self) -> None:
        if self._width == 0:
            if self._lrx is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self._width = self._lrx - self._ulx
        if self._height == 0:
            if self._lry is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self._height = self._lry - self._uly

        if self._lrx == 0:
            if self._width is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self._lrx = self._ulx + self._width
        if self._lry == 0:
            if self._height is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self._lry = self._uly + self._height

        if not (self._ulx >= 0 and self._uly >= 0):
            raise BoundingBoxError("Bounding box ul must be >= (0,0)")
        if not (self._height > 0 and self._width > 0):
            raise BoundingBoxError(
                f"bounding box must have height and width >0. Check coords "
                f"ulx: {self.ulx}, uly: {self.uly}, lrx: {self.lrx}, "
                f"lry: {self.lry}."
            )
        if not self.absolute_coords and not (
            0 <= self.ulx <= 1 and 0 <= self.uly <= 1 and 0 <= self.lrx <= 1 and 0 <= self.lry <= 1
        ):
            raise BoundingBoxError("coordinates must be between 0 and 1")

    @property
    def ulx(self) -> BoxCoordinate:
        """ulx property"""
        return self._ulx / RELATIVE_COORD_SCALE_FACTOR if not self.absolute_coords else self._ulx

    @ulx.setter
    def ulx(self, value: BoxCoordinate) -> None:
        """ulx setter"""
        self._ulx = round(value * RELATIVE_COORD_SCALE_FACTOR) if not self.absolute_coords else round(value)
        self._width = self._lrx - self._ulx

    @property
    def uly(self) -> BoxCoordinate:
        """uly property"""
        return self._uly / RELATIVE_COORD_SCALE_FACTOR if not self.absolute_coords else self._uly

    @uly.setter
    def uly(self, value: BoxCoordinate) -> None:
        """uly setter"""
        self._uly = round(value * RELATIVE_COORD_SCALE_FACTOR) if not self.absolute_coords else round(value)
        self._height = self._lry - self._uly

    @property
    def lrx(self) -> BoxCoordinate:
        """lrx property"""
        return self._lrx / RELATIVE_COORD_SCALE_FACTOR if not self.absolute_coords else self._lrx

    @lrx.setter
    def lrx(self, value: BoxCoordinate) -> None:
        """lrx setter"""
        self._lrx = round(value * RELATIVE_COORD_SCALE_FACTOR) if not self.absolute_coords else round(value)
        self._width = self._lrx - self._ulx

    @property
    def lry(self) -> BoxCoordinate:
        """lry property"""
        return self._lry / RELATIVE_COORD_SCALE_FACTOR if not self.absolute_coords else self._lry

    @lry.setter
    def lry(self, value: BoxCoordinate) -> None:
        """lry setter"""
        self._lry = round(value * RELATIVE_COORD_SCALE_FACTOR) if not self.absolute_coords else round(value)
        self._height = self._lry - self._uly

    @property
    def width(self) -> BoxCoordinate:
        """width property"""
        return self._width / RELATIVE_COORD_SCALE_FACTOR if not self.absolute_coords else self._width

    @width.setter
    def width(self, value: BoxCoordinate) -> None:
        """width setter"""
        self._width = round(value * RELATIVE_COORD_SCALE_FACTOR) if not self.absolute_coords else round(value)
        self._lrx = self._ulx + self._width

    @property
    def height(self) -> BoxCoordinate:
        """height property"""
        return self._height / RELATIVE_COORD_SCALE_FACTOR if not self.absolute_coords else self._height

    @height.setter
    def height(self, value: BoxCoordinate) -> None:
        """height setter"""
        self._height = round(value * RELATIVE_COORD_SCALE_FACTOR) if not self.absolute_coords else round(value)
        self._lry = self._uly + self._height

    @property
    def cx(self) -> BoxCoordinate:
        """cx property"""
        if self.absolute_coords:
            return round(self.ulx + 0.5 * self.width)
        return self.ulx + 0.5 * self.width

    @property
    def cy(self) -> BoxCoordinate:
        """cy property"""
        if self.absolute_coords:
            return round(self.uly + 0.5 * self.height)
        return self.uly + 0.5 * self.height

    @property
    def center(self) -> tuple[BoxCoordinate, BoxCoordinate]:
        """center property"""
        return (self.cx, self.cy)

    @property
    def area(self) -> Union[int, float]:
        """area property"""
        if self.absolute_coords:
            return self.width * self.height
        raise ValueError("Cannot calculate area, when bounding box coords are relative")

    def to_np_array(self, mode: str, scale_x: float = 1.0, scale_y: float = 1.0) -> npt.NDArray[np.float32]:
        """
        Returns the coordinates as numpy array.

        Args:
            mode: Mode for coordinate arrangement:
                     `xyxy` for upper left/lower right point representation,
                     `xywh` for upper left and width/height representation or
                     `poly` for full eight coordinate polygon representation. `x,y` coordinates will be
                      returned in counter-clockwise order.

            scale_x: rescale the `x` coordinate. Defaults to `1`
            scale_y: rescale the `y` coordinate. Defaults to `1`

        Returns:
            box coordinates
        """
        np_box_scale = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        np_poly_scale = np.array(
            [scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x, scale_y], dtype=np.float32
        )
        assert mode in ("xyxy", "xywh", "poly"), "Not a valid mode"
        if mode == "xyxy":
            return np.array([self.ulx, self.uly, self.lrx, self.lry], dtype=np.float32) * np_box_scale
        if mode == "xywh":
            return np.array([self.ulx, self.uly, self.width, self.height], dtype=np.float32) * np_box_scale
        return (
            np.array([self.ulx, self.uly, self.lrx, self.uly, self.lrx, self.lry, self.ulx, self.lry], dtype=np.float32)
            * np_poly_scale
        )

    def to_list(self, mode: str, scale_x: float = 1.0, scale_y: float = 1.0) -> list[BoxCoordinate]:
        """
        Returns the coordinates as list

        Args:
            mode:  Mode for coordinate arrangement:
                     `xyxy` for upper left/lower right point representation,
                     `xywh` for upper left and width/height representation or
                     `poly` for full eight coordinate polygon representation. `x,y` coordinates will be
                      returned in counter-clockwise order.

            scale_x: rescale the x coordinate. Defaults to 1
            scale_y: rescale the y coordinate. Defaults to 1

        Returns:
            box coordinates
        """
        assert mode in ("xyxy", "xywh", "poly"), "Not a valid mode"
        if mode == "xyxy":
            return (
                [
                    round(self.ulx * scale_x),
                    round(self.uly * scale_y),
                    round(self.lrx * scale_x),
                    round(self.lry * scale_y),
                ]
                if self.absolute_coords
                else [
                    self.ulx * scale_x,
                    self.uly * scale_y,
                    self.lrx * scale_x,
                    self.lry * scale_y,
                ]
            )
        if mode == "xywh":
            return (
                [
                    round(self.ulx * scale_x),
                    round(self.uly * scale_y),
                    round(self.width * scale_x),
                    round(self.height * scale_y),
                ]
                if self.absolute_coords
                else [self.ulx * scale_x, self.uly * scale_y, self.width * scale_x, self.height * scale_y]
            )
        return (
            [
                round(self.ulx * scale_x),
                round(self.uly * scale_y),
                round(self.lrx * scale_x),
                round(self.uly * scale_y),
                round(self.lrx * scale_x),
                round(self.lry * scale_y),
                round(self.ulx * scale_x),
                round(self.lry * scale_y),
            ]
            if self.absolute_coords
            else [
                self.ulx * scale_x,
                self.uly * scale_y,
                self.lrx * scale_x,
                self.uly * scale_y,
                self.lrx * scale_x,
                self.lry * scale_y,
                self.ulx * scale_x,
                self.lry * scale_y,
            ]
        )

    def transform(
        self,
        image_width: float,
        image_height: float,
        absolute_coords: bool = False,
    ) -> BoundingBox:
        """
        Transforms bounding box coordinates into absolute or relative coords. Internally, a new bounding box will be
        created. Changing coordinates requires width and height of the whole image.

        Args:
            image_width: The horizontal image size
            image_height: The vertical image size
            absolute_coords: Whether to recalculate into absolute coordinates.

        Returns:
            Either a `list` or `np.array`.
        """
        if absolute_coords != self.absolute_coords:
            if self.absolute_coords:
                transformed_box = BoundingBox(
                    absolute_coords=not self.absolute_coords,
                    ulx=max(self.ulx / image_width, 0.0),
                    uly=max(self.uly / image_height, 0.0),
                    lrx=min(self.lrx / image_width, 1.0),
                    lry=min(self.lry / image_height, 1.0),
                )
            else:
                transformed_box = BoundingBox(
                    absolute_coords=not self.absolute_coords,
                    ulx=self.ulx * image_width,
                    uly=self.uly * image_height,
                    lrx=self.lrx * image_width,
                    lry=self.lry * image_height,
                )
            return transformed_box
        return self

    def __str__(self) -> str:
        return (
            f"Bounding Box(absolute_coords: {self.absolute_coords},"
            f"ulx: {self.ulx}, uly: {self.uly}, lrx: {self.lrx}, lry: {self.lry})"
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(absolute_coords={self.absolute_coords}, ulx={self.ulx}, uly={self.uly}, lrx={self.lrx},"
            f" lry={self.lry}, width={self.width}, height={self.height})"
        )

    def get_legacy_string(self) -> str:
        """Legacy string representation of the bounding box. Do not use"""
        return f"Bounding Box ulx: {self.ulx}, uly: {self.uly}, lrx: {self.lrx}, lry: {self.lry}"

    @staticmethod
    def remove_keys() -> list[str]:
        """Removing keys when converting the dataclass object to a dict"""
        return ["_height", "_width"]

    @staticmethod
    def replace_keys() -> dict[str, str]:
        """Replacing keys when converting the dataclass object to a dict. Useful for backward compatibility"""
        return {"_ulx": "ulx", "_uly": "uly", "_lrx": "lrx", "_lry": "lry"}

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> BoundingBox:
        """from dict"""
        return cls(**kwargs)


def intersection_box(
    box_1: BoundingBox, box_2: BoundingBox, width: Optional[float] = None, height: Optional[float] = None
) -> BoundingBox:
    """
    Returns the intersection bounding box of two boxes.
    If coords are absolute, it will floor the lower and ceil the upper coord to ensure the resulting box has same
    coordinates as the box induces from `crop_box_from_image`

    Args:
        box_1: bounding box
        box_2: bounding box
        width: Total width of image. This optional parameter is needed if the value of `absolute_coords` of `box_1`
                  and `box_2` are not equal.
        height: Total height of image. This optional parameter is needed if the value of `absolute_coords` of `box_1`
                   and `box_2` are not equal.

    Returns:
        bounding box. Will have same `absolute_coords` as `box_2`, if `absolute_coords` of `box_1` and `box_2` are
             not equal#

    Raises:
        ValueError: If the intersection is empty, i.e. if the boxes do not overlap.
    """

    if box_1.absolute_coords != box_2.absolute_coords:
        # will transform box_1
        assert (
            width is not None and height is not None
        ), "when absolute coords of boxes are not equal must pass width and height"
        box_1 = box_1.transform(width, height, box_2.absolute_coords)
    ulx = max(box_1.ulx, box_2.ulx)
    uly = max(box_1.uly, box_2.uly)
    lrx = min(box_1.lrx, box_2.lrx)
    lry = min(box_1.lry, box_2.lry)
    if box_2.absolute_coords:
        ulx, uly, lrx, lry = np.floor(ulx), np.floor(uly), np.ceil(lrx), np.ceil(lry)
    return BoundingBox(box_2.absolute_coords, ulx=ulx, uly=uly, lrx=lrx, lry=lry)


def crop_box_from_image(
    np_image: PixelValues, crop_box: BoundingBox, width: Optional[float] = None, height: Optional[float] = None
) -> PixelValues:
    """
    Crop a box (the crop_box) from a image given as `np.array`. Will floor the left  and ceil the right coordinate
    point.

    Args:
        np_image: Image to crop from.
        crop_box: Bounding box to crop.
        width: Total width of image. This optional parameter is needed if the value of `absolute_coords` of
               `crop_box` is `False`.
        height: Total width of image. This optional parameter is needed if the value of `absolute_coords` of
                `crop_box` is `False`.

    :return: A `np.array` cropped according to the bounding box.
    """
    if not crop_box.absolute_coords:
        assert (
            width is not None and height is not None
        ), "when crop_box has absolute coords set to False must pass width and height"
        absolute_coord_box = crop_box.transform(width, height, absolute_coords=True)
    else:
        absolute_coord_box = crop_box

    assert isinstance(absolute_coord_box, BoundingBox)
    np_max_y, np_max_x = np_image.shape[0:2]
    return np_image[
        int(floor(absolute_coord_box.uly)) : min(int(ceil(absolute_coord_box.lry)), np_max_y),
        int(floor(absolute_coord_box.ulx)) : min(int(ceil(absolute_coord_box.lrx)), np_max_x),
    ]


def local_to_global_coords(local_box: BoundingBox, embedding_box: BoundingBox) -> BoundingBox:
    """
    Transform coords in terms of a cropped image into global coords. The local box coords are given in terms of the
    embedding box. The global coords will be determined by transforming the upper left point (which is `(0,0)` in
    local terms) into the upper left point given by the embedding box. This will shift the ul point of the
    local box to `ul + embedding_box.ul`

    Args:
        local_box: bounding box with coords in terms of an embedding (e.g. local coordinates)
        embedding_box: bounding box of the embedding.

    Returns:
        Bounding box with local box transformed to absolute coords
    """

    assert local_box.absolute_coords and embedding_box.absolute_coords, (
        f"absolute coords "
        f"(={local_box.absolute_coords} for local_box and embedding_box (={embedding_box.absolute_coords}) must be "
        f"True"
    )
    assert embedding_box.ulx is not None and embedding_box.uly is not None
    assert (
        local_box.ulx is not None
        and local_box.uly is not None
        and local_box.lrx is not None
        and local_box.lry is not None
    )
    return BoundingBox(
        absolute_coords=True,
        ulx=embedding_box.ulx + local_box.ulx,
        uly=embedding_box.uly + local_box.uly,
        lrx=embedding_box.ulx + local_box.lrx,
        lry=embedding_box.uly + local_box.lry,
    )


def global_to_local_coords(global_box: BoundingBox, embedding_box: BoundingBox) -> BoundingBox:
    """
    Transforming global bounding box coords into the coordinate system given by the embedding box. The transformation
    requires that the global bounding box coordinates lie completely within the rectangle of the embedding box.
    The transformation results from a shift of all coordinates given by the shift of the upper left point of the
    embedding box into `(0,0)`.

    Args:
        global_box: The bounding box to be embedded
        embedding_box: The embedding box. Must cover the global box completely.

    Returns:
        Bounding box of the embedded box in local coordinates.
    """

    assert global_box.absolute_coords and embedding_box.absolute_coords, (
        f"absolute coords "
        f"(={global_box.absolute_coords} for local_box and embedding_box (={embedding_box.absolute_coords}) must be "
        f"True"
    )

    return BoundingBox(
        absolute_coords=True,
        ulx=max(global_box.ulx - embedding_box.ulx, 0),
        uly=max(global_box.uly - embedding_box.uly, 0),
        lrx=min(global_box.lrx - embedding_box.ulx, embedding_box.width),
        lry=min(global_box.lry - embedding_box.uly, embedding_box.height),
    )


def merge_boxes(*boxes: BoundingBox) -> BoundingBox:
    """
    Generating the smallest box containing an arbitrary tuple/list of boxes.

    Args:
        boxes: An arbitrary tuple/list of bounding boxes `BoundingBox`.
    """
    absolute_coords = boxes[0].absolute_coords
    assert all(box.absolute_coords == absolute_coords for box in boxes), "all boxes must have same absolute_coords"

    ulx = min(box.ulx for box in boxes)
    uly = min(box.uly for box in boxes)
    lrx = max(box.lrx for box in boxes)
    lry = max(box.lry for box in boxes)

    return BoundingBox(absolute_coords=absolute_coords, ulx=ulx, uly=uly, lrx=lrx, lry=lry)


def rescale_coords(
    box: BoundingBox,
    current_total_width: float,
    current_total_height: float,
    scaled_total_width: float,
    scaled_total_height: float,
) -> BoundingBox:
    """
    Generating a bounding box with scaled coordinates. Will rescale `x` coordinate with factor
    `*(current_total_width/scaled_total_width)`, resp. `y` coordinate with factor
    `* (current_total_height/scaled_total_height)`,

    while not changing anything if `absolute_coords` is set to `False`.

    Args:
        box: BoudingBox to rescale
        current_total_width: absolute coords of width of image
        current_total_height: absolute coords of height of image
        scaled_total_width:  absolute width of rescaled image
        scaled_total_height: absolute height of rescaled image

    Returns:
        rescaled `BoundingBox`
    """

    if not box.absolute_coords:
        return box
    scale_width = scaled_total_width / current_total_width
    scale_height = scaled_total_height / current_total_height

    return BoundingBox(
        absolute_coords=True,
        ulx=box.ulx * scale_width,
        uly=box.uly * scale_height,
        lrx=box.lrx * scale_width,
        lry=box.lry * scale_height,
    )


def intersection_boxes(boxes_1: Sequence[BoundingBox], boxes_2: Sequence[BoundingBox]) -> Sequence[BoundingBox]:
    """
    The multiple version of `intersection_box`: Given two lists of `m` and `n` bounding boxes, it will calculate the
    pairwise intersection of both groups. There will be at most `mxn` intersection boxes.

    Args:
        boxes_1: sequence of m BoundingBox
        boxes_2: sequence of n BoundingBox

    Returns:
        list of at most mxn BoundingBox
    """
    if not boxes_1 and boxes_2:
        return boxes_2
    if not boxes_2 and boxes_1:
        return boxes_1
    if not boxes_1 and not boxes_2:
        return []
    if boxes_1[0].absolute_coords != boxes_2[0].absolute_coords:
        raise ValueError("absolute_coords of boxes_1 and boxes_2 mus be equal")
    absolute_coords = boxes_1[0].absolute_coords
    boxes1 = np.array([box.to_list(mode="xyxy") for box in boxes_1])
    boxes2 = np.array([box.to_list(mode="xyxy") for box in boxes_2])
    [x_min1, y_min1, x_max1, y_max1] = np.split(boxes1, 4, axis=1)  # pylint: disable=W0632
    [x_min2, y_min2, x_max2, y_max2] = np.split(boxes2, 4, axis=1)  # pylint: disable=W0632

    ulys = np.maximum(y_min1, np.transpose(y_min2))
    lrys = np.minimum(y_max1, np.transpose(y_max2))
    intersect_heights = np.maximum(np.zeros(ulys.shape, dtype="f4"), lrys - ulys).flatten()

    ulxs = np.maximum(x_min1, np.transpose(x_min2))
    lrxs = np.minimum(x_max1, np.transpose(x_max2))
    intersect_widths = np.maximum(np.zeros(ulxs.shape, dtype="f4"), lrxs - ulxs).flatten()
    np_boxes_output = np.swapaxes(
        [ulxs.flatten(), ulys.flatten(), intersect_widths.flatten(), intersect_heights.flatten()], 1, 0
    )
    boxes_output = []
    for idx in range(np_boxes_output.shape[0]):
        try:
            boxes_output.append(
                BoundingBox(
                    ulx=np_boxes_output[idx][0],
                    uly=np_boxes_output[idx][1],
                    width=np_boxes_output[idx][2],
                    height=np_boxes_output[idx][3],
                    absolute_coords=absolute_coords,
                )
            )
        except BoundingBoxError:
            log_dict = {
                "ulx": np_boxes_output[idx][0],
                "uly": np_boxes_output[idx][1],
                "width": np_boxes_output[idx][2],
                "height": np_boxes_output[idx][3],
            }

            logger.warning(LoggingRecord("intersection_boxes", log_dict))  # type: ignore

    return boxes_output
