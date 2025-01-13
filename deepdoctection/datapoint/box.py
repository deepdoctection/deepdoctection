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
Implementation of BoundingBox class and related methods
"""

from dataclasses import dataclass
from math import ceil, floor
from typing import Optional, Sequence, no_type_check

import numpy as np
import numpy.typing as npt
from lazy_imports import try_import
from numpy import float32

from ..utils.error import BoundingBoxError
from ..utils.file_utils import cocotools_available
from ..utils.logger import LoggingRecord, logger
from ..utils.types import PixelValues

with try_import() as import_guard:
    import pycocotools.mask as coco_mask


# taken from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/common.py


def coco_iou(box_a: npt.NDArray[float32], box_b: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Calculate iou for two arrays of bounding boxes in xyxy format

    :param box_a: Array of shape Nx4
    :param box_b: Array of shape Mx4

    :return: Array of shape NxM
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

    :param boxes: numpy array with shape [N, 4] holding N boxes in xyxy format

    :return: a numpy array with shape [N*1] representing box areas
    """
    return np.array((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=float32)


# taken from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/np_box_ops.py


def intersection(boxes1: npt.NDArray[float32], boxes2: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Compute pairwise intersection areas between boxes.

    :param boxes1: a numpy array with shape [N, 4] holding N boxes in xyxy format
    :param  boxes2: a numpy array with shape [M, 4] holding M boxes in xyxy format

    :return: a numpy array with shape [N*M] representing pairwise intersection area
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

    :param  boxes1: a numpy array with shape [N, 4] holding N boxes in xyxy format.
    :param  boxes2: a numpy array with shape [M, 4] holding M boxes in xyxy format.

    :return:  a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect

    return intersect / union


def iou(boxes1: npt.NDArray[float32], boxes2: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Computes pairwise intersection-over-union between box collections. The method will be chosen based on what
    is installed:

    - If pycocotools is installed it will choose pycocotools.mask.iou which is a cython implementation and much faster
    - Otherwise it will use the numpy implementation as fallback

    :param boxes1:  a numpy array with shape [N, 4] holding N boxes in xyxy format.
    :param boxes2:  a numpy array with shape [N, 4] holding N boxes in xyxy format.

    :return: a numpy array with shape [N, M] representing pairwise iou scores.
    """

    if cocotools_available():
        return coco_iou(boxes1, boxes2)
    return np_iou(boxes1, boxes2)


@dataclass
class BoundingBox:
    """
    Rectangular bounding box dataclass for object detection. Store coordinates and allows several
    representations. You can define an instance by passing the upper left point along with either height
    and width or along with the lower right point. Pass absolute_coords = 'True' if you work with image
    pixel coordinates. If you work with coordinates in the range between (0,1) then pass absolute_coords
    ='False'. A bounding box is a disposable object. Do not change the coordinates once the have been set but define
    a new box.

    `absolute_coords` indicates, whether given coordinates are in absolute or in relative terms

    `ulx`: upper left x

    `uly`: upper left y

    `lrx`: lower right x

    `lry`: lower right y

    `height`: height

    `width`: width
    """

    absolute_coords: bool
    ulx: float
    uly: float
    lrx: float = 0.0
    lry: float = 0.0
    height: float = 0.0
    width: float = 0.0

    def __post_init__(self) -> None:
        if self.width == 0.0:
            if self.lrx is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self.width = self.lrx - self.ulx
        if self.height == 0.0:
            if self.lry is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self.height = self.lry - self.uly

        if self.lrx == 0.0:
            if self.width is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self.lrx = self.ulx + self.width
        if self.lry == 0.0:
            if self.height is None:
                raise BoundingBoxError("Bounding box not fully initialized")
            self.lry = self.uly + self.height

        if not (self.ulx >= 0.0 and self.uly >= 0.0):
            raise BoundingBoxError("Bounding box ul must be >= (0.,0.)")
        if not (self.height > 0.0 and self.width > 0.0):
            raise BoundingBoxError(
                f"bounding box must have height and width >0. Check coords "
                f"ulx: {self.ulx}, uly: {self.uly}, lrx: {self.lrx}, "
                f"lry: {self.lry}."
            )
        if not self.absolute_coords:
            if not (self.ulx <= 1.0 and self.uly <= 1.0 and self.lrx <= 1.0 and self.lry <= 1.0):
                raise BoundingBoxError("coordinates must be between 0 and 1")

    @property
    def cx(self) -> float:
        """
        Bounding box center x coordinate
        """
        return self.ulx + 0.5 * self.width

    @property
    def cy(self) -> float:
        """
        Bounding box center y coordinate
        """
        return self.uly + 0.5 * self.height

    @property
    def center(self) -> list[float]:
        """
        Bounding box center [x,y]
        """
        return [self.cx, self.cy]

    @property
    def area(self) -> float:
        """
        Bounding box area
        """
        if self.absolute_coords:
            return self.width * self.height
        raise ValueError("Cannot calculate area, when bounding box coords are relative")

    def to_np_array(self, mode: str, scale_x: float = 1.0, scale_y: float = 1.0) -> npt.NDArray[np.float32]:
        """
        Returns the coordinates as numpy array.

        :param mode: Mode for coordinate arrangement:
                     "xyxy" for upper left/lower right point representation,
                     "xywh" for upper left and width/height representation or
                     "poly" for full eight coordinate polygon representation. x,y coordinates will be
                      returned in counter-clockwise order.

        :param scale_x: rescale the x coordinate. Defaults to 1
        :param scale_y: rescale the y coordinate. Defaults to 1
        :return: box coordinates
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

    def to_list(self, mode: str, scale_x: float = 1.0, scale_y: float = 1.0) -> list[float]:
        """
        Returns the coordinates as list

        :param mode:  Mode for coordinate arrangement:
                     "xyxy" for upper left/lower right point representation,
                     "xywh" for upper left and width/height representation or
                     "poly" for full eight coordinate polygon representation. x,y coordinates will be
                      returned in counter-clockwise order.

        :param scale_x: rescale the x coordinate. Defaults to 1
        :param scale_y: rescale the y coordinate. Defaults to 1
        :return: box coordinates
        """
        assert mode in ("xyxy", "xywh", "poly"), "Not a valid mode"
        if mode == "xyxy":
            return [
                self.ulx * scale_x,
                self.uly * scale_y,
                self.lrx * scale_x,
                self.lry * scale_y,
            ]
        if mode == "xywh":
            return [
                self.ulx * scale_x,
                self.uly * scale_y,
                self.width * scale_x,
                self.height * scale_y,
            ]
        return [
            self.ulx * scale_x,
            self.uly * scale_y,
            self.lrx * scale_x,
            self.uly * scale_y,
            self.lrx * scale_x,
            self.lry * scale_y,
            self.ulx * scale_x,
            self.lry * scale_y,
        ]

    def transform(
        self,
        image_width: float,
        image_height: float,
        absolute_coords: bool = False,
    ) -> "BoundingBox":
        """
        Transforms bounding box coordinates into absolute or relative coords. Internally, a new bounding box will be
        created. Changing coordinates requires width and height of the whole image.

        :param image_width: The horizontal image size
        :param image_height: The vertical image size
        :param absolute_coords: Whether to recalculate into absolute coordinates.

        :return: Either a list or np.array.
        """

        if absolute_coords != self.absolute_coords:  # only transforming in this case
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
        return f"Bounding Box ulx: {self.ulx}, uly: {self.uly}, lrx: {self.lrx}, lry: {self.lry}"

    @staticmethod
    def remove_keys() -> list[str]:
        """
        A list of attributes to suspend from as_dict creation.
        """
        return ["height", "width"]

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> "BoundingBox":
        """
        Create `BoundingBox` instance from dict

        :param kwargs: dict with  `BoundingBox` attributes
        :return: Initialized BoundingBox
        """
        return cls(**kwargs)


def intersection_box(
    box_1: BoundingBox, box_2: BoundingBox, width: Optional[float] = None, height: Optional[float] = None
) -> BoundingBox:
    """
    Returns the intersection bounding box of two boxes. Will raise a `ValueError` if the intersection is empty.
    If coords are absolute, it will floor the lower and ceil the upper coord to ensure the resulting box has same
    coordinates as the box induces from `crop_box_from_image`

    :param box_1: bounding box
    :param box_2: bounding box
    :param width: Total width of image. This optional parameter is needed if the value of `absolute_coords` of `box_1`
                  and `box_2` are not equal.
    :param height: Total height of image. This optional parameter is needed if the value of `absolute_coords` of `box_1`
                   and `box_2` are not equal.

    :return: bounding box. Will have same `absolute_coords` as `box_2`, if absolute_coords of `box_1` and `box_2` are
             not equal
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
    Crop a box (the crop_box) from a np_image. Will floor the left  and ceil the right coordinate point.

    :param np_image: Image to crop from.
    :param crop_box: Bounding box to crop.
    :param width: Total width of image. This optional parameter is needed if the value of absolute_coords of crop_box is
                  False

    :param height:Total width of image. This optional parameter is needed if the value of absolute_coords of crop_box is
                  False

    :return: A numpy array cropped according to the bounding box.
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
    embedding box. The global coords will be determined by transforming the upper left point (which is (0,0) in
    local terms) into the upper left point given by the embedding box. This will shift the ul point of the
    local box to ul + embedding_box.ul

    :param local_box: bounding box with coords in terms of an embedding (e.g. local coordinates)
    :param embedding_box: bounding box of the embedding.
    :return: bounding box with local box transformed to absolute coords
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
    embedding box into (0,0).

    :param global_box: The bounding box to be embedded
    :param embedding_box: The embedding box. Must cover the global box completely.
    :return: Bounding box of the embedded box in local coordinates.
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
    :param boxes: An arbitrary tuple/list of bounding boxes `BoundingBox`.
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
    Generating a bounding box with scaled coordinates. Will rescale x coordinate with factor

    * (current_total_width/scaled_total_width),

    resp. y coordinate with factor

    * (current_total_height/scaled_total_height),

    while not changing anything if `absolute_coords` is set to False.

    :param box: BoudingBox to rescale
    :param current_total_width: absolute coords of width of image
    :param current_total_height: absolute coords of height of image
    :param scaled_total_width:  absolute width of rescaled image
    :param scaled_total_height: absolute height of rescaled image
    :return: rescaled BoundingBox
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
    The multiple version of 'intersection_box': Given two lists of m and n bounding boxes, it will calculate the
    pairwise intersection of both groups. There will be at most mxn intersection boxes.

    :param boxes_1: sequence of m BoundingBox
    :param boxes_2: sequence of n BoundingBox
    :return: list of at most mxn BoundingBox
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
