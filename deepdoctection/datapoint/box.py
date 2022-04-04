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
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pycocotools.mask as coco_mask
from numpy import float32

from ..utils.detection_types import ImageType


# Much faster than utils/np_box_ops
def np_iou(box_a: npt.NDArray[float32], box_b: npt.NDArray[float32]) -> npt.NDArray[float32]:
    """
    Calculate iou for two arrays of bounding boxes

    :param box_a: Array of shape Nx4
    :param box_b: Array of shape Mx4

    :return: Array of shape NxM
    """

    def to_xywh(box: npt.NDArray[float32]) -> npt.NDArray[float32]:
        box = box.copy()
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]
        return box

    ret = coco_mask.iou(to_xywh(box_a), to_xywh(box_b), np.zeros((len(box_b),), dtype=np.bool))  # type: ignore
    # can accelerate even more, if using float32
    return ret.astype("float32")


@dataclass
class BoundingBox:
    """
    Rectangular bounding box dataclass for object detection. Store coordinates and allows several
    representations. You can define an instance by passing the upper left point along with either height
    and width or along with the lower right point. Pass absolute_coords = 'True' if you work with image
    pixel coordinates. If you work with coordinates in the range between (0,1) then pass absolute_coords
    ='False'. A bounding box is a disposable object. Do not change the coordinates once the have been set but define
    a new box.

    :attr:`absolute_coords` indicates, whether given coordinates are in absolute or in relative terms

    :attr:`ulx`: upper left x

    :attr:`uly`: upper left y

    :attr:`lrx`: lower right x

    :attr:`lry`: lower right y

    :attr:`height`: height

    :attr:`width`: width
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
            assert self.lrx is not None, "Bounding box not fully initialized"
            self.width = self.lrx - self.ulx
        if self.height == 0.0:
            assert self.lry is not None, "Bounding box not fully initialized"
            self.height = self.lry - self.uly

        if self.lrx == 0.0:
            assert self.width is not None, "Bounding box not fully initialized"
            self.lrx = self.ulx + self.width
        if self.lry == 0.0:
            assert self.height is not None, "Bounding box not fully initialized"
            self.lry = self.uly + self.height

        assert self.ulx >= 0.0 and self.uly >= 0.0, "bounding box ul must be >= (0.,0.)"
        assert self.height > 0.0 and self.width > 0.0, (
            f"bounding box must have height and width >0. Check coords "
            f"ulx: {self.ulx}, uly: {self.uly}, lrx: {self.lrx}, "
            f"lry: {self.lry}."
        )

        if not self.absolute_coords:
            assert (
                self.ulx <= 1 and self.uly <= 1 and self.lrx <= 1 and self.lry <= 1
            ), "coordinates must be between 0 and 1"

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
    def center(self) -> List[float]:
        """
        Bounding box center [x,y]
        """
        return [self.cx, self.cy]

    @property
    def area(self) -> float:
        """
        Bounding box area
        """
        return self.width * self.height

    def to_np_array(self, mode: str, scale_x: float = 1.0, scale_y: float = 1.0) -> npt.NDArray[np.float32]:
        """
        Returns the coordinates as numpy array.

        :param mode: * "xyxy" for upper left/lower right point representation,
                     * "xywh" for upper left and width/height representation or
                     * "poly" for full eight coordinate polygon representation. x,y coordinates will be returned in
                        counter-clockwise order.
        :param scale_x: rescale the x coordinate. Defaults to 1
        :param scale_y: rescale the y coordinate. Defaults to 1
        :return: box coordinates
        """
        np_box_scale = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        np_poly_scale = np.array(
            [scale_x, scale_y, scale_x, scale_y, scale_x, scale_y, scale_x, scale_y], dtype=np.float32
        )
        assert mode in ("xyxy", "xywh", "poly"), "not a valid mode"
        if mode == "xyxy":
            return np.array([self.ulx, self.uly, self.lrx, self.lry], dtype=np.float32) * np_box_scale
        if mode == "xywh":
            return np.array([self.ulx, self.uly, self.width, self.height], dtype=np.float32) * np_box_scale
        return (
            np.array([self.ulx, self.uly, self.lrx, self.uly, self.lrx, self.lry, self.ulx, self.lry], dtype=np.float32)
            * np_poly_scale
        )

    def to_list(self, mode: str, scale_x: float = 1, scale_y: float = 1.0) -> List[float]:
        """
        Returns the coordinates as list

        :param mode: * "xyxy" for upper left/lower right point representation,
                     * "xywh" for upper left and width/height representation or
                     * "poly" for full four-point polygon representation. Points will be returned in counter-clockwise
                        order
        :param scale_x: rescale the x coordinate. Defaults to 1
        :param scale_y: rescale the y coordinate. Defaults to 1
        :return: box coordinates
        """
        assert mode in ("xyxy", "xywh", "poly"), "not a valid mode"
        if mode == "xyxy":
            return [
                float(self.ulx) * scale_x,
                float(self.uly) * scale_y,
                float(self.lrx) * scale_x,
                float(self.lry) * scale_y,
            ]
        if mode == "xywh":
            return [
                float(self.ulx) * scale_x,
                float(self.uly) * scale_y,
                float(self.width) * scale_x,
                float(self.height) * scale_y,
            ]
        return [
            float(self.ulx) * scale_x,
            float(self.uly) * scale_y,
            float(self.lrx) * scale_x,
            float(self.uly) * scale_y,
            float(self.lrx) * scale_x,
            float(self.lry) * scale_y,
            float(self.ulx) * scale_x,
            float(self.lry) * scale_y,
        ]

    def transform(
        self,
        image_width: float,
        image_height: float,
        absolute_coords: bool = False,
        output: str = "list",
        mode: str = "xyxy",
    ) -> Union[npt.NDArray[np.float32], List[float], "BoundingBox"]:
        """
        Transforms bounding box coordinates into absolute or relative coords. Internally, a new bounding box will be
        created. Changing coordinates requires width and height of the whole image.

        :param image_width: The horizontal image size
        :param image_height: The vertical image size
        :param absolute_coords: Whether to recalculate into absolute coordinates.
        :param output: If true will return the bounding box as list, otherwise as numpy array
        :param mode: "xyxy", "xywh" or "poly" mode as described in :meth:`BoundingBox.as_list`
                     or :meth:`BoundingBox.as_np_array`.

        :return: Either a list or np.array.
        """
        assert output in ["list", "np", "box"]

        if absolute_coords != self.absolute_coords:  # only transforming in this case
            if self.absolute_coords:
                transformed_box = BoundingBox(
                    absolute_coords=not self.absolute_coords,
                    ulx=self.ulx / image_width,
                    uly=self.uly / image_height,
                    lrx=self.lrx / image_width,
                    lry=self.lry / image_height,
                )
            else:
                transformed_box = BoundingBox(
                    absolute_coords=not self.absolute_coords,
                    ulx=self.ulx * image_width,
                    uly=self.uly * image_height,
                    lrx=self.lrx * image_width,
                    lry=self.lry * image_height,
                )

            if output == "list":
                return transformed_box.to_list(mode)
            if output == "np":
                return transformed_box.to_np_array(mode)
            return transformed_box
        if output == "list":
            return self.to_list(mode)
        if output == "np":
            return self.to_np_array(mode)
        return self

    def __str__(self) -> str:
        return f"Bounding Box ulx: {self.ulx} uly: {self.uly} lrx: {self.lrx} lry: {self.lry}"

    @staticmethod
    def remove_keys() -> List[str]:
        """
        A list of attributes to suspend from as_dict creation.
        """
        return ["height", "width"]


def intersection_box(
    box_1: BoundingBox, box_2: BoundingBox, width: Optional[float] = None, height: Optional[float] = None
) -> BoundingBox:
    """
    Returns the intersection bounding box of two boxes. Will raise a ValueError if the intersection is empty.
    If coords are absolute, it will floor the lower and ceil the upper coord to ensure the resulting box has same
    coordinates as the box induces from :func:`crop_box_from_image`

    :param box_1: bounding box
    :param box_2: bounding box
    :param width: Total width of image. This optional parameter is needed if the value of absolute_coords of box_1 and
                  box_2 are not equal.

    :param height:Total height of image. This optional parameter is needed if the value of absolute_coords of box_1 and
                  box_2 are not equal.

    :return: bounding box. Will have same absolute_coords as box_2, if absolute_coords of box_1 and box_2 are note same
    """
    if width is None and height is None:
        assert box_1.absolute_coords == box_2.absolute_coords, (
            "when absolute coords of boxes are not equal must " "pass width and height"
        )
    if box_1.absolute_coords != box_2.absolute_coords:
        # will transform box_1
        box_1 = box_1.transform(width, height, box_2.absolute_coords, output="box")  # type: ignore
    ulx = max(box_1.ulx, box_2.ulx)
    uly = max(box_1.uly, box_2.uly)
    lrx = min(box_1.lrx, box_2.lrx)
    lry = min(box_1.lry, box_2.lry)
    if box_2.absolute_coords:
        ulx, uly, lrx, lry = np.floor(ulx), np.floor(uly), np.ceil(lrx), np.ceil(lry)
    return BoundingBox(box_2.absolute_coords, ulx=ulx, uly=uly, lrx=lrx, lry=lry)


def crop_box_from_image(
    np_image: ImageType, crop_box: BoundingBox, width: Optional[float] = None, height: Optional[float] = None
) -> ImageType:
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
    if width is None and height is None:
        assert crop_box.absolute_coords, (
            "when crop_box has absolute coords set to False, then width and height are " "positional args"
        )
    absolute_coord_box = (
        crop_box
        if crop_box.absolute_coords
        else crop_box.transform(width, height, absolute_coords=True, output="box")  # type: ignore
    )
    assert isinstance(absolute_coord_box, BoundingBox)
    np_max_y, np_max_x = np_image.shape[0:2]
    return np_image[  # type: ignore
        np.int32(np.floor(absolute_coord_box.uly)) : min(  # type: ignore
            np.int32(np.ceil(absolute_coord_box.lry)), np_max_y
        ),
        np.int32(np.floor(absolute_coord_box.ulx)) : min(  # type: ignore
            np.int32(np.ceil(absolute_coord_box.lrx)), np_max_x
        ),
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

    assert local_box.absolute_coords and embedding_box.absolute_coords, "absolute coords required"
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

    assert global_box.absolute_coords and embedding_box.absolute_coords, "absolute coords required"
    assert embedding_box.ulx is not None and embedding_box.uly is not None
    assert embedding_box.width is not None and embedding_box.height is not None
    assert (
        global_box.ulx is not None
        and global_box.uly is not None
        and global_box.lrx is not None
        and global_box.lry is not None
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
    Generating the smallest box containing an arbitrary tuple/list of boxes. This function is only implemented for boxes
    with absolute coords = "True".

    :param boxes: An arbitrary tuple/list of bounding boxes :class:`BoundingBox` all having absolute_coords="True".
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
    Generating a bounding box with scaled coordinates. Will rescale x coordinate with *
    (current_total_width/scaled_total_width), resp. y coordinate with * (current_total_height/scaled_total_height),
    while not changing anything if absolute_coords is set to False.

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
