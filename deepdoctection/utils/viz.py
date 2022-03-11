# -*- coding: utf-8 -*-
# File: viz.py

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
Some visualisation utils. Copied and pasted from

https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/viz.py

and

https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/colormap.py
"""
import sys
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from numpy import float32

from .detection_types import ImageType

__all__ = ["draw_text", "draw_boxes", "interactive_imshow"]

_COLORS = (
    np.array(
        [
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def random_color(rgb: bool = True, maximum: int = 255) -> Tuple[int, int, int]:
    """
    :param rgb: Whether to return RGB colors or BGR colors.
    :param maximum: either 255 or 1
    :return:
    """

    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return tuple(int(x) for x in ret)  # type: ignore


def draw_text(
    np_image: ImageType, pos: Tuple[int, int], text: str, color: Tuple[int, int, int], font_scale: float = 0.4
) -> ImageType:
    """
    Draw text on an image.

    :param np_image: image as np.ndarray
    :param pos: x, y; the position of the text
    :param text: text string to draw
    :param color: a 3-tuple BGR color in [0, 255]
    :param font_scale: float
    :return: numpy array
    """

    np_image = np_image.astype(np.uint8)
    x_0, y_0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x_0 + text_w > np_image.shape[1]:
        x_0 = np_image.shape[1] - text_w
    if y_0 - int(1.15 * text_h) < 0:
        y_0 = int(1.15 * text_h)
    back_top_left = x_0, y_0 - int(1.3 * text_h)
    back_bottom_right = x_0 + text_w, y_0
    cv2.rectangle(np_image, back_top_left, back_bottom_right, color, -1)
    # Show text.
    text_bottomleft = x_0, y_0 - int(0.25 * text_h)
    cv2.putText(np_image, text, text_bottomleft, font, font_scale, (222, 222, 222), thickness=2, lineType=cv2.LINE_AA)
    return np_image


def draw_boxes(
    np_image: ImageType,
    boxes: npt.NDArray[float32],
    category_names_list: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> ImageType:
    """
    Dray bounding boxes with category names into image.

    :param np_image: Image as np.ndarray
    :param boxes: A numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
    :param category_names_list: List of N category names.
    :param color: A 3-tuple BGR color (in range [0, 255])
    :return: A new image np.ndarray
    """

    boxes = np.asarray(boxes, dtype="int32")
    if category_names_list is not None:
        assert len(category_names_list) == len(boxes), f"{len(category_names_list)} != {len(boxes)}"
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)  # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert (
        boxes[:, 0].min() >= 0
        and boxes[:, 1].min() >= 0
        and boxes[:, 2].max() <= np_image.shape[1]
        and boxes[:, 3].max() <= np_image.shape[0]
    ), f"Image shape: {str(np_image.shape)}\n Boxes:\n{str(boxes)}"

    np_image = np_image.copy()

    if np_image.ndim == 2 or (np_image.ndim == 3 and np_image.shape[2] == 1):
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        box = boxes[i, :]
        if category_names_list is not None:
            choose_color = random_color() if color is None else color
            np_image = draw_text(np_image, (box[0], box[1]), category_names_list[i], color=choose_color, font_scale=1.0)
            cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), color=choose_color, thickness=4)
    return np_image


def interactive_imshow(
    img: ImageType,
    lclick_cb: Optional[Callable[[npt.NDArray[float32], int, int], None]] = None,
    rclick_cb: Optional[Callable[[npt.NDArray[float32], int, int], None]] = None,
    **kwargs: str,
) -> None:
    """
    Display an image in a pop-up window

    :param img: An image (expect BGR) to show.
    :param lclick_cb: a callback ``func(img, x, y)`` for left/right click event.
    :param rclick_cb: a callback ``func(img, x, y)`` for left/right click event.
    :param kwargs: can be {key_cb_a: callback_img, key_cb_b: callback_img}, to specify a callback ``func(img)`` for
                   keypress. Some existing keypress event handler:

                          * q: destroy the current window
                          * x: execute ``sys.exit()``
                          * s: save image to "out.png"
    """
    name = "q, x: quit / s: save"
    cv2.imshow(name, img)

    def mouse_cb(event, x, y, *args):  # type: ignore
        if event == cv2.EVENT_LBUTTONUP and lclick_cb is not None:
            lclick_cb(img, x, y)
        elif event == cv2.EVENT_RBUTTONUP and rclick_cb is not None:
            rclick_cb(img, x, y)

    cv2.setMouseCallback(name, mouse_cb)
    key = cv2.waitKey(-1)
    while key >= 128:
        key = cv2.waitKey(-1)
    key = chr(key & 0xFF)
    cb_name = "key_cb_" + key
    if cb_name in kwargs:
        kwargs[cb_name](img)  # type: ignore
    elif key == "q":
        cv2.destroyWindow(name)
    elif key == "x":
        sys.exit()
    elif key == "s":
        cv2.imwrite("out.png", img)
    elif key in ["+", "="]:
        img = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        interactive_imshow(img, lclick_cb, rclick_cb, **kwargs)
    elif key == "-":
        img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
        interactive_imshow(img, lclick_cb, rclick_cb, **kwargs)
