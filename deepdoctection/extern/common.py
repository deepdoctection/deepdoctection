# -*- coding: utf-8 -*-
# File: common.py

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
Some classes used throughout the various external libraries.
"""

import cv2
from dataflow.dataflow.imgaug import ResizeTransform  # type: ignore

from ..utils.detection_types import ImageType


class InferenceResize:  # pylint: disable=R0903
    """
    Try resizing the shortest edge to a certain number while avoiding the longest edge to exceed max_size. This is
    the inference version of :class:`extern.tp.frcnn.common.CustomResize` .
    """

    def __init__(self, short_edge_length: int, max_size: int, interp: int = cv2.INTER_LINEAR) -> None:
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

        scale = self.short_edge_length * 1.0 / min(h, w)
        if h < w:
            new_h, new_w = self.short_edge_length, scale * w
        else:
            new_h, new_w = scale * h, self.short_edge_length  # type: ignore
        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h = new_h * scale  # type: ignore
            new_w = new_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return ResizeTransform(h, w, new_h, new_w, self.interp)
