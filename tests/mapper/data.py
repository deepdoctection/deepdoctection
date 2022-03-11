# -*- coding: utf-8 -*-
# File: _fixtures.py

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
Some datapoint samples in a separate module
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from deepdoctection.datapoint import BoundingBox, Image, ImageAnnotation, convert_np_array_to_b64
from deepdoctection.datasets.info import DatasetCategories
from deepdoctection.extern.base import TokenClassResult
from deepdoctection.utils.detection_types import ImageType, JsonDict
from deepdoctection.utils.settings import names

_SAMPLE_COCO = {
    "file_name": "/test/path/PMC5447509_00002.jpg",
    "height": 794,
    "id": 346767,
    "width": 596,
    "annotations": [
        {
            "segmentation": [
                [
                    37.59,
                    360.34,
                    288.66,
                    360.34,
                    288.66,
                    370.34,
                    288.63,
                    370.34,
                    288.63,
                    381.37,
                    288.63,
                    381.37,
                    288.63,
                    391.26,
                    272.86,
                    391.26,
                    272.86,
                    401.7,
                    37.59,
                    401.7,
                    37.59,
                    391.8,
                    37.59,
                    381.37,
                    37.59,
                    370.89,
                    37.59,
                    360.34,
                ]
            ],
            "area": 10218.471181684348,
            "iscrowd": 0,
            "image_id": 346767,
            "bbox": [37.59, 360.34, 251.07, 41.36],
            "category_id": 1,
            "id": 3377124,
        },
        {
            "segmentation": [
                [
                    50.06,
                    433.64,
                    288.65,
                    433.64,
                    288.65,
                    443.53,
                    285.36,
                    443.53,
                    285.36,
                    454.02,
                    37.59,
                    454.02,
                    37.59,
                    444.13,
                    50.06,
                    444.13,
                    50.06,
                    433.64,
                ]
            ],
            "area": 4951.1586551328655,
            "iscrowd": 0,
            "image_id": 346767,
            "bbox": [37.59, 433.64, 251.07, 20.38],
            "category_id": 1,
            "id": 3377125,
        },
    ],
}

_SAMPLE_PUBTABNET = {
    "imgid": 16,
    "html": {
        "cells": [
            {"tokens": ["<b>", "S", "u", "b", "n", "e", "t", "w", "o", "r", "k", "</b>"], "bbox": [11, 5, 60, 14]},
            {
                "tokens": [
                    "<b>",
                    "D",
                    "i",
                    "m",
                    "e",
                    "r",
                    " ",
                    "f",
                    "o",
                    "r",
                    "m",
                    "a",
                    "t",
                    "i",
                    "o",
                    "n",
                    "</b>",
                ],
                "bbox": [82, 5, 150, 14],
            },
            {
                "tokens": [
                    "<b>",
                    "M",
                    "o",
                    "n",
                    "o",
                    "m",
                    "e",
                    "r",
                    " ",
                    "b",
                    "i",
                    "n",
                    "d",
                    "i",
                    "n",
                    "g",
                    "</b>",
                ],
                "bbox": [175, 5, 245, 14],
            },
            {
                "tokens": ["<b>", "D", "i", "m", "e", "r", " ", "b", "i", "n", "d", "i", "n", "g", "</b>"],
                "bbox": [269, 5, 326, 14],
            },
            {
                "tokens": [
                    "<b>",
                    "M",
                    "u",
                    "l",
                    "t",
                    "i",
                    "p",
                    "l",
                    "e",
                    " ",
                    "e",
                    "q",
                    "u",
                    "i",
                    "l",
                    "i",
                    "b",
                    "r",
                    "i",
                    "a",
                    " ",
                    "r",
                    "u",
                    "l",
                    "e",
                    "d",
                    " ",
                    "o",
                    "u",
                    "t",
                    "?",
                    "</b>",
                ],
                "bbox": [362, 5, 475, 14],
            },
            {"tokens": []},
            {"tokens": []},
            {"tokens": []},
            {"tokens": []},
            {"tokens": ["<b>", "I", "G", "-", "T", "</b>"], "bbox": [343, 27, 363, 36]},
            {"tokens": ["<b>", "I", "G", "-", "K", "</b>"], "bbox": [376, 27, 396, 36]},
            {"tokens": ["<b>", "S", "R", "G", "</b>"], "bbox": [408, 27, 428, 36]},
            {"tokens": ["<b>", "I", "N", "J", "</b>"], "bbox": [439, 27, 454, 36]},
            {"tokens": ["<b>", "C", "R", "N", "T", "</b>"], "bbox": [467, 27, 493, 36]},
            {"tokens": ["<i>", "a", "b", "</i>"], "bbox": [31, 49, 41, 59]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [111, 49, 121, 59]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [205, 49, 215, 59]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [293, 49, 302, 59]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [348, 49, 359, 59]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [380, 49, 392, 59]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [412, 49, 424, 59]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [441, 49, 452, 59]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 49, 486, 59]},
            {"tokens": ["<i>", "a", "b", "c", "d", "</i>"], "bbox": [27, 59, 44, 69]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [111, 59, 121, 69]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [203, 59, 218, 69]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [293, 59, 302, 69]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 59, 358, 69]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 59, 391, 69]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 59, 423, 69]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [442, 59, 451, 69]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 59, 486, 69]},
            {"tokens": ["<i>", "a", "b", "c", "</i>"], "bbox": [29, 70, 42, 79]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [111, 70, 121, 79]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [203, 70, 217, 79]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [293, 70, 302, 79]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 70, 358, 79]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 70, 391, 79]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 70, 423, 79]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [441, 70, 452, 79]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 70, 486, 79]},
            {"tokens": ["<i>", "a", "b", "e", "</i>"], "bbox": [29, 80, 42, 89]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 80, 122, 89]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [205, 80, 215, 89]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [293, 80, 302, 89]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 80, 358, 89]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 80, 391, 89]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [412, 80, 424, 89]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [441, 80, 452, 89]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 80, 486, 89]},
            {"tokens": ["<i>", "a", "b", "e", "f", "g", "</i>"], "bbox": [26, 90, 45, 100]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 90, 122, 100]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [205, 90, 215, 100]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [290, 90, 305, 100]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 90, 358, 100]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 90, 391, 100]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 90, 423, 100]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [442, 90, 451, 100]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [475, 90, 485, 100]},
            {"tokens": ["<i>", "a", "b", "e", "f", "</i>"], "bbox": [28, 100, 44, 110]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 100, 122, 110]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [205, 100, 215, 110]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [291, 100, 304, 110]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 100, 358, 110]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 100, 391, 110]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 100, 423, 110]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [441, 100, 452, 110]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 100, 486, 110]},
            {"tokens": ["<i>", "a", "b", "c", "d", "e", "</i>"], "bbox": [26, 111, 46, 120]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 111, 122, 120]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [203, 111, 218, 120]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [293, 111, 302, 120]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 111, 358, 120]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 111, 391, 120]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 111, 423, 120]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [442, 111, 451, 120]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 111, 486, 120]},
            {"tokens": ["<i>", "a", "b", "c", "d", "e", "f", "g", "</i>"], "bbox": [23, 121, 49, 130]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 121, 122, 130]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [203, 121, 218, 130]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [290, 121, 305, 130]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 121, 358, 130]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 121, 391, 130]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 121, 423, 130]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [442, 121, 451, 130]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [475, 121, 485, 130]},
            {"tokens": ["<i>", "a", "b", "c", "d", "e", "f", "</i>"], "bbox": [24, 131, 47, 141]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 131, 122, 141]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [203, 131, 218, 141]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [291, 131, 304, 141]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 131, 358, 141]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 131, 391, 141]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 131, 423, 141]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [442, 131, 451, 141]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 131, 486, 141]},
            {"tokens": ["<i>", "a", "b", "c", "e", "</i>"], "bbox": [27, 141, 44, 151]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 141, 122, 151]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [203, 141, 217, 151]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [293, 141, 302, 151]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 141, 358, 151]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 141, 391, 151]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 141, 423, 151]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [441, 141, 452, 151]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 141, 486, 151]},
            {"tokens": ["<i>", "a", "b", "c", "e", "f", "</i>"], "bbox": [26, 152, 45, 161]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 152, 122, 161]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [203, 152, 217, 161]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [291, 152, 304, 161]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 152, 358, 161]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 152, 391, 161]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 152, 423, 161]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [441, 152, 452, 161]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [474, 152, 486, 161]},
            {"tokens": ["<i>", "a", "b", "c", "e", "f", "g", "</i>"], "bbox": [25, 162, 47, 171]},
            {"tokens": ["<i>", "y", "e", "s", "</i>"], "bbox": [110, 162, 122, 171]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "-", "</sup>"], "bbox": [203, 162, 217, 171]},
            {"tokens": ["<i>", "y", "e", "s", "</i>", "<sup>", "+", "</sup>"], "bbox": [290, 162, 305, 171]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [349, 162, 358, 171]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [381, 162, 391, 171]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [413, 162, 423, 171]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [442, 162, 451, 171]},
            {"tokens": ["<i>", "n", "o", "</i>"], "bbox": [475, 162, 485, 171]},
        ],
        "structure": {
            "tokens": [
                "<thead>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td",
                ' colspan="5"',
                ">",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "</thead>",
                "<tbody>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "</tbody>",
            ]
        },
    },
    "split": "train",
    "filename": "PMC2759935_007_01.png",
}

_SAMPLE_PRODIGY = {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAiCAIAAAA24aWuAAAAWElEQVQ4EZXBAQEAAAABIP6P"
    "zgZV5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5FTkVORU5DTzFG"
    "W9r8aRmwAAAABJRU5ErkJggg==",
    "text": "99999984_310518_J_1_150819_page_252.png",
    "spans": [
        {
            "label": "TABLE",
            "x": 100,
            "y": 223.7,
            "w": 1442,
            "h": 1875.3,
            "type": "rect",
            "points": [[1, 2.7], [1, 29], [15, 29], [15, 2.7]],
        },
        {
            "label": "TITLE",
            "x": 1181.6,
            "y": 99.2,
            "w": 364.6,
            "h": 68.8,
            "type": "rect",
            "points": [[11.6, 9.2], [11.6, 18], [16.2, 18], [16.2, 9.2]],
        },
    ],
    "_input_hash": -2030256107,
    "_task_hash": 1310891334,
    "_view_id": "image_manual",
    "width": 1650,
    "height": 2350,
    "answer": "reject",
}

Box = namedtuple("Box", ["ulx", "uly", "w", "h"])


@dataclass
class DatapointCoco:
    """
    Class for datapoint in coco annotation format
    """

    dp = _SAMPLE_COCO
    white_image: ImageType = np.ones((794, 596, 3), dtype=np.int32) * 255  # type: ignore
    white_image_string = convert_np_array_to_b64(white_image)
    categories = {"1": "text", "2": "title", "3": "table", "4": "figure", "5": "list"}
    first_ann_box = Box(37.59, 360.34, 251.07, 41.36)

    def get_white_image(self, path: str, type_id: str = "np") -> Optional[Union[str, ImageType]]:
        """
        white image
        :return: np.array
        """
        if path == self.dp["file_name"]:
            if type_id == "np":
                return self.white_image
            return self.white_image_string
        return None

    def get_number_anns(self) -> int:
        """
        number of annotations
        """
        return len(self.dp["annotations"])  # type: ignore

    def get_width(self, image_loaded: bool) -> float:
        """
        width
        """
        if image_loaded:
            return self.white_image.shape[1]
        return float(self.dp["width"])  # type: ignore

    def get_height(self, image_loaded: bool) -> float:
        """
        height
        """
        if image_loaded:
            return self.white_image.shape[0]
        return float(self.dp["height"])  # type: ignore

    def get_first_ann_box(self) -> Box:
        """
        box coordinates of first annotation
        """
        return self.first_ann_box

    def get_first_ann_category(self, as_index: bool = True) -> str:
        """
        category_name or category_id
        """
        if as_index:
            return "1"
        return self.categories["1"]


@dataclass
class DatapointPubtabnet:  # pylint: disable=R0904
    """
    Class for datapoint in pubtabnet annotation format
    """

    dp = _SAMPLE_PUBTABNET
    categories = {"1": "CELL", "2": "ITEM"}
    categories_as_names = {v: k for k, v in categories.items()}
    first_ann_box = Box(475, 162, 10, 9)
    white_image: ImageType = np.ones((1334, 996, 3), dtype=np.int32) * 255  # type: ignore
    white_image_string = convert_np_array_to_b64(white_image)

    def get_white_image(self, path: str, type_id: str = "np") -> Union[str, ImageType]:
        """
        white image
        :return: np.array
        """
        assert path is not None
        if type_id == "np":
            return self.white_image
        return self.white_image_string

    def get_width(self) -> float:
        """
        width
        """
        return self.white_image.shape[1]

    def get_height(self) -> float:
        """
        height
        """
        return self.white_image.shape[0]

    def get_number_cell_anns(self) -> int:
        """
        number of annotations which have category name "CELL"
        """
        return len(list(filter(lambda ele: "bbox" in ele, self.dp["html"]["cells"])))  # type: ignore

    def get_first_ann_box(self) -> Box:
        """
        box coordinates of first annotation
        """
        return self.first_ann_box

    def get_first_ann_category(self, as_index: bool = True) -> str:  # pylint: disable=R0201
        """
        category_name or category_id
        """
        if as_index:
            return "1"
        return self.categories["1"]

    def get_first_ann_sub_category_header_name(self) -> str:  # pylint: disable=R0201
        """
        category_name of sub category
        """
        return "BODY"

    def get_last_ann_category_name(self) -> str:
        """
        category_name
        """
        return self.categories["1"]

    def get_last_ann_sub_category_header_name(self) -> str:  # pylint: disable=R0201
        """
        category_name of sub category
        """
        return "HEAD"

    def get_last_ann_sub_category_row_number_id(self) -> str:  # pylint: disable=R0201
        """
        row number
        """
        return "1"

    def get_last_ann_sub_category_col_number_id(self) -> str:  # pylint: disable=R0201
        """
        col number
        """
        return "1"

    def get_last_ann_sub_category_row_span_id(self) -> str:  # pylint: disable=R0201
        """
        row span
        """
        return "1"

    def get_last_ann_sub_category_col_span_id(self) -> str:  # pylint: disable=R0201
        """
        col span
        """
        return "1"

    def get_first_ann_sub_category_row_number_id(self) -> str:  # pylint: disable=R0201
        """
        row number
        """
        return "14"

    def get_first_ann_sub_category_col_number_id(self) -> str:  # pylint: disable=R0201
        """
        col number
        """
        return "9"

    def get_first_ann_sub_category_row_span_id(self) -> str:  # pylint: disable=R0201
        """
        row span
        """
        return "1"

    def get_first_ann_sub_category_col_span_id(self) -> str:  # pylint: disable=R0201
        """
        col span
        """
        return "1"

    def get_summary_ann_sub_category_rows_id(self) -> str:  # pylint: disable=R0201
        """
        number rows
        """
        return "14"

    def get_summary_ann_sub_category_col_id(self) -> str:  # pylint: disable=R0201
        """
        number cols
        """
        return "9"

    def get_summary_ann_sub_category_row_span_id(self) -> str:  # pylint: disable=R0201
        """
        max row span
        """
        return "1"

    def get_summary_ann_sub_category_col_span_id(self) -> str:  # pylint: disable=R0201
        """
        max col span
        """
        return "5"

    def get_number_of_heads(self) -> int:  # pylint: disable=R0201
        """
        number of head cells
        """
        return 10

    def get_number_of_bodies(self) -> int:  # pylint: disable=R0201
        """
        number of body cells
        """
        return 108


@dataclass
class DatapointProdigy:
    """
    Class for datapoint in coco annotation format
    """

    dp = _SAMPLE_PRODIGY
    categories = {"TEXT": "1", "TITLE": "2", "TABLE": "3", "FIGURE": "4", "LIST": "5"}
    first_ann_box = Box(1, 2.7, 14, 26.3)

    def get_width(self, image_loaded: bool) -> float:
        """
        width
        """
        if image_loaded:
            return 17.0
        return float(self.dp["width"])  # type: ignore

    def get_height(self, image_loaded: bool) -> float:
        """
        height
        """
        if image_loaded:
            return 34.0
        return float(self.dp["height"])  # type: ignore

    def get_number_anns(self) -> int:
        """
        number of annotations
        """
        return len(self.dp["spans"])  # type: ignore

    def get_first_ann_box(self) -> Box:
        """
        box coordinates of first annotation
        """
        return self.first_ann_box

    def get_first_ann_category(self, as_index: bool = True) -> str:
        """
        category_name or category_id
        """
        if as_index:
            return self.categories["TABLE"]
        return "TABLE"


class DatapointImage:
    """
    Class for datapoint in standard Image format
    """

    def __init__(self) -> None:
        self.image: Image = Image(file_name="sample.png", location="/to/path")
        _img_np: ImageType = np.ones((34, 96, 3), dtype=np.int32) * 255  # type: ignore
        self.image.image = _img_np
        box = BoundingBox(ulx=2.6, uly=3.7, lrx=4.6, lry=5.7, absolute_coords=True)
        ann = ImageAnnotation(category_name="FOO", bounding_box=box, score=0.53, category_id="1")
        self.image.dump(ann)
        box = BoundingBox(ulx=16.6, uly=26.6, height=4.0, width=14.0, absolute_coords=True)
        ann = ImageAnnotation(category_name="BAK", bounding_box=box, score=0.99, category_id="2")
        self.image.dump(ann)
        self.dict_image: str = "data:image/png;base64," + convert_np_array_to_b64(_img_np)
        self.dict_text: str = "sample.png"
        self.len_spans: int = 2
        self.first_span: JsonDict = {
            "label": "FOO",
            "annotation_id": "2a761d95-14f8-3a03-a5ee-f64c1130cb80",
            "score": 0.53,
            "type": "rect",
            "points": [[2.6, 3.7], [2.6, 5.7], [4.6, 5.7], [4.6, 3.7]],
        }
        self.second_span: JsonDict = {
            "label": "BAK",
            "annotation_id": "e1dad126-ced5-3a1b-a8ae-6af5cf067b1f",
            "score": 0.99,
            "type": "rect",
            "points": [[16.6, 26.6], [16.6, 30.6], [30.6, 30.6], [30.6, 26.6]],
        }
        self.coco_image: JsonDict = {"id": 4217040713909021022429, "width": 96, "height": 34, "file_name": "sample.png"}
        self.coco_anns: List[JsonDict] = [
            {
                "id": 276195148303564113080,
                "image_id": 4217040713909021022429,
                "category_id": 1,
                "iscrowd": 0,
                "area": 4.0,
                "bbox": [2.6, 3.7, 2.0, 2.0],
                "score": 0.53,
            },
            {
                "id": 11265318650671,
                "image_id": 4217040713909021022429,
                "category_id": 2,
                "iscrowd": 0,
                "area": 56.0,
                "bbox": [16.6, 26.6, 4.0, 14.0],
                "score": 0.99,
            },
        ]
        self.categories = DatasetCategories(init_categories=["FOO", "BAK"])
        self.tp_frcnn_training: JsonDict = {
            "image": _img_np,
            "gt_boxes": np.asarray([[2.6, 3.7, 4.6, 5.7], [16.6, 26.6, 30.6, 30.6]]).astype("float32"),
            "gt_labels": np.asarray([1, 2]).astype("float32"),
            "file_name": "/to/path",
        }

    def get_image_str(self) -> str:
        """
        Dictionary in prodigy input structure
        """
        return self.dict_image

    def get_text(self) -> str:
        """
        text value
        """
        return self.dict_text

    def get_len_spans(self) -> int:
        """
        len of span value
        """
        return self.len_spans

    def get_first_span(self) -> JsonDict:
        """
        Dict of first span
        """
        return self.first_span

    def get_second_span(self) -> JsonDict:
        """
        Dict of second span
        """
        return self.second_span

    def get_coco_image(self) -> JsonDict:
        """
        Dict of coco image
        """
        return self.coco_image

    def get_coco_anns(self) -> List[JsonDict]:
        """
        List of coco anns
        """
        return self.coco_anns

    def get_dataset_categories(self) -> DatasetCategories:
        """
        DatasetCategories
        """
        return self.categories

    def get_tp_frcnn_training_anns(self) -> JsonDict:
        """
        Dict of tp frcnn training anns
        """
        return self.tp_frcnn_training


class DatapointPageDict:  # pylint: disable=R0903
    """
    Page object as dict
    """

    page_dict = {
        "uuid": "1143e118-ecbd-3b92-a6fa-b753d3f96753",
        "file_name": "sample.png",
        "width": 1654,
        "height": 2339,
        "items": [],
        "tables": [
            {
                "uuid": "e661c3b5-b57b-36c4-9bf2-36c89c03b1d3",
                "bounding_box": [94.0, 539.0, 1270.0, 1877.0],
                "cells": [
                    {
                        "uuid": "49465de7-55be-3cc9-af1f-8231cd398ed8",
                        "bounding_box": [125.49425506591797, 540.958251953125, 284.058349609375, 609.9114990234375],
                        "text": "Candriam Bonds Credit Opportunities",
                        "row_number": 1,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9855870008468628,
                    },
                    {
                        "uuid": "bfc79f21-4247-30df-8a1e-5a6158a485c6",
                        "bounding_box": [401.4119873046875, 1567.8818359375, 478.17529296875, 1594.5858154296875],
                        "text": "227.21",
                        "row_number": 9,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9811129570007324,
                    },
                    {
                        "uuid": "4f579d89-3da0-3287-b2da-048be60fa0ee",
                        "bounding_box": [993.150146484375, 1567.5748291015625, 1071.13916015625, 1595.33251953125],
                        "text": "156.82",
                        "row_number": 9,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.980900228023529,
                    },
                    {
                        "uuid": "d19be95e-0fa4-3580-a24e-edfd60e753c7",
                        "bounding_box": [1204.041259765625, 1023.6016235351562, 1268.241943359375, 1052.302978515625],
                        "text": "1,395",
                        "row_number": 4,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9807637929916382,
                    },
                    {
                        "uuid": "541d3b8e-ca20-3b7f-93cd-06177d219578",
                        "bounding_box": [388.65875244140625, 1023.9036865234375, 479.9609680175781, 1052.2979736328125],
                        "text": "772,865",
                        "row_number": 4,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9803057909011841,
                    },
                    {
                        "uuid": "18f9451d-47e8-3b1c-94d7-5c5cae0dcfee",
                        "bounding_box": [993.045166015625, 1606.0902099609375, 1071.206787109375, 1633.4171142578125],
                        "text": "154.36",
                        "row_number": 10,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9774023294448853,
                    },
                    {
                        "uuid": "89cb4d04-d2ef-3375-9694-f83eef8798c4",
                        "bounding_box": [993.4404907226562, 1023.4800415039062, 1071.055419921875, 1051.977783203125],
                        "text": "14,305",
                        "row_number": 4,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9769163131713867,
                    },
                    {
                        "uuid": "af1510a6-1b8d-37d9-a70f-1d8e59513eac",
                        "bounding_box": [597.9375, 1568.2313232421875, 677.766845703125, 1595.3912353515625],
                        "text": "228.87",
                        "row_number": 9,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9754486083984375,
                    },
                    {
                        "uuid": "3f2b3a87-15b5-3b4a-8e75-d24514307b43",
                        "bounding_box": [797.3232421875, 1567.4969482421875, 874.4328002929688, 1594.8890380859375],
                        "text": "159.03",
                        "row_number": 9,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9753007888793945,
                    },
                    {
                        "uuid": "65ed0467-cb42-3c2c-a5ef-e0230c1ef78c",
                        "bounding_box": [205.7230987548828, 1566.915771484375, 283.59100341796875, 1594.9793701171875],
                        "text": "183.86",
                        "row_number": 9,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9752539396286011,
                    },
                    {
                        "uuid": "9e4d3fe6-3d53-3801-ba18-eb44f224d6ae",
                        "bounding_box": [401.55224609375, 1606.6212158203125, 480.878662109375, 1633.172607421875],
                        "text": "222.10",
                        "row_number": 10,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.974498450756073,
                    },
                    {
                        "uuid": "d768b496-c6f5-39c8-b16a-e72ecc2fa8b4",
                        "bounding_box": [569.65234375, 1064.1385498046875, 676.7476806640625, 1090.50830078125],
                        "text": "(841,113)",
                        "row_number": 5,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9729200601577759,
                    },
                    {
                        "uuid": "c5693aaa-219b-390d-9770-70426e56be5d",
                        "bounding_box": [206.0513153076172, 1605.47314453125, 283.477294921875, 1633.956298828125],
                        "text": "181.79",
                        "row_number": 10,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9729058146476746,
                    },
                    {
                        "uuid": "553011be-5715-3a77-a511-2379a0465242",
                        "bounding_box": [796.7996215820312, 1607.2425537109375, 872.5521240234375, 1632.9739990234375],
                        "text": "155.91",
                        "row_number": 10,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9688429236412048,
                    },
                    {
                        "uuid": "d00da177-ee11-3421-94c1-dcb29a5fe52c",
                        "bounding_box": [598.25830078125, 1605.69287109375, 676.5005493164062, 1633.4932861328125],
                        "text": "223.24",
                        "row_number": 10,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9682562351226807,
                    },
                    {
                        "uuid": "2abe96c5-c167-3dfe-b9c7-53dd8f2d58ee",
                        "bounding_box": [1191.1055908203125, 1566.9591064453125, 1265.17919921875, 1594.1065673828125],
                        "text": "149.51",
                        "row_number": 9,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9672756195068359,
                    },
                    {
                        "uuid": "3c71125c-cc76-32e4-a721-03122afc72f6",
                        "bounding_box": [584.6256103515625, 1023.3775634765625, 676.3963012695312, 1052.3546142578125],
                        "text": "679,644",
                        "row_number": 4,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9652320742607117,
                    },
                    {
                        "uuid": "9f7cd129-7078-3508-9888-f01289142163",
                        "bounding_box": [350.5904541015625, 1062.985595703125, 480.7196350097656, 1090.94287109375],
                        "text": "(1,447,261)",
                        "row_number": 5,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9650256037712097,
                    },
                    {
                        "uuid": "e25ba17d-ddf3-3193-8c2d-3fe957690b63",
                        "bounding_box": [977.8805541992188, 1062.0576171875, 1070.9984130859375, 1091.6904296875],
                        "text": "(36,582)",
                        "row_number": 5,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9646384716033936,
                    },
                    {
                        "uuid": "a272455a-c201-3d21-bda9-af7407b4cbec",
                        "bounding_box": [780.2045288085938, 1063.4993896484375, 874.2966918945312, 1090.3651123046875],
                        "text": "(82,102)",
                        "row_number": 5,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9643425941467285,
                    },
                    {
                        "uuid": "c575c70a-8261-3756-9398-893fa5609222",
                        "bounding_box": [120.5513687133789, 1394.454345703125, 282.6224365234375, 1421.420654296875],
                        "text": "2,040,776,153",
                        "row_number": 7,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9638470411300659,
                    },
                    {
                        "uuid": "62b03cba-5fd4-3d58-91fd-092b24355a3d",
                        "bounding_box": [141.6884002685547, 1433.8345947265625, 283.42791748046875, 1459.762939453125],
                        "text": "837,522,782",
                        "row_number": 8,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9630524516105652,
                    },
                    {
                        "uuid": "a332c675-d48b-377d-8d0d-7ee09aa9cde6",
                        "bounding_box": [190.3619384765625, 1022.6398315429688, 284.41253662109375, 1053.62646484375],
                        "text": "373,120",
                        "row_number": 4,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9622296094894409,
                    },
                    {
                        "uuid": "325a7cb9-bed4-3756-bb88-c592ae50fbc2",
                        "bounding_box": [176.48977661132812, 1063.5234375, 283.30914306640625, 1090.2584228515625],
                        "text": "(629,877)",
                        "row_number": 5,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9565564393997192,
                    },
                    {
                        "uuid": "6c4b3962-2b2e-3ef4-86e6-bd478e4e3677",
                        "bounding_box": [781.3336791992188, 1022.8023681640625, 874.5203857421875, 1053.30859375],
                        "text": "485,538",
                        "row_number": 4,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9557906985282898,
                    },
                    {
                        "uuid": "16a8c826-ee2e-3eb5-b906-55b34031f33a",
                        "bounding_box": [1152.13916015625, 659.1488647460938, 1185.6856689453125, 684.5174560546875],
                        "text": "R2",
                        "row_number": 2,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9507045149803162,
                    },
                    {
                        "uuid": "98104566-4028-32f4-8ee8-8b2277a34a9c",
                        "bounding_box": [568.8727416992188, 659.5617065429688, 587.183349609375, 684.954345703125],
                        "text": "",
                        "row_number": 2,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.9140693545341492,
                    },
                    {
                        "uuid": "7bb9eec9-bfc2-38da-a73c-a55eb26c6e42",
                        "bounding_box": [765.7601928710938, 658.85986328125, 786.072998046875, 685.46337890625],
                        "text": "",
                        "row_number": 2,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.8892556428909302,
                    },
                    {
                        "uuid": "cae7a64e-945f-3201-b730-cd70a32b047d",
                        "bounding_box": [372.6829528808594, 659.8545532226562, 390.6653747558594, 684.9542236328125],
                        "text": "",
                        "row_number": 2,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.8454699516296387,
                    },
                    {
                        "uuid": "6e4bc23a-1081-3c2a-a084-ee836c63da91",
                        "bounding_box": [174.4996795654297, 658.3178100585938, 195.52696228027344, 685.1928100585938],
                        "text": "N",
                        "row_number": 2,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.8390962481498718,
                    },
                    {
                        "uuid": "f9f557a2-9536-3118-9d72-9d816b88021a",
                        "bounding_box": [1251.5390625, 1062.8822021484375, 1267.7294921875, 1088.6181640625],
                        "text": "",
                        "row_number": 5,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.8283389806747437,
                    },
                    {
                        "uuid": "69b1c2a4-6b46-3ccd-8bc0-36fe703d97fb",
                        "bounding_box": [1256.9453125, 1610.80078125, 1268.1087646484375, 1631.687255859375],
                        "text": "",
                        "row_number": 10,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": 0.5811387896537781,
                    },
                    {
                        "uuid": "4f5073f1-ae7a-3446-b068-beb1277fa104",
                        "bounding_box": [
                            205.69393920898438,
                            1646.9119873046875,
                            283.97767639160156,
                            1862.0328369140625,
                        ],
                        "text": "176.73",
                        "row_number": 11,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "a2dded91-2aeb-3cda-bd65-d9c174520caf",
                        "bounding_box": [124.79571342468262, 1157.5601806640625, 285.39479064941406, 1381.83935546875],
                        "text": "354,414 EUR 1,944,638,470",
                        "row_number": 6,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "ed99cf63-a812-35dc-80b6-b74408e75e50",
                        "bounding_box": [1143.533447265625, 756.3648834228516, 1268.046142578125, 1012.0814208984375],
                        "text": "Distribution Distribution Ausschittung Uitkering",
                        "row_number": 3,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "89b21711-dfb4-3f66-a7cb-7283a44f829f",
                        "bounding_box": [155.2247200012207, 757.0559234619141, 283.2872772216797, 1012.6659851074219],
                        "text": "Capitalization Capitalisation Thesaurierung Kapitalisatie 611,171",
                        "row_number": 3,
                        "col_number": 1,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "95b2dd1a-3c7f-3783-8db8-ec1e07b60042",
                        "bounding_box": [550.94140625, 756.5910034179688, 677.4175415039062, 1011.9159545898438],
                        "text": "Capitalization Capitalisation Thesaurierung Kapitalisatie 883,060",
                        "row_number": 3,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "8802810c-a112-38c6-a04b-2a9f338b7fa5",
                        "bounding_box": [367.5026550292969, 1157.6786499023438, 481.69256591796875, 1343.6871337890625],
                        "text": "1,318,649 EUR",
                        "row_number": 6,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "47e29521-b8fc-3cfd-8d88-8b97b0bfdfe7",
                        "bounding_box": [1204.1654052734375, 1158.48974609375, 1268.6932373046875, 1343.3408203125],
                        "text": "1,395 EUR",
                        "row_number": 6,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "2e2910be-869d-3ae8-825a-5be06277f0a2",
                        "bounding_box": [993.0194702148438, 1647.2979736328125, 1071.7824096679688, 1859.4637451171875],
                        "text": "149.37",
                        "row_number": 11,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "458fd27f-3b44-324e-a7d1-003040b80c81",
                        "bounding_box": [354.45416259765625, 756.1832427978516, 480.6596984863281, 1011.7024841308594],
                        "text": "Capitalization Capitalisation Thesaurierung Kapitalisatie 1,993,045",
                        "row_number": 3,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "ee66bba7-a037-3959-8614-3e7772889724",
                        "bounding_box": [992.0205688476562, 1157.9804077148438, 1071.3461303710938, 1344.1744384765625],
                        "text": "31,590 CHF",
                        "row_number": 6,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "55842dae-77b8-335b-9a22-04c5ad243997",
                        "bounding_box": [586.2476806640625, 1157.8330688476562, 679.1338500976562, 1343.2540893554688],
                        "text": "721,591 EUR",
                        "row_number": 6,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "5eee8a34-79dd-3f5c-a5ed-467108aad988",
                        "bounding_box": [797.1563720703125, 1647.0692138671875, 875.0097045898438, 1861.8829345703125],
                        "text": "150.28",
                        "row_number": 11,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "b2e4efab-5e72-3cbb-917c-3e2517f9c9db",
                        "bounding_box": [902.4461669921875, 755.9923553466797, 1072.0050659179688, 1012.2868041992188],
                        "text": "Capitalization RCH Capitalisation RCH Thesaurierung RCH Kapitalisatie RCH 53,867",
                        "row_number": 3,
                        "col_number": 5,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "2c968497-efe3-3ba0-afec-bc7a0d8a36cf",
                        "bounding_box": [597.8861999511719, 1647.6390380859375, 677.8275756835938, 1861.5386962890625],
                        "text": "213.83",
                        "row_number": 11,
                        "col_number": 3,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "ea97e1c0-9487-38a6-9c5c-1981f067a706",
                        "bounding_box": [748.5528564453125, 756.223388671875, 874.7618408203125, 1012.126953125],
                        "text": "Capitalization Capitalisation Thesaurierung Kapitalisatie 186,290",
                        "row_number": 3,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "6b740522-4a51-37a7-adc7-49efe3f55c19",
                        "bounding_box": [781.4551391601562, 1157.5936889648438, 875.3937377929688, 1343.7017211914062],
                        "text": "589,726 EUR",
                        "row_number": 6,
                        "col_number": 4,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "0b124773-1147-3329-b88f-da9690045a3d",
                        "bounding_box": [401.41778564453125, 1647.12939453125, 480.584228515625, 1862.4959716796875],
                        "text": "213.26",
                        "row_number": 11,
                        "col_number": 2,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                    {
                        "uuid": "9b6797c0-a759-3c1f-9933-7e5fc11fbf5b",
                        "bounding_box": [1256.1204833984375, 1654.3905029296875, 1268.257568359375, 1864.7877197265625],
                        "text": "",
                        "row_number": 11,
                        "col_number": 6,
                        "row_span": 1,
                        "col_span": 1,
                        "score": -1.0,
                    },
                ],
                "items": [
                    {
                        "uuid": "bf05ab53-605c-3a3f-a4f9-a6525198fdc8",
                        "bounding_box": [874.1968994140625, 540.0, 1071.980224609375, 1876.0],
                        "category": "COLUMN",
                        "score": 0.9861695170402527,
                    },
                    {
                        "uuid": "61254f8c-3a7d-3e43-a7d8-ac070c72608f",
                        "bounding_box": [677.861328125, 540.0, 874.1968994140625, 1876.0],
                        "category": "COLUMN",
                        "score": 0.9854723811149597,
                    },
                    {
                        "uuid": "6932439f-defd-3159-ac8a-8e2514fa814e",
                        "bounding_box": [479.7930908203125, 540.0, 677.861328125, 1876.0],
                        "category": "COLUMN",
                        "score": 0.9804749488830566,
                    },
                    {
                        "uuid": "6d3ad54f-c2d4-3d18-be1e-5f6ff718b2e1",
                        "bounding_box": [284.3885498046875, 540.0, 479.7930908203125, 1876.0],
                        "category": "COLUMN",
                        "score": 0.9804415106773376,
                    },
                    {
                        "uuid": "ad335a49-4be8-3e26-ae82-f0eae84bfc19",
                        "bounding_box": [95.0, 1382.655029296875, 1269.0, 1423.134765625],
                        "category": "ROW",
                        "score": 0.9592576026916504,
                    },
                    {
                        "uuid": "14407b01-bced-31da-b945-f9aee51b6919",
                        "bounding_box": [95.0, 1093.056640625, 1269.0, 1382.655029296875],
                        "category": "ROW",
                        "score": 0.958727240562439,
                    },
                    {
                        "uuid": "7cb6aee5-a9dc-31c3-af44-c5a3c3ad998f",
                        "bounding_box": [1071.980224609375, 540.0, 1269.0, 1876.0],
                        "category": "COLUMN",
                        "score": 0.9512147903442383,
                    },
                    {
                        "uuid": "d1b30ad4-f8b4-3abd-ab2a-330f8e55ff1a",
                        "bounding_box": [95.0, 540.0, 1269.0, 610.9435424804688],
                        "category": "ROW",
                        "score": 0.9469785690307617,
                    },
                    {
                        "uuid": "98292a14-1e9e-3531-a06d-d8b77fda3b2b",
                        "bounding_box": [95.0, 610.9435424804688, 1269.0, 686.9849243164062],
                        "category": "ROW",
                        "score": 0.9108189344406128,
                    },
                    {
                        "uuid": "bfced1fe-4f48-3e73-8c5f-ad6174b34f2d",
                        "bounding_box": [95.0, 540.0, 284.3885498046875, 1876.0],
                        "category": "COLUMN",
                        "score": 0.8943723440170288,
                    },
                    {
                        "uuid": "0ba304a8-6a21-3905-8970-6ea8fca5011d",
                        "bounding_box": [95.0, 686.9849243164062, 1269.0, 1012.4632568359375],
                        "category": "ROW",
                        "score": 0.8671841621398926,
                    },
                    {
                        "uuid": "e73a6540-c186-346c-a763-48150565818f",
                        "bounding_box": [95.0, 1461.740966796875, 1269.0, 1594.39208984375],
                        "category": "ROW",
                        "score": 0.8563193678855896,
                    },
                    {
                        "uuid": "baed7acb-95d5-3f80-a365-ec68b5f3e4d7",
                        "bounding_box": [95.0, 1594.39208984375, 1269.0, 1634.2286376953125],
                        "category": "ROW",
                        "score": 0.8350787162780762,
                    },
                    {
                        "uuid": "5940c138-c3d3-3cbc-85d6-4ba85c9f7161",
                        "bounding_box": [95.0, 1423.134765625, 1269.0, 1461.740966796875],
                        "category": "ROW",
                        "score": 0.8347094058990479,
                    },
                    {
                        "uuid": "e0fb298b-88b0-3b09-b506-956eaf6813d9",
                        "bounding_box": [95.0, 1634.2286376953125, 1269.0, 1876.0],
                        "category": "ROW",
                        "score": 0.8227292895317078,
                    },
                    {
                        "uuid": "8b3916f8-b434-32d6-9879-87a9b71d9860",
                        "bounding_box": [95.0, 1012.4632568359375, 1269.0, 1052.615966796875],
                        "category": "ROW",
                        "score": 0.7618075609207153,
                    },
                    {
                        "uuid": "551b1371-e41f-3988-b0a3-66dcea46d8a7",
                        "bounding_box": [95.0, 1052.615966796875, 1269.0, 1093.056640625],
                        "category": "ROW",
                        "score": 0.7100715637207031,
                    },
                ],
                "number_rows": 11,
                "number_cols": 6,
                "html": "<table><tr><td>Candriam Bonds Credit Opportunities</td><td></td><td></td><td></td><td>"
                "</td><td></td></tr><tr><td>R2</td><td></td><td></td><td></td><td></td><td>N</td></tr>"
                "<tr><td>Distribution Distribution Ausschittung Uitkering</td><td>Capitalization Capitalisation "
                "Thesaurierung Kapitalisatie 611,171</td><td>Capitalization Capitalisation Thesaurierung "
                "Kapitalisatie 883,060</td><td>Capitalization Capitalisation Thesaurierung Kapitalisatie "
                "1,993,045</td><td>Capitalization RCH Capitalisation RCH Thesaurierung RCH Kapitalisatie RCH "
                "53,867</td><td>Capitalization Capitalisation Thesaurierung Kapitalisatie 186,290</td></tr>"
                "<tr><td>1,395</td><td>772,865</td><td>14,305</td><td>679,644</td><td>373,120</td><td>485,538"
                "</td></tr><tr><td>(841,113)</td><td>(1,447,261)</td><td>(36,582)</td><td>(82,102)</td><td>"
                "(629,877)</td><td></td></tr><tr><td>354,414 EUR 1,944,638,470</td><td>1,318,649 EUR</td><td>"
                "1,395 EUR</td><td>31,590 CHF</td><td>721,591 EUR</td><td>589,726 EUR</td></tr><tr><td>2,040,"
                "776,153</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>837,522,782</td><td>"
                "</td><td></td><td></td><td></td><td></td></tr><tr><td>227.21</td><td>156.82</td><td>228.87</td>"
                "<td>159.03</td><td>183.86</td><td>149.51</td></tr><tr><td>154.36</td><td>222.10</td><td>181.79"
                "</td><td>155.91</td><td>223.24</td><td></td></tr><tr><td>176.73</td><td>149.37</td><td>150.28"
                "</td><td>213.83</td><td>213.26</td><td></td></tr></table>",
                "score": 0.6471484303474426,
            }
        ],
        "image": "",
    }
    file_name = "sample.png"
    number_tables = 1
    number_cells = 50
    first_cell_text = "Candriam Bonds Credit Opportunities"

    def get_page_dict(self) -> JsonDict:
        """
        page dictionary
        """
        return self.page_dict


_SAMPLE_XFUND = {
    "lang": "de",
    "version": "0.1",
    "split": "val",
    "documents": [
        {
            "img": {"fname": "/path/to/dir/de_val_0.jpg", "width": 2480, "height": 3508},
            "id": "de_val_0",
            "uid": "09006d3b9d97e3797ac9b59464f6a8f487da6ad4176d0fbd53caa9ecf1a7bca4",
            "document": [
                {
                    "box": [325, 183, 833, 231],
                    "text": "Akademisches Auslandsamt",
                    "label": "other",
                    "words": [
                        {"box": [325, 184, 578, 230], "text": "Akademisches"},
                        {"box": [586, 186, 834, 232], "text": "Auslandsamt"},
                    ],
                    "linking": [],
                    "id": 0,
                },
                {
                    "box": [1057, 413, 1700, 483],
                    "text": "Bewerbungsformular",
                    "label": "header",
                    "words": [{"box": [1058, 413, 1701, 482], "text": "Bewerbungsformular"}],
                    "linking": [],
                    "id": 15,
                },
            ],
        }
    ],
}


@dataclass
class DatapointXfund:
    """
    Xfund datapoint sample
    """

    dp = _SAMPLE_XFUND["documents"][0]

    category_names_mapping = {"other": names.C.O, "question": names.C.Q, "answer": names.C.A, "header": names.C.HEAD}
    layout_input = {
        "image": np.ones((1000, 1000, 3)),
        "ids": [
            "CLS",
            "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
            "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
            "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
            "0d0600cf-df94-34fa-9b30-5ecbbd1b36ab",
            "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
            "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
            "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
            "34bb95dc-7fe6-3982-9dd5-e49d362b3fd7",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "a77dfce6-32ff-31b4-8e39-cbbdd4c0acf1",
            "SEP",
        ],
        "boxes": [
            [0.0, 0.0, 0.0, 0.0],
            [325.0, 184.0, 578.0, 230.0],
            [325.0, 184.0, 578.0, 230.0],
            [325.0, 184.0, 578.0, 230.0],
            [325.0, 184.0, 578.0, 230.0],
            [586.0, 186.0, 834.0, 232.0],
            [586.0, 186.0, 834.0, 232.0],
            [586.0, 186.0, 834.0, 232.0],
            [586.0, 186.0, 834.0, 232.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1058.0, 413.0, 1701.0, 482.0],
            [1000.0, 1000.0, 1000.0, 1000.0],
        ],
        "tokens": [
            "CLS",
            "aka",
            "##de",
            "##mis",
            "##ches",
            "aus",
            "##lands",
            "##am",
            "##t",
            "be",
            "##wer",
            "##bu",
            "##ng",
            "##sf",
            "##or",
            "##mu",
            "##lar",
            "SEP",
        ],
        "input_ids": [
            [
                101,
                9875,
                3207,
                15630,
                8376,
                17151,
                8653,
                3286,
                2102,
                2022,
                13777,
                8569,
                3070,
                22747,
                2953,
                12274,
                8017,
                102,
            ]
        ],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        "token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    }

    def get_category_names_mapping(self) -> Dict[str, str]:
        """
        category_names_mapping
        """
        return self.category_names_mapping

    def get_layout_input(self) -> JsonDict:
        """
        layout_input
        """
        return self.layout_input

    def get_token_class_results(self) -> List[TokenClassResult]:
        """
        List of TokenClassResult
        """
        uuids = self.layout_input["ids"]
        input_ids = self.layout_input["input_ids"][0]  # type: ignore
        token_class_predictions = [0, 1, 1, 0, 1, 2, 1, 1, 1, 0, 0, 0, 1, 2, 1, 0, 1, 1]
        tokens = self.layout_input["tokens"]
        return [
            TokenClassResult(uuid=out[0], token_id=out[1], class_id=out[2], token=out[3])
            for out in zip(uuids, input_ids, token_class_predictions, tokens)  # type: ignore
        ]

    @staticmethod
    def get_categories_semantics() -> List[str]:
        """
        categories semantics
        """
        return ["FOO"]

    @staticmethod
    def get_categories_bio() -> List[str]:
        """
        categories bio
        """
        return ["B", "I", "O"]

    @staticmethod
    def get_token_class_names() -> List[str]:
        """
        token class names
        """
        return [
            "B-FOO",
            "I-FOO",
            "I-FOO",
            "B-FOO",
            "I-FOO",
            "O",
            "I-FOO",
            "I-FOO",
            "I-FOO",
            "B-FOO",
            "B-FOO",
            "B-FOO",
            "I-FOO",
            "O",
            "I-FOO",
            "B-FOO",
            "I-FOO",
            "I-FOO",
        ]


@dataclass
class IIITar13KJson:
    """
    Xfund datapoint sample already converted to json format
    """

    dp = {
        "annotation": "2004",
        "filename": "/home/janis/.cache/deepdoctection/datasets/iiitar13k/validation_xml/ar_alphabet_2004_eng_32.xml",
        "path": "/home/cvit/Desktop/Phase2_OpenText_Annotation/Annual_Report/Alphabet/2004/NASDAQ_GOOG_2004_80.png",
        "database": "Unknown",
        "width": "1100",
        "height": "850",
        "depth": "3",
        "segmented": "0",
        "objects": [
            {
                "name": "table",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": 0,
                "xmin": 127,
                "ymin": 202,
                "xmax": 1006,
                "ymax": 580,
            }
        ],
    }

    categories_name_as_keys = {names.C.TAB: "1", names.C.LOGO: "2", names.C.FIG: "3", names.C.SIGN: "4"}
    category_names_mapping = {
        "natural_image": names.C.FIG,
        "figure": names.C.FIG,
        "logo": names.C.LOGO,
        "signature": names.C.SIGN,
        "table": names.C.TAB,
    }

    first_ann_box = Box(ulx=127, uly=202, w=1006 - 127, h=580 - 202)

    def get_category_names_mapping(self) -> Dict[str, str]:
        """
        category_names_mapping
        """
        return self.category_names_mapping

    def get_number_anns(self) -> int:
        """
        number of annotations
        """
        return len(self.dp["objects"])

    def get_width(self) -> float:
        """
        imager width
        """
        return float(self.dp["width"])  # type: ignore

    def get_height(self) -> float:
        """
        imager height
        """
        return float(self.dp["height"])  # type: ignore

    def get_first_ann_box(self) -> Box:
        """
        box coordinates of first annotation
        """
        return self.first_ann_box

    @staticmethod
    def get_first_ann_category_name() -> str:
        """
        first annotation category name
        """
        return names.C.TAB

    def get_categories_name_as_keys(self) -> Dict[str, str]:
        """
        categories name as keys
        """
        return self.categories_name_as_keys
