# -*- coding: utf-8 -*-
# File: convert.py

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
Conversion functions associated to functionalities of datapoint classes
"""

import base64
import copy
from dataclasses import fields, is_dataclass
from io import BytesIO
from shutil import which
from typing import Any, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from PyPDF2 import PdfFileReader  # type: ignore

from ..utils.detection_types import ImageType
from ..utils.develop import deprecated
from ..utils.pdf_utils import pdf_to_np_array

__all__ = [
    "convert_b64_to_np_array",
    "convert_np_array_to_b64",
    "convert_np_array_to_b64_b",
    "convert_pdf_bytes_to_np_array_v2",
    "box_to_point4",
    "point4_to_box",
]


def as_dict(obj: Any, dict_factory) -> Union[Any]:  # type: ignore
    """
    custom func: as_dict to use instead of :func:`dataclasses.asdict` . It also checks if a dataclass has a
    :meth:'remove_keys' and will remove all attributes that are returned. Ensures that private attributes are not taken
    into account when generating a dict.

    :param obj: Object to convert into a dict.
    :param dict_factory: A factory to generate the dict.
    """

    if is_dataclass(obj):
        result = []
        for attribute in fields(obj):
            value = as_dict(getattr(obj, attribute.name), dict_factory)
            if hasattr(obj, "remove_keys"):
                if attribute.name in obj.remove_keys():
                    continue
            result.append((attribute.name, value))
        return dict_factory(result)
    if isinstance(obj, (list, tuple)):
        return type(obj)(as_dict(v, dict_factory) for v in obj)  # pylint: disable=E0110
    if isinstance(obj, dict):
        return type(obj)(  # pylint: disable=E0110
            (as_dict(k, dict_factory), as_dict(v, dict_factory)) for k, v in obj.items()
        )
    if isinstance(obj, (np.float32, np.float64)):
        obj = obj.astype(float)
    return copy.deepcopy(obj)


def convert_b64_to_np_array(image: str) -> ImageType:
    """
    Converts an image in base4 string encoding representation to a numpy array of shape (width,height,channel).

    :param image: An image as base64 string.
    :return: numpy array.
    """
    np_array = np.fromstring(base64.b64decode(image), np.uint8)  # type: ignore
    np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR).astype(np.float32)
    return np_array


def convert_np_array_to_b64(np_image: ImageType) -> str:
    """
    Converts an image from numpy array into a base64 string encoding representation

    :param np_image: An image as numpy array.
    :return: An image as base64 string.
    """
    np_encode = cv2.imencode(".png", np_image)
    image = base64.b64encode(np_encode[1]).decode("utf-8")
    return image


def convert_np_array_to_b64_b(np_image: ImageType) -> bytes:
    """
    Converts an image from numpy array into a base64 bytes encoding representation

    :param np_image: An image as numpy array.
    :return: An image as base64 bytes.
    """
    np_encode = cv2.imencode(".png", np_image)
    b_image = np_encode[1].tobytes()
    return b_image


@deprecated("Use convert_pdf_bytes_to_np_array_v2", "2022-02-23")
def convert_pdf_bytes_to_np_array(pdf_bytes: bytes, dpi: Optional[int] = None) -> ImageType:
    """
    Converts a pdf passed as bytes into a numpy array. Note, that this method expects poppler to be installed.
    Please check the installation guides at https://poppler.freedesktop.org/ . If no value for dpi is provided
    the output size will be determined by the mediaBox of the pdf file ready. Note, that the image size will be in
    this case rather small.

    :param pdf_bytes: A pdf as bytes object. A byte representation can from a pdf file can be generated e.g. with
                      :func:`utils.fs.load_bytes_from_pdf_file`
    :param dpi: The dpi value of the resulting output image. For high resolution set dpi=300.
    :return: Image as numpy array.
    """
    from pdf2image import convert_from_bytes  # type: ignore # pylint: disable=C0415, E0401

    assert which("pdftoppm") is not None, "convert_pdf_bytes_to_np_array requires poppler to be installed"

    with BytesIO(pdf_bytes) as pdf_file:
        pdf = PdfFileReader(pdf_file).getPage(0)
    shape = pdf.mediaBox
    height = shape[3] - shape[1]
    width = shape[2] - shape[0]
    buffered = BytesIO()

    if dpi is None:

        image = convert_from_bytes(pdf_bytes, size=(width, height))[0]
    else:
        image = convert_from_bytes(pdf_bytes, dpi=dpi)[0]

    image.save(buffered, format="JPEG")
    image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    np_array = convert_b64_to_np_array(image)
    return np_array


def convert_pdf_bytes_to_np_array_v2(pdf_bytes: bytes, dpi: Optional[int] = None) -> ImageType:
    """
    Converts a pdf passed as bytes into a numpy array. Note, that this method expects poppler to be installed. This
    function, however does not rely on the wrapper pdf2image but uses a function of this lib which calls poppler
    directly.

    :param pdf_bytes: A pdf as bytes object. A byte representation can from a pdf file can be generated e.g. with
                      :func:`utils.fs.load_bytes_from_pdf_file`
    :param dpi: The dpi value of the resulting output image. For high resolution set dpi=300.
    :return: Image as numpy array.
    """

    with BytesIO(pdf_bytes) as pdf_file:
        pdf = PdfFileReader(pdf_file).getPage(0)
    shape = pdf.mediaBox
    height = shape[3] - shape[1]
    width = shape[2] - shape[0]

    if dpi is None:
        return pdf_to_np_array(pdf_bytes, size=(int(width), int(height)))
    return pdf_to_np_array(pdf_bytes, dpi=dpi)


def box_to_point4(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    :param boxes: nx4
    :return: (nx4)x2
    """
    box = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    box = box.reshape((-1, 2))
    return box


def point4_to_box(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    :param points: (nx4)x2
    :return: nx4 boxes (x1y1x2y2)
    """
    points = points.reshape((-1, 4, 2))
    min_xy = points.min(axis=1)  # nx2
    max_xy = points.max(axis=1)  # nx2
    return np.concatenate((min_xy, max_xy), axis=1)
