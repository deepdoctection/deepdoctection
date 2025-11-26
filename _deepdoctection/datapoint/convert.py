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
Conversion functions for images and pdfs
"""

import base64
import copy
from dataclasses import fields, is_dataclass
from io import BytesIO
from shutil import which
from typing import Any, Optional, Union, no_type_check

import numpy as np
from numpy import uint8
from pypdf import PdfReader

from ..utils.develop import deprecated
from ..utils.error import DependencyError
from ..utils.pdf_utils import pdf_to_np_array
from ..utils.types import PixelValues
from ..utils.viz import viz_handler

__all__ = [
    "convert_b64_to_np_array",
    "convert_np_array_to_b64",
    "convert_np_array_to_b64_b",
    "convert_bytes_to_np_array",
    "convert_pdf_bytes_to_np_array_v2",
    "as_dict",
]


def as_dict(obj: Any, dict_factory) -> Union[Any]:  # type: ignore
    """
    Args:
        custom func: as_dict to use instead of `dataclasses.asdict` . It also checks if a dataclass has a
                     'remove_keys' and will remove all attributes that are returned. Ensures that private attributes
                     are not taken into account when generating a `dict`.

        obj: Object to convert into a dict.
        dict_factory: A factory to generate the dict.
    """

    if is_dataclass(obj):
        result = []
        for attribute in fields(obj):
            value = as_dict(getattr(obj, attribute.name), dict_factory)
            if hasattr(obj, "remove_keys"):
                if attribute.name in obj.remove_keys():
                    continue
            if hasattr(obj, "replace_keys"):
                old_to_new_keys = obj.replace_keys()
                if attribute.name in old_to_new_keys:
                    attribute.name = obj.replace_keys()[attribute.name]
            result.append((attribute.name, value))
        return dict_factory(result)
    if isinstance(obj, (list, tuple)):
        return type(obj)(as_dict(v, dict_factory) for v in obj)  # pylint: disable=E0110
    if isinstance(obj, dict):
        return type(obj)((as_dict(k, dict_factory), as_dict(v, dict_factory)) for k, v in obj.items())
    if isinstance(obj, (np.float32, np.float64)):
        obj = obj.astype(float)
    return copy.deepcopy(obj)


def convert_b64_to_np_array(image: str) -> PixelValues:
    """
    Converts an image in base4 string encoding representation to a `np.array` of shape `(width,height,channel)`.

    Args:
        image: An image as `base64` string.

    Returns:
        numpy array.
    """

    return viz_handler.convert_b64_to_np(image).astype(uint8)


def convert_np_array_to_b64(np_image: PixelValues) -> str:
    """
    Converts an image from numpy array into a base64 string encoding representation

    Args:
        np_image: An image as numpy array.

    Returns:
        An image as `base64` string.
    """
    return viz_handler.convert_np_to_b64(np_image)


@no_type_check
def convert_np_array_to_b64_b(np_image: PixelValues) -> bytes:
    """
    Converts an image from numpy array into a base64 bytes encoding representation

    Args:
        np_image: An image as numpy array.

    Returns:
        An image as `base64` bytes.
    """
    return viz_handler.encode(np_image)


def convert_bytes_to_np_array(image_bytes: bytes) -> PixelValues:
    """
    Converts an image in `bytes` to a `np.array`

    Args:
        image_bytes: An image as bytes.

    Returns:
        numpy array.
    """
    return viz_handler.convert_bytes_to_np(image_bytes)


@deprecated("Use convert_pdf_bytes_to_np_array_v2", "2022-02-23")
def convert_pdf_bytes_to_np_array(pdf_bytes: bytes, dpi: Optional[int] = None) -> PixelValues:
    """
    Converts a pdf passed as bytes into a `np.array`. Note, that this method expects poppler to be installed.
    Please check the installation guides at <https://poppler.freedesktop.org/> . If no value for `dpi` is provided
    the output size will be determined by the mediaBox of the pdf file ready.

    Note:
        The image size will be in this case rather small.

    Args:
        pdf_bytes: A pdf as bytes object. A byte representation can from a pdf file can be generated e.g. with
                   `utils.fs.load_bytes_from_pdf_file`
        dpi: The dpi value of the resulting output image. For high resolution set `dpi=300`.

    Returns:
        Image as numpy array.
    """
    from pdf2image import convert_from_bytes  # type: ignore # pylint: disable=C0415, E0401

    if which("pdftoppm") is None:
        raise DependencyError("convert_pdf_bytes_to_np_array requires poppler to be installed")

    with BytesIO(pdf_bytes) as pdf_file:
        pdf = PdfReader(pdf_file).pages[0]
    shape = pdf.mediabox  # pylint: disable=E1101
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
    return np_array.astype(uint8)


def convert_pdf_bytes_to_np_array_v2(
    pdf_bytes: bytes, dpi: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None
) -> PixelValues:
    """
    Converts a pdf passed as bytes into a numpy array. We use poppler or `pdfmium` to convert the pdf to an image.

    Note:
        If both is available you can steer the selection of the render engine with environment variables:

        ```
        # Set the environment variable to use poppler
        USE_DD_POPPLER="1" or  ("TRUE", "True")
        USE_DD_PDFIUM="0" or anything that is not ("1", "TRUE", "True")
        ```

    Args:
        pdf_bytes: A pdf as bytes object. A byte representation can from a pdf file can be generated e.g. with
                   `utils.fs.load_bytes_from_pdf_file`
        dpi: The dpi value of the resulting output image. For high resolution set dpi=300.
        width: The width of the resulting output image. This option does only work when using Poppler as
               PDF renderer
        height: The height of the resulting output image. This option does only work when using Poppler as
                PDF renderer

    Returns:
        Image as numpy array.
    """

    if dpi is None:
        if width is None or height is None:
            with BytesIO(pdf_bytes) as pdf_file:
                pdf = PdfReader(pdf_file).pages[0]
            shape = pdf.mediabox  # pylint: disable=E1101
            height = shape[3] - shape[1]
            width = shape[2] - shape[0]
        return pdf_to_np_array(pdf_bytes, size=(int(width), int(height)))  # type: ignore
    return pdf_to_np_array(pdf_bytes, dpi=dpi)
