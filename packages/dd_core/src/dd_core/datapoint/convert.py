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

from __future__ import annotations

from io import BytesIO
from typing import Optional, no_type_check

from lazy_imports import try_import
from numpy import uint8

from ..utils.pdf_utils import pdf_to_np_array
from ..utils.types import PixelValues
from ..utils.viz import viz_handler

with try_import() as pypdf_import_guard:
    from pypdf import PdfReader

with try_import() as torch_import_guard:
    import torch

__all__ = [
    "convert_b64_to_np_array",
    "convert_np_array_to_b64",
    "convert_np_array_to_b64_b",
    "convert_bytes_to_np_array",
    "convert_pdf_bytes_to_np_array_v2",
    "convert_np_array_to_torch",
]


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
                try:
                    pdf = PdfReader(pdf_file).pages[0]  # type: ignore
                except NameError:
                    raise ImportError("pypdf is not installed.")
            shape = pdf.mediabox  # pylint: disable=E1101
            height = shape[3] - shape[1]
            width = shape[2] - shape[0]
        return pdf_to_np_array(pdf_bytes, size=(int(width), int(height)))  # type: ignore
    return pdf_to_np_array(pdf_bytes, dpi=dpi)


def convert_np_array_to_torch(np_image: PixelValues, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Converts a numpy image array into a torch tensor (dtype uint8) and moves it to the given device.

    Args:
        np_image: An image as numpy array.
        device: Target torch device.

    Returns:
        A torch.Tensor with dtype uint8 on the given device.

    Raises:
        ImportError: If torch is not installed.
    """
    try:
        tensor = torch.from_numpy(np_image)
    except NameError as exc:
        raise ImportError("torch is not installed.") from exc

    if device is None:
        device = torch.device("cpu")

    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)

    return tensor.to(device)
