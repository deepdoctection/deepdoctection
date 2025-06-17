# -*- coding: utf-8 -*-
# File: misc.py

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
Small mapping functions
"""

from __future__ import annotations

import ast
import os
from typing import Mapping, Optional, Sequence, Union

from lazy_imports import try_import

from ..datapoint.convert import convert_bytes_to_np_array, convert_pdf_bytes_to_np_array_v2
from ..datapoint.image import Image
from ..utils.fs import get_load_image_func, load_image_from_file
from ..utils.types import JsonDict
from ..utils.utils import is_file_extension
from .maputils import MappingContextManager, curry

with try_import() as import_guard:
    from lxml import etree  # pylint: disable=W0611


def to_image(
    dp: Union[str, Mapping[str, Union[str, bytes]]],
    dpi: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Image]:
    """
    Maps an input from `dataflow.SerializerFiles` or similar to an `Image`.

    Args:
        dp: Image.
        dpi: Dot per inch definition for PDF resolution when converting to `np.array`.
        width: Target width of the image. This option only works when using Poppler as PDF renderer.
        height: Target height of the image. This option only works when using Poppler as PDF renderer.

    Returns:
        Image

    Raises:
        TypeError: If `dp` is not of the expected type for converting to image.
    """

    file_name: Optional[str]
    location: Optional[str]
    image_bytes: Optional[bytes] = None

    if isinstance(dp, str):
        _, file_name = os.path.split(dp)
        location = dp
        page_number = 0
        document_id = None
    elif isinstance(dp, dict):
        file_name = str(dp.get("file_name", ""))
        page_number = int(dp.get("page_number", 0))
        location = str(dp.get("location", ""))
        document_id = dp.get("document_id")
        if location == "":
            location = str(dp.get("path", ""))
        image_bytes = dp.get("image_bytes")
    else:
        raise TypeError("datapoint not of expected type for converting to image")

    with MappingContextManager(dp_name=file_name) as mapping_context:
        dp_image = Image(file_name=file_name, location=location)
        dp_image.page_number = page_number
        if document_id:
            dp_image.document_id = document_id
        if file_name is not None:
            if is_file_extension(file_name, ".pdf") and isinstance(dp, dict):
                dp_image.pdf_bytes = dp.get("pdf_bytes")
                if dp_image.pdf_bytes is not None:
                    if isinstance(dp_image.pdf_bytes, bytes):
                        dp_image.image = convert_pdf_bytes_to_np_array_v2(
                            dp_image.pdf_bytes, dpi=dpi, width=width, height=height
                        )
            elif image_bytes is not None:
                dp_image.image = convert_bytes_to_np_array(image_bytes)
            else:
                dp_image.image = load_image_from_file(location)

    if mapping_context.context_error:
        return None

    return dp_image


def maybe_load_image(dp: Image) -> Image:
    """
    If `image` is `None`, loads the image.

    Args:
        dp: An `Image`.

    Returns:
        Image with attribute `image` not `None`.
    """

    if dp.image is None:
        assert dp.location is not None, "cannot load image, no location provided"
        loader = get_load_image_func(dp.location)
        dp.image = loader(dp.location)  # type: ignore
    return dp


def maybe_remove_image(dp: Image) -> Image:
    """
    Removes `image` if a location is provided.

    Args:
        dp: An `Image`.

    Returns:
        Image with attribute `image` set to `None`.
    """

    if dp.location is not None:
        dp.clear_image()
    return dp


@curry
def maybe_remove_image_from_category(dp: Image, category_names: Optional[Union[str, Sequence[str]]] = None) -> Image:
    """
    Removes `image` from image annotation for some `category_name`s.

    Args:
        dp: An `Image`.
        category_names: Category names.

    Returns:
        Image with `image` attributes from image annotations removed.
    """
    if category_names is None:
        category_names = []
    elif isinstance(category_names, str):
        category_names = [category_names]

    anns = dp.get_annotation(category_names=category_names)

    for ann in anns:
        ann.image = None

    return dp


def image_ann_to_image(dp: Image, category_names: Union[str, list[str]], crop_image: bool = True) -> Image:
    """
    Adds `image` to annotations with given category names.

    Args:
        dp: `Image`.
        category_names: A single or a list of category names.
        crop_image: If `True`, will add `np.array` to `image.image`.

    Returns:
        Image
    """

    img_anns = dp.get_annotation(category_names=category_names)
    for ann in img_anns:
        dp.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=crop_image)

    return dp


@curry
def maybe_ann_to_sub_image(
    dp: Image, category_names_sub_image: Union[str, list[str]], category_names: Union[str, list[str]], add_summary: bool
) -> Image:
    """
    Assigns to sub image with given category names all annotations with given category names whose bounding box lies
    within the bounding box of the sub image.

    Args:
        dp: `Image`.
        category_names_sub_image: A single or a list of category names that will form a sub image.
        category_names: A single or a list of category names that may be assigned to a sub image, conditioned on the
                        bounding box lying within the sub image.
        add_summary: If `True`, will add the whole summary annotation to the sub image.

    Returns:
        Image
    """

    anns = dp.get_annotation(category_names=category_names_sub_image)
    for ann in anns:
        dp.maybe_ann_to_sub_image(annotation_id=ann.annotation_id, category_names=category_names)
        if add_summary and ann.image:
            ann.image.summary = dp.summary

    return dp


@curry
def xml_to_dict(dp: JsonDict, xslt_obj: etree.XSLT) -> JsonDict:
    """
    Converts an XML object into a dict using an XSL style sheet.

    Example:
        ```python
        with open(path_xslt) as xsl_file:
            xslt_file = xsl_file.read().encode('utf-8')
        xml_obj = etree.XML(xslt_file, parser=etree.XMLParser(encoding='utf-8'))
        xslt_obj = etree.XSLT(xml_obj)
        df = MapData(df, xml_to_dict(xslt_obj))
        ```

    Args:
        dp: String representing the XML.
        xslt_obj: XSLT object to parse the string.

    Returns:
        Parsed XML as a dict.
    """

    output = str(xslt_obj(dp["xml"]))
    dp.pop("xml")
    dp["json"] = ast.literal_eval(output.replace('<?xml version="1.0"?>', ""))
    return dp
