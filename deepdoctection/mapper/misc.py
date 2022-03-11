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
Module for small mapping functions
"""

import ast
import os
from typing import Dict, List, Optional, Union

from lxml import etree  # type: ignore

from ..datapoint.convert import convert_pdf_bytes_to_np_array_v2
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.fs import get_load_image_func, is_file_extension, load_image_from_file
from .maputils import MappingContextManager, cur


def to_image(dp: Union[str, Dict[str, Union[str, bytes]]], dpi: Optional[int] = None) -> Optional[Image]:
    """
    Mapping an input from :class:`dataflow.SerializerFiles` or similar to an Image

    :param dp: Image
    :param dpi: dot per inch definition for pdf resolution when converting to numpy array
    :return: Image
    """

    file_name: Optional[str]
    location: Optional[str]

    if isinstance(dp, str):
        _, file_name = os.path.split(dp)
        location = dp
    elif isinstance(dp, dict):
        file_name = dp.get("file_name")  # type: ignore
        location = dp.get("location")  # type: ignore
        if location is None:
            location = dp.get("path")  # type: ignore
            location = os.path.join(location, file_name)  # type: ignore
    else:
        raise TypeError("datapoint not of expected type for converting to image")

    with MappingContextManager(dp_name=file_name) as mapping_context:
        dp_image = Image(file_name=file_name, location=location)  # type: ignore
        if file_name is not None:
            if is_file_extension(file_name, ".pdf") and isinstance(dp, dict):
                dp_image.pdf_bytes = dp.get("pdf_bytes")  # type: ignore
                if dp_image.pdf_bytes is not None:
                    if isinstance(dp_image.pdf_bytes, bytes):
                        dp_image.image = convert_pdf_bytes_to_np_array_v2(dp_image.pdf_bytes, dpi=dpi)
            else:
                dp_image.image = load_image_from_file(location)  # type: ignore

    if mapping_context.context_error:
        return None

    return dp_image


def maybe_load_image(dp: Image) -> Image:
    """
    If :attr:`image` is None will load the image.

    :param dp: An Image
    :return: Image with attr: image not None
    """

    if dp.image is None:
        assert dp.location is not None, "cannot load image, no location provided"
        loader = get_load_image_func(dp.location)
        dp.image = loader(dp.location)  # type: ignore
    return dp


def maybe_remove_image(dp: Image) -> Image:
    """
    Remove :attr:`image` if a location is provided.

    :param dp: An Image
    :return: Image with None attr: image
    """

    if dp.location is not None:
        dp.clear_image()
    return dp


def image_ann_to_image(dp: Image, category_names: Union[str, List[str]], crop_image: bool = True) -> Image:
    """
    Adds attr: image to annotations with given category names

    :param dp: Image
    :param category_names: A single or a list of category names
    :param crop_image: Will add numpy array to :attr:`image.image`
    :return: Image
    """

    img_anns = dp.get_annotation_iter(category_names=category_names)
    for ann in img_anns:
        dp.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=crop_image)

    return dp


@cur  # type: ignore
def maybe_ann_to_sub_image(
    dp: Image, category_names_sub_image: Union[str, List[str]], category_names: Union[str, List[str]]
) -> Image:
    """
    Assigns to sub image with given category names all annotations with given category names whose bounding box lie
    within the bounding box of the sub image.

    :param dp: Image
    :param category_names_sub_image: A single or a list of category names that will form a sub image.
    :param category_names: A single or a list of category names that will may be assigned to a sub image, conditioned
                           on the bounding box lying within the sub image.
    :return: Image
    """

    anns = dp.get_annotation(category_names=category_names_sub_image)
    for ann in anns:
        dp.maybe_ann_to_sub_image(annotation_id=ann.annotation_id, category_names=category_names)

    return dp


@cur  # type: ignore
def xml_to_dict(dp: JsonDict, xslt_obj: etree.XSLT) -> JsonDict:
    """
    Convert a xml object into a dict using a xsl style sheet.

    **Example:**

        .. code-block:: python

           with open(path_xslt) as xsl_file:
               xslt_file = xsl_file.read().encode('utf-8')
           xml_obj = etree.XML(xslt_file, parser=etree.XMLParser(encoding='utf-8'))
           xslt_obj = etree.XSLT(xml_obj)
           df = MapData(df, xml_to_dict(xslt_obj))

    :param dp: string representing the xml
    :param xslt_obj: xslt object to parse the string
    :return: parsed xml
    """

    output = str(xslt_obj(dp["xml"]))
    output = ast.literal_eval(output.replace('<?xml version="1.0"?>', ""))
    dp.pop("xml")
    dp["json"] = output
    return dp
