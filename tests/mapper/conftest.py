# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixtures for mapper package testing
"""
from typing import Any, Dict, Optional, Union

from pytest import fixture

from deepdoctection.datapoint import Image
from deepdoctection.utils.detection_types import ImageType, JsonDict

from .data import (
    DatapointCoco,
    DatapointImage,
    DatapointPageDict,
    DatapointProdigy,
    DatapointPubtabnet,
    DatapointXfund,
    IIITar13KJson,
)


@fixture(name="datapoint_coco")
def fixture_datapoint_coco() -> Dict[str, Any]:
    """
    Datapoint as received from SerializerCoco
    """

    return DatapointCoco().dp


@fixture(name="categories_coco")
def fixture_categories_coco() -> Dict[str, str]:
    """
    Categories as Dict
    """
    return DatapointCoco().categories


def get_coco_white_image(path: str, type_id: str = "np") -> Optional[Union[str, ImageType]]:
    """
    Returns a white image
    :param path: An image path
    :param type_id: "np" or "b64"
    :return:
    """
    return DatapointCoco().get_white_image(path, type_id)


@fixture(name="coco_results")
def fixture_coco_results() -> DatapointCoco:
    """
    DatapointCoco
    """
    return DatapointCoco()


@fixture(name="datapoint_pubtabnet")
def fixture_datapoint_pubtabnet() -> Dict[str, Any]:
    """
    Datapoint as received from SerializerCoco
    """

    return DatapointPubtabnet().dp


@fixture(name="categories_name_as_key_pubtabnet")
def fixture_categories_name_as_key_pubtabnet() -> Dict[str, str]:
    """
    Categories as Dict
    """
    return DatapointPubtabnet().categories_as_names


@fixture(name="pubtabnet_results")
def fixture_pubtabnet_results() -> DatapointPubtabnet:
    """
    DatapointPubtabnet
    """
    return DatapointPubtabnet()


def get_pubtabnet_white_image(path: str, type_id: str = "np") -> Optional[Union[str, ImageType]]:
    """
    Returns a white image
    :param path: An image path
    :param type_id: "np" or "b64"
    """

    if path == DatapointPubtabnet().dp["filename"]:
        return DatapointPubtabnet().get_white_image(path, type_id)
    return None


def get_always_pubtabnet_white_image(path: str, type_id: str = "np") -> Optional[Union[str, ImageType]]:
    """
    Returns a white image
    :param path: An image path
    :param type_id: "np" or "b64"
    """

    return DatapointPubtabnet().get_white_image(path, type_id)


def get_always_pubtabnet_white_image_from_bytes(
    pdf_bytes: str, dpi: Optional[int] = None
) -> Optional[Union[str, ImageType]]:
    """
    Returns a white image
    """
    if pdf_bytes and dpi is not None:
        pass
    return DatapointPubtabnet().get_white_image("", "np")


def get_always_bytes(path: str) -> bytes:
    """
    Returns bytes
    """
    if path:
        pass
    return b"\x04\x00"


@fixture(name="datapoint_prodigy")
def fixture_datapoint_prodigy() -> JsonDict:
    """
    Datapoint as received from Prodigy db
    """

    return DatapointProdigy().dp


@fixture(name="categories_prodigy")
def fixture_categories_prodigy() -> Dict[str, str]:
    """
    Categories as Dict
    """
    return DatapointProdigy().categories


def get_datapoint_prodigy() -> DatapointProdigy:
    """
    DatapointProdigy
    """
    return DatapointProdigy()


@fixture(name="prodigy_results")
def fixture_prodigy_results() -> DatapointProdigy:
    """
    DatapointProdigy
    """
    return DatapointProdigy()


@fixture(name="datapoint_image")
def fixture_datapoint_image() -> Image:
    """
    Image
    """
    return DatapointImage().image


@fixture(name="page_dict")
def fixture_page_dict() -> JsonDict:
    """
    page file
    """
    return DatapointPageDict().get_page_dict()


@fixture(name="datapoint_xfund")
def fixture_datapoint_xfund() -> Dict[str, Any]:
    """
    Datapoint as received from Xfund dataset
    """

    return DatapointXfund().dp  # type: ignore


@fixture(name="xfund_category_names")
def fixture_xfund_category_names() -> Dict[str, str]:
    """
    Xfund category names mapping
    """

    return DatapointXfund().get_category_names_mapping()


@fixture(name="layoutlm_input")
def fixture_layoutlm_input() -> JsonDict:
    """
    Layoutlm input
    """
    return DatapointXfund().get_layout_input()


@fixture(name="datapoint_iiitar13kjson")
def fixture_datapoint_iiitar13kjson() -> Dict[str, Any]:
    """
    Datapoint as received from iiitar13k dataset already converted into json format
    """

    return IIITar13KJson().dp


@fixture(name="iiitar13k_categories_name_as_keys")
def fixture_iiitar13k_categories_name_as_keys() -> Dict[str, str]:
    """
    iiitar13k category names dict
    """

    return IIITar13KJson().get_categories_name_as_keys()


@fixture(name="iiitar13k_category_names_mapping")
def fixture_xfund_category_names_mapping() -> Dict[str, str]:
    """
    iiitar13k category names mapping
    """

    return IIITar13KJson().get_category_names_mapping()


@fixture(name="iiitar13k_results")
def fixture_iiitar13k_results() -> IIITar13KJson:
    """
    iiitar13k results
    """
    return IIITar13KJson()
