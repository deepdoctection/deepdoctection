# -*- coding: utf-8 -*-
# File: xxx.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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

import json
from pathlib import Path

import numpy as np
from pytest import fixture

import shared_test_utils as stu
from dd_core.dataflow.custom_serialize import SerializerCoco
from dd_core.datapoint.annotation import ImageAnnotation
from dd_core.datapoint.image import Image
from dd_core.mapper.cats import filter_cat
from dd_core.mapper.xfundstruct import xfund_to_image

from ..conftest import XFundSample
from .data import IIITAR13K_DATAPOINT, PRODIGY_DATAPOINT, XFUND_LAYOUTLM_FEATURES, XFUND_RAW_LAYOUTLM_FEATURES


@fixture(name="image")
def fixture_page(page_json_path: Path) -> Image:
    """Provide a Page instance loaded from page_json fixture."""
    return Image.from_file(file_path=str(page_json_path))


@fixture(name="image_with_layout_anns")
def fixture_image_with_layout_anns(image: Image) -> Image:
    """An image with layout annotations"""
    categories_as_list_filtered = [
        "text",
        "page_footer",
        "caption",
        "table",
        "figure",
        "section_header",
        "page_header",
        "column_header",
    ]
    categories_as_list_unfiltered = [
        "text",
        "page_footer",
        "caption",
        "cell",
        "table",
        "line",
        "column",
        "word",
        "row",
        "figure",
        "section_header",
        "page_header",
        "column_header",
    ]
    image = filter_cat(
        categories_as_list_filtered=categories_as_list_filtered,
        categories_as_list_unfiltered=categories_as_list_unfiltered,
    )(image)
    return image


@fixture(name="table_image")
def fixture_table_image(image: Image) -> Image:
    """An image from a table image annotation crop"""
    return image.get_annotation(category_names="table")[0].image


@fixture(name="coco_datapoint")
def fixture_coco_datapoint() -> dict:
    """Provide a COCO datapoint dict loaded from coco_datapoint_json fixture."""
    path = stu.asset_path("coco_like")
    df = SerializerCoco.load(path=path)
    df.reset_state()
    return next(iter(df))


@fixture(name="xfund_datapoint")
def fixture_xfund_datapoint() -> dict:
    xfund_dict = XFundSample().data
    return xfund_dict


@fixture(name="xfund_image")
def fixture_xfund_image(xfund_datapoint: dict, monkeypatch) -> Image:

    def _fake_loader(_path: str):
        return np.zeros((3508, 2480, 3), dtype=np.uint8)

    # Patch the function used inside xfund_to_image
    monkeypatch.setattr("dd_core.mapper.xfundstruct.load_image_from_file", _fake_loader)

    categories_dict = {"word": 1, "text": 2}
    token_class_names_mapping = {
        "other": "other",
        "header": "header",
    }
    ner_token_to_id_mapping = {
        "word": {
            "token_class": {"other": 1, "question": 2, "answer": 3, "header": 4},
            "tag": {"I": 1, "O": 2, "B": 3},
            "token_tag": {
                "B-answer": 1,
                "B-header": 2,
                "B-question": 3,
                "I-answer": 4,
                "I-header": 5,
                "I-question": 6,
                "O": 7,
            },
        },
        "text": {"token_class": {"other": 1, "question": 2, "answer": 3, "header": 4}},
    }

    img: Image = xfund_to_image(
        load_image=False,
        fake_score=False,
        categories_dict_name_as_key=categories_dict,
        token_class_names_mapping=token_class_names_mapping,
        ner_token_to_id_mapping=ner_token_to_id_mapping,
    )(xfund_datapoint)
    return img


@fixture(name="xfund_raw_layoutlm_features")
def fixture_xfund_raw_layoutlm_features() -> dict:
    return XFUND_RAW_LAYOUTLM_FEATURES


@fixture(name="layoutlm_features")
def fixture_layoutlm_features() -> dict:
    return XFUND_LAYOUTLM_FEATURES[0]


@fixture(name="prodigy_datapoint")
def fixture_prodigy_datapoint() -> dict:
    return PRODIGY_DATAPOINT


@fixture(name="pubtabnet_datapoint")
def fixture_pubtabnet_datapoint() -> dict:
    path = stu.asset_path("pubtabnet_like")
    with open(path, "r") as f:
        pubtabnet_dict = json.load(f)
    return pubtabnet_dict


@fixture(name="iiitar13k_datapoint")
def fixture_iiitar13k_datapoint() -> dict:
    return IIITAR13K_DATAPOINT


@fixture(name="dp_image")
def fixture_dp_image() -> Image:
    """fixture Image datapoint"""
    img = Image(location="/test/to/path", file_name="test_name")
    img.image = np.ones([400, 600, 3], dtype=np.float32)
    return img


@fixture(name="annotations")
def fixture_annotations_dict(dp_image: Image):
    path = stu.asset_path("annotations")
    with open(path, "r") as f:
        annotations_dict = json.load(f)

    def make_annotation(use_layout: bool, use_captions: bool):

        layout_anns = [ImageAnnotation(**data) for data in annotations_dict["layout_anns"]]
        captions = [ImageAnnotation(**data) for data in annotations_dict["caption_anns"]]

        if use_layout:
            for img_ann in layout_anns:
                dp_image.dump(img_ann)
                dp_image.image_ann_to_image(img_ann.annotation_id, True)
        if use_captions:
            for cap_ann in captions:
                dp_image.dump(cap_ann)
                dp_image.image_ann_to_image(cap_ann.annotation_id, True)

        return dp_image

    return make_annotation
