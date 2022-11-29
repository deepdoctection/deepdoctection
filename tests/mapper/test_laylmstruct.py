# -*- coding: utf-8 -*-
# File: test_laylmstruct.py

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
Testing the module mapper.laylmstruct
"""


from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
from pytest import mark

from deepdoctection.datapoint import Image
from deepdoctection.mapper.laylmstruct import image_to_raw_layoutlm_features, raw_features_to_layoutlm_features
from deepdoctection.mapper.xfundstruct import xfund_to_image
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.file_utils import transformers_available
from deepdoctection.utils.settings import DatasetType, ObjectTypes, WordType

if transformers_available():
    from transformers import LayoutLMTokenizerFast


@mark.basic
@patch("deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(return_value=np.ones((1000, 1000, 3))))
def test_image_to_raw_layoutlm_features_for_token_data(
    datapoint_xfund: JsonDict,
    xfund_category_names: Dict[str, str],
    xfund_category_dict: Dict[ObjectTypes, str],
    raw_layoutlm_features: JsonDict,
    ner_token_to_id_mapping: JsonDict,
) -> None:
    """
    testing image_to_raw_layoutlm_features is mapping correctly for dataset type "TOKEN_CLASSIFICATION"
    """

    # Arrange
    image = xfund_to_image(True, False, xfund_category_dict, xfund_category_names, ner_token_to_id_mapping)(
        datapoint_xfund
    )

    # Act
    raw_features = image_to_raw_layoutlm_features(DatasetType.token_classification)(image)

    # Assert
    assert raw_features is not None
    assert raw_features["image_id"] == raw_layoutlm_features["image_id"]
    assert raw_features["width"] == raw_layoutlm_features["width"]
    assert raw_features["height"] == raw_layoutlm_features["height"]
    assert raw_features["ann_ids"] == raw_layoutlm_features["ann_ids"]
    assert raw_features["words"] == raw_layoutlm_features["words"]
    assert raw_features["bbox"] == raw_layoutlm_features["bbox"]
    assert raw_features["dataset_type"] == raw_layoutlm_features["dataset_type"]
    assert raw_features["labels"] == raw_layoutlm_features["labels"]


@mark.basic
@patch("deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(return_value=np.ones((1000, 1000, 3))))
def test_image_to_raw_layoutlm_features_for_inference(
    datapoint_xfund: JsonDict,
    xfund_category_names: Dict[str, str],
    xfund_category_dict: Dict[ObjectTypes, str],
    raw_layoutlm_features: JsonDict,
    ner_token_to_id_mapping: JsonDict,
) -> None:
    """
    testing image_to_raw_layoutlm_features is mapping correctly. Semantic entities and tags have been removed, so that
    the scenario covers the inference case
    """

    # Arrange
    image = xfund_to_image(True, False, xfund_category_dict, xfund_category_names, ner_token_to_id_mapping)(
        datapoint_xfund
    )

    assert image is not None

    for ann in image.get_annotation():
        ann.remove_sub_category(WordType.token_class)
        ann.remove_sub_category(WordType.tag)

    # Act
    raw_features = image_to_raw_layoutlm_features(DatasetType.token_classification)(image)

    # Assert
    assert raw_features is not None
    assert raw_features["image_id"] == raw_layoutlm_features["image_id"]
    assert raw_features["width"] == raw_layoutlm_features["width"]
    assert raw_features["height"] == raw_layoutlm_features["height"]
    assert raw_features["ann_ids"] == raw_layoutlm_features["ann_ids"]
    assert raw_features["words"] == raw_layoutlm_features["words"]
    assert raw_features["bbox"] == raw_layoutlm_features["bbox"]
    assert raw_features["dataset_type"] == raw_layoutlm_features["dataset_type"]
    assert "labels" not in raw_features


@mark.requires_pt
def test_raw_features_to_layoutlm_features(
    dp_image_with_layout_and_word_annotations: Image, layoutlm_features: JsonDict
) -> None:
    """
    testing image_to_layoutlm_features is mapping correctly
    """

    # Arrange
    dp = image_to_raw_layoutlm_features()(dp_image_with_layout_and_word_annotations)  # pylint: disable=E1120
    tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

    # Act
    assert dp is not None
    feature = raw_features_to_layoutlm_features(dp, tokenizer_fast, padding="do_not_pad")

    # Assert
    # pylint: disable=E1136
    assert feature["image_ids"] == layoutlm_features["image_ids"]
    assert feature["width"] == layoutlm_features["width"]
    assert feature["height"] == layoutlm_features["height"]
    assert feature["ann_ids"] == layoutlm_features["ann_ids"]
    assert feature["input_ids"] == layoutlm_features["input_ids"]
    assert feature["token_type_ids"] == layoutlm_features["token_type_ids"]
    assert feature["attention_mask"] == layoutlm_features["attention_mask"]
    assert feature["bbox"] == layoutlm_features["bbox"]
    assert feature["tokens"] == layoutlm_features["tokens"]
    # pylint: enable=E1136
