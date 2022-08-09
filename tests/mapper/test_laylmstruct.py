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


from copy import copy
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
from pytest import mark

from deepdoctection.mapper.laylmstruct import image_to_layoutlm, image_to_raw_layoutlm_features
from deepdoctection.mapper.xfundstruct import xfund_to_image
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.settings import names


@mark.requires_pt
@patch("deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(return_value=np.ones((1000, 1000, 3))))
def test_image_to_layoutlm(
    datapoint_xfund: JsonDict, xfund_category_names: Dict[str, str], layoutlm_input: JsonDict
) -> None:
    """
    testing image_to_layoutlm is mapping correctly
    """

    # Arrange
    xfund_to_image_func = xfund_to_image(True, False, xfund_category_names)  # pylint: disable=E1120
    image = xfund_to_image_func(datapoint_xfund)
    tokenizer_output = {
        "input_ids": layoutlm_input["input_ids"],
        "attention_mask": layoutlm_input["attention_mask"],
        "token_type_ids": layoutlm_input["token_type_ids"],
    }

    tokenizer = MagicMock(return_value=tokenizer_output)
    word_output = copy(layoutlm_input["tokens"][0]) # tokens are now batched
    word_output.pop(0)
    word_output.pop(-1)
    word_output = [word_output[0:6], word_output[6:13], word_output[13:16]]
    tokenizer.tokenize = MagicMock(side_effect=word_output)
    tokenizer.max_model_input_sizes={"microsoft/layoutlm-base-uncased": 512}

    # Act
    output = image_to_layoutlm(tokenizer=tokenizer)(image)  # pylint: disable=E1102

    # Assert
    assert len(output["ids"]) == 18
    assert output["boxes"].shape == (1, 18, 4)
    assert output["tokens"] == layoutlm_input["tokens"][0]
    assert output["input_ids"] == layoutlm_input["input_ids"]
    assert output["attention_mask"] == layoutlm_input["attention_mask"]
    assert output["token_type_ids"] == layoutlm_input["token_type_ids"]


@patch("deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(return_value=np.ones((1000, 1000, 3))))
def test_image_to_raw_layoutlm_features_for_token_data(datapoint_xfund: JsonDict,
                                                       xfund_category_names: Dict[str, str],
                                                       xfund_categories_dict_name_as_key: Dict[str,str],
                                                       raw_layoutlm_features: JsonDict):
    """
    testing image_to_raw_layoutlm_features is mapping correctly for dataset type "TOKEN_CLASSIFICATION"
    """

    # Arrange
    image = xfund_to_image(True, False, xfund_category_names)(datapoint_xfund)

    # Act
    raw_features = image_to_raw_layoutlm_features(xfund_categories_dict_name_as_key,names.DS.TYPE.TOK)(image)

    # Assert
    assert raw_features["image_id"] == raw_layoutlm_features["image_id"]
    assert raw_features["width"] == raw_layoutlm_features["width"]
    assert raw_features["height"] == raw_layoutlm_features["height"]
    assert raw_features["ann_ids"] == raw_layoutlm_features["ann_ids"]
    assert raw_features["words"] == raw_layoutlm_features["words"]
    assert raw_features["bbox"] == raw_layoutlm_features["bbox"]
    assert raw_features["dataset_type"] == raw_layoutlm_features["dataset_type"]
    assert raw_features["labels"] == raw_layoutlm_features["labels"]


@patch("deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(return_value=np.ones((1000, 1000, 3))))
def test_image_to_raw_layoutlm_features_for_inference(datapoint_xfund: JsonDict,
                                                       xfund_category_names: Dict[str, str],
                                                       raw_layoutlm_features: JsonDict):
    """
    testing image_to_raw_layoutlm_features is mapping correctly. Semantic entities and tags have been removed, so that
    the scenario covers the inference case
    """

    # Arrange
    image = xfund_to_image(True, False, xfund_category_names)(datapoint_xfund)

    for ann in image.get_annotation():
        ann.remove_sub_category(names.C.SE)
        ann.remove_sub_category(names.NER.TAG)

    # Act
    raw_features = image_to_raw_layoutlm_features(None,names.DS.TYPE.TOK)(image)

    # Assert
    assert raw_features["image_id"] == raw_layoutlm_features["image_id"]
    assert raw_features["width"] == raw_layoutlm_features["width"]
    assert raw_features["height"] == raw_layoutlm_features["height"]
    assert raw_features["ann_ids"] == raw_layoutlm_features["ann_ids"]
    assert raw_features["words"] == raw_layoutlm_features["words"]
    assert raw_features["bbox"] == raw_layoutlm_features["bbox"]
    assert raw_features["dataset_type"] == raw_layoutlm_features["dataset_type"]
    assert "labels" not in raw_features


