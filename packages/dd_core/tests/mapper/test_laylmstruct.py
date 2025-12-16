# -*- coding: utf-8 -*-
# File: test_laylmstruct.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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

import numpy as np
import pytest

from dd_core.datapoint import Image
from dd_core.mapper.laylmstruct import image_to_raw_layoutlm_features, raw_features_to_layoutlm_features
from dd_core.utils.file_utils import transformers_available
from dd_core.utils.object_types import DatasetType, WordType
from dd_core.utils.types import JsonDict

if transformers_available():
    from transformers import LayoutLMTokenizerFast


def test_image_to_raw_layoutlm_features_for_token_data(
    monkeypatch: pytest.MonkeyPatch,
    xfund_image: Image,
    xfund_raw_layoutlm_features: JsonDict,
) -> None:
    """
    testing image_to_raw_layoutlm_features is mapping correctly for dataset type "TOKEN_CLASSIFICATION"
    using xfund_image fixture and xfund_raw_layoutlm_features as expected.
    """

    monkeypatch.setattr(
        "dd_core.mapper.xfundstruct.load_image_from_file",
        lambda fn: np.zeros((3508, 2480, 3), dtype=np.uint8),
    )

    # Act
    raw_features = image_to_raw_layoutlm_features(DatasetType.TOKEN_CLASSIFICATION)(xfund_image)

    # Assert
    assert raw_features is not None
    assert raw_features["image_id"] == xfund_raw_layoutlm_features["image_id"]
    assert raw_features["width"] == xfund_raw_layoutlm_features["width"]
    assert raw_features["height"] == xfund_raw_layoutlm_features["height"]
    assert raw_features["ann_ids"] == xfund_raw_layoutlm_features["ann_ids"]
    assert raw_features["words"] == xfund_raw_layoutlm_features["words"]
    assert raw_features["bbox"] == xfund_raw_layoutlm_features["bbox"]
    assert raw_features["dataset_type"] == xfund_raw_layoutlm_features["dataset_type"]
    assert raw_features["labels"] == xfund_raw_layoutlm_features["labels"]


def test_image_to_raw_layoutlm_features_for_inference(
    monkeypatch: pytest.MonkeyPatch,
    xfund_image: Image,
    xfund_raw_layoutlm_features: JsonDict,
) -> None:
    """
    testing image_to_raw_layoutlm_features mapping when semantic entities and tags are removed
    (inference case), using xfund_image fixture.
    """
    # Mock image loader as in test_xfundstruct
    monkeypatch.setattr(
        "dd_core.mapper.xfundstruct.load_image_from_file",
        lambda fn: np.zeros((3508, 2480, 3), dtype=np.uint8),
    )

    # Arrange
    image = xfund_image
    for ann in image.get_annotation():
        ann.remove_sub_category(WordType.TOKEN_CLASS)
        ann.remove_sub_category(WordType.TAG)
        ann.remove_sub_category(WordType.TOKEN_TAG)

    # Act
    raw_features = image_to_raw_layoutlm_features(DatasetType.TOKEN_CLASSIFICATION)(image)

    # Assert
    assert raw_features is not None
    assert raw_features["image_id"] == xfund_raw_layoutlm_features["image_id"]
    assert raw_features["width"] == xfund_raw_layoutlm_features["width"]
    assert raw_features["height"] == xfund_raw_layoutlm_features["height"]
    assert raw_features["ann_ids"] == xfund_raw_layoutlm_features["ann_ids"]
    assert raw_features["words"] == xfund_raw_layoutlm_features["words"]
    assert raw_features["bbox"] == xfund_raw_layoutlm_features["bbox"]
    assert raw_features["dataset_type"] == xfund_raw_layoutlm_features["dataset_type"]
    assert "labels" not in raw_features


@pytest.mark.skipif(not transformers_available(), reason="Transformers is not installed")
def test_raw_features_to_layoutlm_features(xfund_image: Image, layoutlm_features: JsonDict) -> None:
    """
    testing image_to_layoutlm_features is mapping correctly using xfund_image as input
    and layoutlm_features as expected output.
    """
    # Arrange
    dp = image_to_raw_layoutlm_features()(xfund_image)  # pylint: disable=E1120
    tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")  # pylint: disable=E0606

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
