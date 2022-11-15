# -*- coding: utf-8 -*-
# File: test_hflayoutlm.py

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
Testing module extern.hflayoutlm
"""
from typing import List
from unittest.mock import MagicMock, patch

from pytest import mark, raises

from deepdoctection.extern.base import SequenceClassResult, TokenClassResult
from deepdoctection.extern.hflayoutlm import HFLayoutLmSequenceClassifier, HFLayoutLmTokenClassifier
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.file_utils import pytorch_available
from deepdoctection.utils.settings import BioTag, TokenClasses, get_type

from ..mapper.data import DatapointXfund

if pytorch_available():
    import torch


def get_token_class_results(  # type: ignore
    uuids: List[str], input_ids, attention_mask, token_type_ids, boxes, tokens, model, images  # pylint: disable=W0613
) -> List[TokenClassResult]:
    """
    token class result list
    """
    return DatapointXfund().get_token_class_results()


def get_sequence_class_result(  # type: ignore
    input_ids, attention_mask, token_type_ids, boxes, model  # pylint: disable=W0613
) -> SequenceClassResult:
    """
    sequence class result
    """
    return DatapointXfund().get_sequence_class_results()


class TestHFLayoutLmTokenClassifier:
    """
    Test HFLayoutLmTokenClassifier
    """

    @staticmethod
    @mark.requires_pt
    @patch(
        "deepdoctection.extern.hflayoutlm.get_pytorch_requirement", MagicMock(return_value=("torch", False, "DUMMY"))
    )
    @patch("deepdoctection.extern.hflayoutlm.PretrainedConfig.from_pretrained", MagicMock())
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForTokenClassification.from_pretrained", MagicMock())
    def test_hf_layout_lm_does_not_build_when_pt_not_available() -> None:
        """
        HFLayoutLmTokenClassifier needs pytorch. Construction fails, when requirement is not satisfied
        """

        # Arrange, Act & Assert
        with raises(ImportError):
            HFLayoutLmTokenClassifier("path/to/json", "path/to/model", ["foo"], ["B", "I", "O"])

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.hflayoutlm.PretrainedConfig.from_pretrained", MagicMock())
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForTokenClassification.from_pretrained", MagicMock())
    def test_categories_are_constructed_properly() -> None:
        """
        HFLayoutLmTokenClassifier creates a full category set depending on semantics, tagging or by passing the
        set of categories directly
        """

        # Arrange, Act & Assert
        with raises(ValueError):
            HFLayoutLmTokenClassifier("path/to/json", "path/to/model", ["foo"], None)

        # Arrange
        categories_semantics = [TokenClasses.header]
        categories_bio = [BioTag.begin, BioTag.inside, BioTag.outside]

        # Act
        model = HFLayoutLmTokenClassifier("path/to/json", "path/to/model", categories_semantics, categories_bio)

        # Assert
        assert set(model.categories.values()) == {BioTag.outside, get_type("B-header"), get_type("I-header")}

        # Arrange
        categories_explicit = {"1": get_type("B-header"), "2": get_type("I-header"), "3": get_type("O")}

        # Act
        model = HFLayoutLmTokenClassifier("path/to/json", "path/to/model", categories=categories_explicit)

        # Assert
        assert model.categories == categories_explicit

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForTokenClassification.from_pretrained", MagicMock())
    @patch("deepdoctection.extern.hflayoutlm.PretrainedConfig.from_pretrained", MagicMock())
    @patch("deepdoctection.extern.hflayoutlm.predict_token_classes", MagicMock(side_effect=get_token_class_results))
    def test_hf_layout_lm_predicts_token(
        layoutlm_input: JsonDict,
        categories_semantics: List[str],
        categories_bio: List[str],
        token_class_names: List[str],
    ) -> None:
        """
        HFLayoutLmTokenClassifier calls predict_token_classes and post processes TokenClassResult correctly
        """

        # Arrange
        categories_semantics = [TokenClasses.header]
        categories_bio = [BioTag.begin, BioTag.inside, BioTag.outside]
        layoutlm = HFLayoutLmTokenClassifier("path/to/json", "path/to/model", categories_semantics, categories_bio)
        layoutlm.model.device = "cpu"

        # Act
        inputs = {
            "image_ids": layoutlm_input["image_ids"],
            "width": layoutlm_input["width"],
            "height": layoutlm_input["height"],
            "ann_ids": layoutlm_input["ann_ids"],
            "tokens": layoutlm_input["tokens"],
            "bbox": torch.tensor(layoutlm_input["bbox"]),
            "input_ids": torch.tensor(layoutlm_input["input_ids"]),
            "attention_mask": torch.tensor(layoutlm_input["attention_mask"]),
            "token_type_ids": torch.tensor(layoutlm_input["token_type_ids"]),
        }

        results = layoutlm.predict(**inputs)

        # Assert
        assert len(results) == 18
        class_names = [res.class_name for res in results]
        assert class_names == token_class_names


class TestHFLayoutLmSequenceClassifier:
    """
    Test HFLayoutLmSequenceClassifier
    """

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForSequenceClassification.from_pretrained", MagicMock())
    @patch("deepdoctection.extern.hflayoutlm.PretrainedConfig.from_pretrained", MagicMock())
    @patch(
        "deepdoctection.extern.hflayoutlm.predict_sequence_classes", MagicMock(side_effect=get_sequence_class_result)
    )
    def test_hf_layout_lm_predicts_token(
        layoutlm_input: JsonDict,
    ) -> None:
        """
        HFLayoutLmTokenClassifier calls predict_token_classes and post processes TokenClassResult correctly
        """

        # Arrange
        categories = {"1": get_type("FOO"), "2": get_type("BAK")}
        layoutlm = HFLayoutLmSequenceClassifier("path/to/json", "path/to/model", categories)
        layoutlm.model.device = "cpu"

        # Act
        inputs = {
            "image_ids": layoutlm_input["image_ids"],
            "width": layoutlm_input["width"],
            "height": layoutlm_input["height"],
            "ann_ids": layoutlm_input["ann_ids"],
            "tokens": layoutlm_input["tokens"],
            "bbox": torch.tensor(layoutlm_input["bbox"]),
            "input_ids": torch.tensor(layoutlm_input["input_ids"]),
            "attention_mask": torch.tensor(layoutlm_input["attention_mask"]),
            "token_type_ids": torch.tensor(layoutlm_input["token_type_ids"]),
        }

        results = layoutlm.predict(**inputs)

        # Assert
        assert results.class_name == "BAK"
