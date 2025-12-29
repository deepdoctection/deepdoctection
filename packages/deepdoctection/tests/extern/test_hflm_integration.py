# -*- coding: utf-8 -*-
# File: test_hflm_integration.py

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
Integration tests for language models using Hugging Face's transformers library.

This module contains integration tests for various language model capabilities,
such as sequence classification, token classification, and language detection,
utilizing Hugging Face's `transformers` library. The tests ensure that the models
are properly instantiated, configured, and capable of making predictions.

"""

from __future__ import annotations

import os

import pytest

from dd_core.utils.file_utils import pytorch_available, transformers_available
from dd_core.utils.types import PathLikeOrStr
from deepdoctection.extern.hflm import (
    HFLmLanguageDetector,
    HFLmSequenceClassifier,
    HFLmTokenClassifier,
)
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager

if pytorch_available() and transformers_available():
    import torch
    from transformers import (
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
    )

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available()),
    reason="Requires PyTorch and transformers installed",
)


class DummyTokenizer:
    """
    A dummy tokenizer for tokenizing input text.

    This class provides a callable interface to tokenize text data, emulating a basic
    tokenization process similar to popular tokenizer implementations. It is primarily
    used for testing or demonstration purposes and does not involve actual tokenization
    logic.

    """

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=512):  # type: ignore
        import torch  # pylint:disable=W0621,C0415

        return {
            "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


def _dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@REQUIRES_PT_AND_TR
@pytest.mark.slow
def test_hflm_sequence_slow_build_and_predict(tmp_path: PathLikeOrStr) -> None:
    """Test sequence classification using a tiny model."""
    cfg = XLMRobertaConfig(num_labels=2)  # pylint:disable=E0606
    model = XLMRobertaForSequenceClassification(cfg)  # pylint:disable=E0606
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "invoice", 2: "financial_report"}
    clf = HFLmSequenceClassifier(
        path_config_json=os.fspath(tmp_path),
        path_weights=os.fspath(tmp_path),
        categories=categories,
        device="cpu",
    )

    L = 5
    inputs = {
        "input_ids": torch.randint(50, (1, L), dtype=torch.long),  # pylint:disable=E0606
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "token_type_ids": torch.zeros((1, L), dtype=torch.long),
    }
    res = clf.predict(**inputs)
    assert res.class_name in categories.values()


@REQUIRES_PT_AND_TR
@pytest.mark.slow
def test_hflm_token_slow_build_and_predict(tmp_path: PathLikeOrStr) -> None:
    """Test token classification using a tiny model."""
    cfg = XLMRobertaConfig(num_labels=3)
    model = XLMRobertaForTokenClassification(cfg)  # pylint:disable=E0606
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "B-header", 2: "I-header", 3: "O"}
    clf = HFLmTokenClassifier(
        path_config_json=os.fspath(tmp_path),
        path_weights=os.fspath(tmp_path),
        categories=categories,
        device="cpu",
    )

    L = 4
    inputs = {
        "ann_ids": [[f"u{i}" for i in range(L)]],
        "image_ids": ["img-1"],
        "tokens": [[f"t{i}" for i in range(L)]],
        "input_ids": torch.randint(100, (1, L), dtype=torch.long),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "token_type_ids": torch.zeros((1, L), dtype=torch.long),
    }
    results = clf.predict(**inputs)
    assert len(results) == L
    assert all(r.class_name in categories.values() for r in results)


@REQUIRES_PT_AND_TR
@pytest.mark.slow
def test_hflm_language_slow_build_and_predict() -> None:
    """Test language detection using a tiny model."""
    # Use xlm-roberta model  as example
    weights = "papluca/xlm-roberta-base-language-detection/model.safetensors"
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    config_path = ModelCatalog.get_full_path_configs(weights)
    categories = ModelCatalog.get_profile(weights).categories
    config_dir = ModelCatalog.get_full_path_configs_dir(weights)
    assert categories is not None

    det = HFLmLanguageDetector(
        path_config_json=config_path,
        path_weights=weights_path,
        categories=categories,
        device="cpu",
        tokenizer_config_dir=config_dir,
    )

    res = det.predict("Sample text for language detection.")
    assert res.class_name in categories.values()
    assert 0.0 <= res.score <= 1.0  # type: ignore
