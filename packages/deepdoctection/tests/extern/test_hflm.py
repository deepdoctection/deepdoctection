# -*- coding: utf-8 -*-
# File: test_hflm.py

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

from typing import Any
from unittest.mock import MagicMock

import pytest

from dd_core.utils.file_utils import pytorch_available, transformers_available
from deepdoctection.extern.base import SequenceClassResult, TokenClassResult
from deepdoctection.extern.hflm import (
    HFLmLanguageDetector,
    HFLmSequenceClassifier,
    HFLmTokenClassifier,
)

if pytorch_available():
    import torch

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available()),
    reason="Requires PyTorch and transformers installed",
)


def _mk_dummy_tokenizer() -> Any:
    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=512):
            # Minimal encoding dict the model expects
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                # No token_type_ids for XLM-R; model code will create zeros_like if absent
            }

    return DummyTokenizer()


@REQUIRES_PT_AND_TR
def test_hflm_sequence_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid network tokenizer download during ctor
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    # Mock model construction (no real weights/model)
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.HFLmSequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    # Mock prediction helper
    def _fake_seq_predict(input_ids, attention_mask, token_type_ids, model):
        return SequenceClassResult(class_id=0, score=0.92)

    monkeypatch.setattr(
        "deepdoctection.extern.hflm.predict_sequence_classes_from_lm",
        MagicMock(side_effect=_fake_seq_predict),
        raising=True,
    )

    categories = {1: "letter", 2: "invoice"}
    clf = HFLmSequenceClassifier("path/to/config.json", "path/to/weights", categories, device="cpu")

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
    }
    result = clf.predict(**inputs)
    assert result.class_id == 1
    assert result.class_name == "letter"
    assert result.score > 0.9


@REQUIRES_PT_AND_TR
def test_hflm_token_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(
        "deepdoctection.extern.hflm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.HFLmTokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_tok_predict(uuids, input_ids, attention_mask, token_type_ids, tokens, model):
        return [
            TokenClassResult(uuid="u1", token_id=101, class_id=2, token="A", score=0.8),  # -> class_id+1 = 3
            TokenClassResult(uuid="u2", token_id=102, class_id=0, token="B", score=0.9),  # -> class_id+1 = 1
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.hflm.predict_token_classes_from_lm",
        MagicMock(side_effect=_fake_tok_predict),
        raising=True,
    )

    categories = {1: "B-header", 2: "I-header", 3: "O"}
    clf = HFLmTokenClassifier(
        "path/to/config.json",
        "path/to/weights",
        categories=categories,
        device="cpu",
        use_xlm_tokenizer=True,
    )

    inputs = {
        "ann_ids": [["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"]],
        "image_ids": ["e2c7b6a4-5f3b-8c2d-a1b2-9f0e4d3c2b1a"],
        "tokens": [["A", "B"]],
        "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
    }
    results = clf.predict(**inputs)
    assert len(results) == 2
    names = sorted(r.class_name for r in results)
    assert names == ["B-header", "O"]
    assert all(r.class_id in categories for r in results)


def _mk_dummy_fast_tokenizer() -> Any:
    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=512):
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                # No token_type_ids for XLM-R; code will create zeros_like if absent
            }

    return DummyTokenizer()


@REQUIRES_PT_AND_TR
def test_hflm_language_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid network tokenizer download during ctor
    monkeypatch.setattr(
        "transformers.XLMRobertaTokenizerFast.from_pretrained",
        lambda *args, **kwargs: _mk_dummy_fast_tokenizer(),
        raising=True,
    )

    # Mock model construction (no real weights/model)
    class _StubLangModel:
        def to(self, device):
            return self

        def eval(self):
            pass

        def __call__(self, input_ids=None, attention_mask=None, token_type_ids=None):
            # Highest score at index 1 -> class_id becomes 2 -> "deu"
            return type("Out", (), {"logits": torch.tensor([[0.1, 2.0, 0.5]], dtype=torch.float32)})

    monkeypatch.setattr(
        "deepdoctection.extern.hflm.HFLmLanguageDetector.get_wrapped_model",
        lambda *args, **kwargs: _StubLangModel(),
        raising=True,
    )

    categories = {1: "eng", 2: "deu", 3: "fre"}
    det = HFLmLanguageDetector("path/to/config.json", "path/to/weights", categories, device="cpu")

    res = det.predict("Hello world")
    assert res.class_name == "deu"


@REQUIRES_PT_AND_TR
def test_hflm_sequence_validate_encodings_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid tokenizer resolution during ctor (model class is MagicMock)
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    # Mock model construction
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.HFLmSequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    categories = {1: "invoice"}
    clf = HFLmSequenceClassifier("path/to/config.json", "path/to/weights", categories, device="cpu")

    # Trigger validation error: input_ids is not a torch.Tensor
    with pytest.raises(ValueError):
        clf.predict(
            input_ids=[[1, 2]],  # wrong type
            attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            token_type_ids=torch.tensor([[0, 0]], dtype=torch.long),
        )


@REQUIRES_PT_AND_TR
def test_hflm_token_validate_encodings_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid tokenizer resolution during ctor (model class is MagicMock)
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    # Mock model construction
    monkeypatch.setattr(
        "deepdoctection.extern.hflm.HFLmTokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    clf = HFLmTokenClassifier(
        "path/to/config.json",
        "path/to/weights",
        categories={1: "B-header", 2: "I-header", 3: "O"},
        device="cpu",
    )

    with pytest.raises(ValueError):
        clf.predict(
            ann_ids=[["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"]],
            tokens=[["A", "B"]],
            input_ids="not-a-tensor",
            attention_mask="not-a-tensor",
            token_type_ids="not-a-tensor",
        )
