# -*- coding: utf-8 -*-
# File: test_hflayoutlm.py

from typing import Any
from unittest.mock import MagicMock

import pytest

from dd_core.utils.file_utils import pytorch_available, transformers_available
from deepdoctection.extern.base import SequenceClassResult, TokenClassResult
from deepdoctection.extern.hflayoutlm import (
    HFLayoutLmSequenceClassifier,
    HFLayoutLmTokenClassifier,
    HFLayoutLmv2SequenceClassifier,
    HFLayoutLmv2TokenClassifier,
    HFLayoutLmv3SequenceClassifier,
    HFLayoutLmv3TokenClassifier,
    HFLiltSequenceClassifier,
    HFLiltTokenClassifier,
)

if pytorch_available():
    import torch

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available()),
    reason="Requires PyTorch and transformers installed",
)


def _mk_dummy_tokenizer() -> Any:
    class DummyTokenizer:
        pass

    return DummyTokenizer()


@REQUIRES_PT_AND_TR
def test_layoutlm_sequence_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid network tokenizer download during ctor
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    # Mock model construction (no real weights/model)
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmSequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    # Mock prediction helper
    def _fake_seq_predict(input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, model: Any, images: Any = None) -> SequenceClassResult:
        return SequenceClassResult(class_id=1, score=0.99)

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_sequence_classes_from_layoutlm",
        MagicMock(side_effect=_fake_seq_predict),
        raising=True,
    )

    categories = {1: "letter", 2: "invoice"}
    clf = HFLayoutLmSequenceClassifier("path/to/json", "path/to/model", categories, device="cpu")

    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
        "bbox": torch.zeros((1, 3, 4), dtype=torch.long),
    }

    result = clf.predict(**inputs)
    assert result.class_name == "invoice"


@REQUIRES_PT_AND_TR
def test_layoutlm_v2_sequence_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmv2SequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_seq_predict(input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, model: Any, images: Any = None) -> SequenceClassResult:
        return SequenceClassResult(class_id=1, score=0.95)

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_sequence_classes_from_layoutlm",
        MagicMock(side_effect=_fake_seq_predict),
        raising=True,
    )

    categories = {1: "letter", 2: "invoice"}
    clf = HFLayoutLmv2SequenceClassifier("path/to/json", "path/to/model", categories, device="cpu")

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
        "bbox": torch.zeros((1, 3, 4), dtype=torch.long),
        "image": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
    }

    result = clf.predict(**inputs)
    assert result.class_name == "invoice"


@REQUIRES_PT_AND_TR
def test_layoutlm_v3_sequence_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmv3SequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_seq_predict(input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, model: Any, images: Any = None) -> SequenceClassResult:
        return SequenceClassResult(class_id=1, score=0.93)

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_sequence_classes_from_layoutlm",
        MagicMock(side_effect=_fake_seq_predict),
        raising=True,
    )

    categories = {1: "letter", 2: "invoice"}
    clf = HFLayoutLmv3SequenceClassifier("path/to/json", "path/to/model", categories, device="cpu")

    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
        "bbox": torch.zeros((1, 2, 4), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
    }

    result = clf.predict(**inputs)
    assert result.class_name == "invoice"


@REQUIRES_PT_AND_TR
def test_layoutlm_token_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmTokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_tok_predict(uuids: Any, input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, tokens: Any, model: Any, images: Any = None) -> list[TokenClassResult]:
        return [
            TokenClassResult(uuid="a", token_id=101, class_id=2, token="X", score=0.9),  # -> id 3 -> "O"
            TokenClassResult(uuid="b", token_id=102, class_id=0, token="Y", score=0.8),  # -> id 1 -> "B-header"
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_token_classes_from_layoutlm",
        MagicMock(side_effect=_fake_tok_predict),
        raising=True,
    )

    categories = {1: "B-header", 2: "I-header", 3: "O"}
    clf = HFLayoutLmTokenClassifier("path/to/json", "path/to/model", categories=categories, device="cpu")

    inputs = {
        "ann_ids": [["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"]],
        "image_ids": ["e2c7b6a4-5f3b-8c2d-a1b2-9f0e4d3c2b1a"],
        "tokens": [["X", "Y"]],
        "bbox": torch.zeros((1, 2, 4), dtype=torch.long),
        "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
    }

    results = clf.predict(**inputs)
    assert len(results) == 2
    class_names = sorted([r.class_name for r in results])
    assert class_names == ["B-header", "O"]


@REQUIRES_PT_AND_TR
def test_layoutlm_v2_token_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmv2TokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_tok_predict(uuids: Any, input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, tokens: Any, model: Any, images: Any = None) -> list[TokenClassResult]:
        return [
            TokenClassResult(uuid="a", token_id=11, class_id=1, token="X", score=0.7),
            TokenClassResult(uuid="b", token_id=12, class_id=2, token="Y", score=0.6),
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_token_classes_from_layoutlm",
        MagicMock(side_effect=_fake_tok_predict),
        raising=True,
    )

    categories = {1: "B-header", 2: "I-header", 3: "O"}
    clf = HFLayoutLmv2TokenClassifier("path/to/json", "path/to/model", categories=categories, device="cpu")

    inputs = {
        "ann_ids": [["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"]],
        "image_ids": ["e2c7b6a4-5f3b-8c2d-a1b2-9f0e4d3c2b1a"],
        "tokens": [["X", "Y"]],
        "bbox": torch.zeros((1, 2, 4), dtype=torch.long),
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
        "image": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
    }

    results = clf.predict(**inputs)
    assert len(results) == 2
    assert set(r.class_name for r in results) <= {"B-header", "I-header", "O"}


@REQUIRES_PT_AND_TR
def test_layoutlm_v3_token_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmv3TokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_tok_predict(uuids: Any, input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, tokens: Any, model: Any, images: Any = None) -> list[TokenClassResult]:
        return [
            TokenClassResult(uuid="u1", token_id=1, class_id=0, token="a", score=0.9),
            TokenClassResult(uuid="u2", token_id=2, class_id=2, token="b", score=0.8),
            TokenClassResult(uuid="u3", token_id=3, class_id=1, token="c", score=0.7),
        ]

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_token_classes_from_layoutlm",
        MagicMock(side_effect=_fake_tok_predict),
        raising=True,
    )

    categories = {1: "B-header", 2: "I-header", 3: "O"}
    clf = HFLayoutLmv3TokenClassifier("path/to/json", "path/to/model", categories=categories, device="cpu")

    inputs = {
        "ann_ids": [["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"]],
        "image_ids": ["e2c7b6a4-5f3b-8c2d-a1b2-9f0e4d3c2b1a"],
        "tokens": [["a", "b", "c"]],
        "bbox": torch.zeros((1, 3, 4), dtype=torch.long),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
    }

    results = clf.predict(**inputs)
    assert len(results) == 3
    assert set(r.class_name for r in results) == {"B-header", "I-header", "O"}


@REQUIRES_PT_AND_TR
def test_lilt_token_and_sequence_predict_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLiltTokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLiltSequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    def _fake_tok_predict(uuids: Any, input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, tokens: Any, model: Any, images: Any = None) -> list[TokenClassResult]:
        return [
            TokenClassResult(uuid="t1", token_id=5, class_id=0, token="foo", score=0.5),
            TokenClassResult(uuid="t2", token_id=6, class_id=2, token="bar", score=0.6),
        ]

    def _fake_seq_predict(input_ids: Any, attention_mask: Any, token_type_ids: Any, boxes: Any, model: Any, images: Any = None) -> SequenceClassResult:
        return SequenceClassResult(class_id=0, score=0.8)

    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_token_classes_from_layoutlm",
        MagicMock(side_effect=_fake_tok_predict),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.predict_sequence_classes_from_layoutlm",
        MagicMock(side_effect=_fake_seq_predict),
        raising=True,
    )

    tok_categories = {1: "B-header", 2: "I-header", 3: "O"}
    seq_categories = {1: "letter", 2: "invoice"}

    tok = HFLiltTokenClassifier("path/to/json", "path/to/model", categories=tok_categories, device="cpu")
    seq = HFLiltSequenceClassifier("path/to/json", "path/to/model", categories=seq_categories, device="cpu")

    tok_inputs = {
        "ann_ids": [["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"]],
        "tokens": [["foo", "bar"]],
        "bbox": torch.zeros((1, 2, 4), dtype=torch.long),
        "input_ids": torch.tensor([[5, 6]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
    }
    seq_inputs = {
        "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
        "bbox": torch.zeros((1, 3, 4), dtype=torch.long),
    }

    tok_res = tok.predict(**tok_inputs)
    seq_res = seq.predict(**seq_inputs)

    assert len(tok_res) == 2
    assert seq_res.class_name == "letter"


@REQUIRES_PT_AND_TR
def test_sequence_validate_encodings_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep ctor offline
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmSequenceClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    categories = {1: "letter", 2: "invoice"}
    clf = HFLayoutLmSequenceClassifier("path/to/json", "path/to/model", categories, device="cpu")

    with pytest.raises(ValueError):
        clf.predict(
            input_ids=[[1, 2]],  # not a torch.Tensor
            attention_mask=torch.tensor([[1, 1]]),
            token_type_ids=torch.tensor([[0, 0]]),
            bbox=torch.zeros((1, 2, 4), dtype=torch.long),
        )


@REQUIRES_PT_AND_TR
def test_token_validate_encodings_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.HFLayoutLmTokenClassifier.get_wrapped_model",
        MagicMock(return_value=MagicMock()),
        raising=True,
    )

    clf = HFLayoutLmTokenClassifier("path/to/json", "path/to/model", categories={1: "O"}, device="cpu")

    with pytest.raises(ValueError):
        clf.predict(
            ann_ids=["3f1c2a9b-7d42-8e3a-b5c1-12f4a6c9d8e7", "a94f0d31-2b7c-8c55-a3d2-5e9b7f1a0c4d"],
            tokens=[["x", "y"]],
            bbox="not-a-tensor",
            input_ids="not-a-tensor",
            attention_mask="not-a-tensor",
            token_type_ids="not-a-tensor",
        )
