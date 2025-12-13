import os
import uuid
from typing import Any, Optional

import pytest

from dd_core.utils.file_utils import detectron2_available, pytorch_available, transformers_available
from dd_core.utils.types import PathLikeOrStr
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

if transformers_available():
    from transformers import (
        LayoutLMConfig,
        LayoutLMForSequenceClassification,
        LayoutLMForTokenClassification,
        LayoutLMv2Config,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv3Config,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3ForTokenClassification,
        LiltConfig,
        LiltForSequenceClassification,
        LiltForTokenClassification,
    )

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available()),
    reason="Requires PyTorch and transformers installed",
)


def _mk_dummy_tokenizer() -> Any:
    class DummyTokenizer:
        pass

    return DummyTokenizer()


@REQUIRES_PT_AND_TR
@pytest.mark.slow
@pytest.mark.parametrize(
    "wrapper_cls, model_cls, cfg_cls, needs_img, img_key",
    [
        (
            HFLayoutLmSequenceClassifier,
            pytest.param(None, None, marks=pytest.mark.skip(reason="Use v2/v3/LiLT for heavier coverage")),
            None,
            False,
            None,
        ),
        pytest.param(
            HFLayoutLmv2SequenceClassifier,
            None,
            None,
            True,
            "image",
            marks=pytest.mark.skipif(
                not detectron2_available(),
                reason="Requires Detectron2 for LayoutLMv2",
            ),
        ),
        (HFLayoutLmv3SequenceClassifier, None, None, True, "pixel_values"),
        (HFLiltSequenceClassifier, None, None, False, None),
    ],
)
def test_sequence_slow_build_and_predict(
    tmp_path: PathLikeOrStr,
    monkeypatch: pytest.MonkeyPatch,
    wrapper_cls: type,
    model_cls: type | None,
    cfg_cls: type | None,
    needs_img: bool,
    img_key: Optional[str],
) -> None:
    """
    Build a tiny model, save it locally, load via wrapped classifier and run one inference.
    """
    # Avoid tokenizer network calls
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )

    if wrapper_cls is HFLayoutLmSequenceClassifier:
        cfg_cls = LayoutLMConfig
        model_cls = LayoutLMForSequenceClassification
    elif wrapper_cls is HFLayoutLmv2SequenceClassifier:
        cfg_cls = LayoutLMv2Config
        model_cls = LayoutLMv2ForSequenceClassification
    elif wrapper_cls is HFLayoutLmv3SequenceClassifier:
        cfg_cls = LayoutLMv3Config
        model_cls = LayoutLMv3ForSequenceClassification
    elif wrapper_cls is HFLiltSequenceClassifier:
        cfg_cls = LiltConfig
        model_cls = LiltForSequenceClassification
    else:
        pytest.skip("Unsupported wrapper")

    cfg = cfg_cls(num_labels=2)
    model = model_cls(cfg)
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "letter", 2: "invoice"}
    clf = wrapper_cls(os.fspath(tmp_path), os.fspath(tmp_path), categories, device="cpu")

    L = 4
    inputs = {
        "input_ids": torch.randint(5, (1, L), dtype=torch.long),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "token_type_ids": torch.zeros((1, L), dtype=torch.long),
        "bbox": torch.zeros((1, L, 4), dtype=torch.long),
    }
    if needs_img:
        assert img_key is not None
        inputs[img_key] = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    result = clf.predict(**inputs)
    assert result.class_name in categories.values()


@REQUIRES_PT_AND_TR
@pytest.mark.slow
@pytest.mark.parametrize(
    "wrapper_cls, model_cls, cfg_cls, needs_img, img_key",
    [
        # Token classifiers
        (
            HFLayoutLmTokenClassifier,
            pytest.param(None, None, marks=pytest.mark.skip(reason="Use v2/v3/LiLT for heavier coverage")),
            None,
            False,
            None,
        ),
        pytest.param(
            HFLayoutLmv2TokenClassifier,
            None,
            None,
            True,
            "image",
            marks=pytest.mark.skipif(
                not detectron2_available(),
                reason="Requires Detectron2 for LayoutLMv2",
            ),
        ),
        (HFLayoutLmv3TokenClassifier, None, None, True, "pixel_values"),
        (HFLiltTokenClassifier, None, None, False, None),
    ],
)
def test_token_slow_build_and_predict(
    tmp_path: PathLikeOrStr,
    monkeypatch: pytest.MonkeyPatch,
    wrapper_cls: type,
    model_cls: type | None,
    cfg_cls: type | None,
    needs_img: bool,
    img_key: Optional[str],
) -> None:
    """
    Build a tiny token model, save/load weights, and run one inference end-to-end.
    """
    # Avoid tokenizer network calls
    monkeypatch.setattr(
        "deepdoctection.extern.hflayoutlm.get_tokenizer_from_model_class",
        lambda cls, use_xlm: _mk_dummy_tokenizer(),
        raising=True,
    )

    # Resolve config/model classes per wrapper
    if wrapper_cls is HFLayoutLmTokenClassifier:
        cfg_cls = LayoutLMConfig
        model_cls = LayoutLMForTokenClassification
    elif wrapper_cls is HFLayoutLmv2TokenClassifier:
        cfg_cls = LayoutLMv2Config
        model_cls = LayoutLMv2ForTokenClassification
    elif wrapper_cls is HFLayoutLmv3TokenClassifier:
        cfg_cls = LayoutLMv3Config
        model_cls = LayoutLMv3ForTokenClassification
    elif wrapper_cls is HFLiltTokenClassifier:
        cfg_cls = LiltConfig
        model_cls = LiltForTokenClassification
    else:
        pytest.skip("Unsupported wrapper")

    # Create and save tiny model
    cfg = cfg_cls(num_labels=3)
    model = model_cls(cfg)
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "B-header", 2: "I-header", 3: "O"}
    clf = wrapper_cls(os.fspath(tmp_path), os.fspath(tmp_path), categories=categories, device="cpu")

    L = 3
    inputs = {
        "ann_ids": [[uuid.uuid4().hex[:8] for _ in range(L)]],
        "image_ids": ["img-1"],
        "tokens": [["t0", "t1", "t2"]],
        "bbox": torch.zeros((1, L, 4), dtype=torch.long),
        "input_ids": torch.randint(10, (1, L), dtype=torch.long),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "token_type_ids": torch.zeros((1, L), dtype=torch.long),
    }
    if needs_img:
        assert img_key is not None
        inputs[img_key] = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    results = clf.predict(**inputs)
    assert len(results) == L
    assert all(r.class_name in categories.values() for r in results)
