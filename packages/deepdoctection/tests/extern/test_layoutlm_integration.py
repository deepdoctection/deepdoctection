# -*- coding: utf-8 -*-
# File: test_layoutlm_integration.py

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
Integration tests for sequence and token classification with HuggingFace's LayoutLM, LayoutLMv2,
LayoutLMv3, and LiLT models.

Allows testing model wrappers for the ability to save, load, and perform inference under various
configurations. Includes support for both sequence and token-level classifications under models
provided from HuggingFace.

"""

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
    class DummyTokenizer:  # pylint:disable=C0115, R0903
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
    wrapper_cls: Any,
    model_cls: Any,
    cfg_cls: Any,
    needs_img: bool,
    img_key: Optional[str],
) -> None:
    """
    Build a tiny model, save it locally, load via wrapped classifier and run one inference.
    """

    if wrapper_cls is HFLayoutLmSequenceClassifier:
        cfg_cls = LayoutLMConfig  # pylint:disable=E0606
        model_cls = LayoutLMForSequenceClassification  # pylint:disable=E0606
    elif wrapper_cls is HFLayoutLmv2SequenceClassifier:
        cfg_cls = LayoutLMv2Config  # pylint:disable=E0606
        model_cls = LayoutLMv2ForSequenceClassification  # pylint:disable=E0606
    elif wrapper_cls is HFLayoutLmv3SequenceClassifier:
        cfg_cls = LayoutLMv3Config  # pylint:disable=E0606
        model_cls = LayoutLMv3ForSequenceClassification  # pylint:disable=E0606
    elif wrapper_cls is HFLiltSequenceClassifier:
        cfg_cls = LiltConfig  # pylint:disable=E0606
        model_cls = LiltForSequenceClassification  # pylint:disable=E0606
    else:
        pytest.skip("Unsupported wrapper")

    assert model_cls is not None
    assert cfg_cls is not None

    cfg = cfg_cls(num_labels=2)
    model = model_cls(cfg)
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "letter", 2: "invoice"}
    clf = wrapper_cls(os.fspath(tmp_path), os.fspath(tmp_path), categories, device="cpu")

    L = 4
    inputs = {
        "input_ids": torch.randint(5, (1, L), dtype=torch.long),  # pylint:disable=E0606
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
    wrapper_cls: Any,
    model_cls: Any,
    cfg_cls: Any,
    needs_img: bool,
    img_key: Optional[str],
) -> None:
    """
    Build a tiny token model, save/load weights, and run one inference end-to-end.
    """

    # Resolve config/model classes per wrapper
    if wrapper_cls is HFLayoutLmTokenClassifier:
        cfg_cls = LayoutLMConfig
        model_cls = LayoutLMForTokenClassification  # pylint:disable=E0606
    elif wrapper_cls is HFLayoutLmv2TokenClassifier:
        cfg_cls = LayoutLMv2Config
        model_cls = LayoutLMv2ForTokenClassification  # pylint:disable=E0606
    elif wrapper_cls is HFLayoutLmv3TokenClassifier:
        cfg_cls = LayoutLMv3Config
        model_cls = LayoutLMv3ForTokenClassification  # pylint:disable=E0606
    elif wrapper_cls is HFLiltTokenClassifier:
        cfg_cls = LiltConfig
        model_cls = LiltForTokenClassification  # pylint:disable=E0606
    else:
        pytest.skip("Unsupported wrapper")

    # Create and save tiny model
    assert model_cls is not None
    assert cfg_cls is not None

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
