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

import os
import pytest

from dd_core.utils.file_utils import pytorch_available, transformers_available
from dd_core.utils.types import PathLikeOrStr
from deepdoctection.extern.hflm import (
    HFLmSequenceClassifier,
    HFLmTokenClassifier,
    HFLmLanguageDetector,
)

if pytorch_available() and transformers_available():
    import torch
    from transformers import (
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaTokenizerFast,
    )

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available()),
    reason="Requires PyTorch and transformers installed",
)


def _dummy_tokenizer():
    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=512):
            import torch
            return {
                "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
    return DummyTokenizer()


@REQUIRES_PT_AND_TR
@pytest.mark.slow
def test_hflm_sequence_slow_build_and_predict(tmp_path: PathLikeOrStr, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = XLMRobertaConfig(num_labels=2)
    model = XLMRobertaForSequenceClassification(cfg)
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "invoice", 2: "report"}
    clf = HFLmSequenceClassifier(
        path_config_json=os.fspath(tmp_path),
        path_weights=os.fspath(tmp_path),
        categories=categories,
        device="cpu",
        use_xlm_tokenizer=True,
    )

    L = 5
    inputs = {
        "input_ids": torch.randint(50, (1, L), dtype=torch.long),
        "attention_mask": torch.ones((1, L), dtype=torch.long),
        "token_type_ids": torch.zeros((1, L), dtype=torch.long),
    }
    res = clf.predict(**inputs)
    assert res.class_name in categories.values()


@REQUIRES_PT_AND_TR
@pytest.mark.slow
def test_hflm_token_slow_build_and_predict(tmp_path: PathLikeOrStr, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = XLMRobertaConfig(num_labels=3)
    model = XLMRobertaForTokenClassification(cfg)
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    categories = {1: "B-head", 2: "I-head", 3: "O"}
    clf = HFLmTokenClassifier(
        path_config_json=os.fspath(tmp_path),
        path_weights=os.fspath(tmp_path),
        categories=categories,
        device="cpu",
        use_xlm_tokenizer=True,
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
def test_hflm_language_slow_build_and_predict(tmp_path: PathLikeOrStr, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a tiny language detection model (sequence classification head)
    cfg = XLMRobertaConfig(num_labels=3)
    model = XLMRobertaForSequenceClassification(cfg)
    model.save_pretrained(tmp_path)
    cfg.save_pretrained(tmp_path)

    # Avoid network tokenizer download
    monkeypatch.setattr(
        "transformers.XLMRobertaTokenizerFast.from_pretrained",
        lambda *args, **kwargs: _dummy_tokenizer(),
        raising=True,
    )

    categories = {1: "en", 2: "de", 3: "fr"}
    det = HFLmLanguageDetector(
        path_config_json=os.fspath(tmp_path),
        path_weights=os.fspath(tmp_path),
        categories=categories,
        device="cpu",
        use_xlm_tokenizer=True,
    )

    res = det.predict("Sample text for language detection.")
    assert res.class_name in categories.values()
    assert 0.0 <= res.score <= 1.0
