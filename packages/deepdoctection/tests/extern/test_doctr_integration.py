# -*- coding: utf-8 -*-
# File: test_doctr_integration.py

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
This module contains tests for integrating the python-doctr library with DeepDoctection components.

It validates the functionality of Doctr textline detector and text recognizer using specified weights
and configurations. The tests ensure that models can be built, used, and cleared correctly on devices
where the dependencies are available.

"""

import pytest

from dd_core.utils import get_torch_device
from dd_core.utils.file_utils import doctr_available, pytorch_available
from deepdoctection.extern.doctrocr import (
    DoctrTextlineDetector,
    DoctrTextRecognizer,
)
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager

REQUIRES_PT_AND_DOCTR = pytest.mark.skipif(
    not (pytorch_available() and doctr_available()),
    reason="Requires PyTorch and python-doctr installed",
)


@REQUIRES_PT_AND_DOCTR
@pytest.mark.slow
def test_slow_build_doctr_textline_detector_pt() -> None:
    """test basic prediction using mocked tokenizers and models."""
    weights = "doctr/db_resnet50/db_resnet50-ac60cadc.pt"
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    profile = ModelCatalog.get_profile(weights)
    device = get_torch_device()
    det = DoctrTextlineDetector(profile.architecture, weights_path, profile.categories, device)  # type: ignore

    assert det.doctr_predictor is not None
    assert len(det.get_category_names()) > 0

    det.clear_model()
    assert det.doctr_predictor is None



@REQUIRES_PT_AND_DOCTR
@pytest.mark.slow
def test_slow_build_doctr_text_recognizer_pt() -> None:
    """test basic prediction using mocked tokenizers and models."""
    weights = "doctr/crnn_vgg16_bn/crnn_vgg16_bn-9762b0b0.pt"
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    profile = ModelCatalog.get_profile(weights)
    device = get_torch_device()
    rec = DoctrTextRecognizer(profile.architecture, weights_path, device)

    assert rec.doctr_predictor is not None

    rec.clear_model()
    assert rec.doctr_predictor is None

