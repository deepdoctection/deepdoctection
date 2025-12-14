# -*- coding: utf-8 -*-
# File: test_hfdetr_integration.py

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

import pytest

from dd_core.utils import get_torch_device
from dd_core.utils.file_utils import pytorch_available, timm_available, transformers_available
from deepdoctection.extern.hfdetr import HFDetrDerivedDetector
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager

REQUIRES_PT_AND_TR = pytest.mark.skipif(
    not (pytorch_available() and transformers_available() and timm_available()),
    reason="Requires PyTorch, Transformers and Timm to be installed",
)


@REQUIRES_PT_AND_TR
@pytest.mark.slow
def test_slow_build_hfdetr_detector_pt() -> None:
    # Use Table Transformer pre-trained model as example
    weights = "microsoft/table-transformer-detection/model.safetensors"
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    config_path = ModelCatalog.get_full_path_configs(weights)
    feature_extractor_config_path = ModelCatalog.get_full_path_preprocessor_configs(weights)
    categories = ModelCatalog.get_profile(weights).categories
    assert categories is not None

    device = get_torch_device()
    det = HFDetrDerivedDetector(
        path_config_json=config_path,
        path_weights=weights_path,
        path_feature_extractor_config_json=feature_extractor_config_path,
        categories=categories,
        device=device,
    )

    assert det.hf_detr_predictor is not None
    assert len(det.get_category_names()) > 0

    det.clear_model()
    assert det.hf_detr_predictor is None
