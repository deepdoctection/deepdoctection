# -*- coding: utf-8 -*-
# File: test_d2detect_integration.py

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
from dd_core.utils.file_utils import detectron2_available, pytorch_available
from deepdoctection.extern.d2detect import D2FrcnnDetector, D2FrcnnTracingDetector
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager

REQUIRES_PT_AND_D2 = pytest.mark.skipif(
    not (pytorch_available() and detectron2_available()),
    reason="Requires PyTorch and Detectron2 installed",
)


@REQUIRES_PT_AND_D2
@pytest.mark.slow
def test_slow_build_d2_frcnn_detector_pt() -> None:
    """
    Build Detectron2 GeneralizedRCNN model and load weights (PT). No inference to keep it fast-ish.
    """
    device = get_torch_device()
    weights = "layout/d2_model_0829999_layout_inf_only.pt"
    config_path = ModelCatalog.get_full_path_configs(weights)
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    categories = ModelCatalog.get_profile(weights).categories
    assert categories is not None

    det = D2FrcnnDetector(
        path_yaml=config_path,
        path_weights=weights_path,
        categories=categories,
        device=device,
        filter_categories=None,
    )

    # Basic sanity of initialization
    assert det.d2_predictor is not None
    assert det.resizer is not None
    assert len(det.get_category_names()) > 0

    # Clear to release memory
    det.clear_model()
    assert det.d2_predictor is None


@REQUIRES_PT_AND_D2
@pytest.mark.slow
def test_slow_build_d2_frcnn_tracing_detector_ts() -> None:
    """
    Build TorchScript exported model and load weights (.ts). No inference to keep it fast-ish.
    """
    weights = "layout/d2_model_0829999_layout_inf_only.ts"
    config_path = ModelCatalog.get_full_path_configs(weights)
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    categories = ModelCatalog.get_profile(weights).categories
    assert categories is not None

    det = D2FrcnnTracingDetector(
        path_yaml=config_path,
        path_weights=weights_path,
        categories=categories,
        filter_categories=None,
    )

    # Basic sanity of initialization
    assert det.d2_predictor is not None
    assert det.resizer is not None
    assert len(det.get_category_names()) > 0

    # Clear to release memory
    det.clear_model()
    assert det.d2_predictor is None
