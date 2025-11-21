# -*- coding: utf-8 -*-
# File: test_hfstruct.py

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


import numpy as np

from dd_core.mapper.hfstruct import image_to_hf_detr_training
from dd_core.datapoint.image import Image


def test_image_to_hf_detr_training_all_categories(image_with_layout_anns: Image) -> None:
    # Mock image pixels to bypass file loading
    image_with_layout_anns.image = np.zeros(
        (int(image_with_layout_anns.height), int(image_with_layout_anns.width), 3),
        dtype=np.uint8,
    )

    output = image_to_hf_detr_training(category_names=None)(image_with_layout_anns)
    assert output is not None
    assert "annotations" in output
    assert len(output["annotations"]) == 21


def test_image_to_hf_detr_training_filtered_text(image_with_layout_anns: Image) -> None:
    # Mock image pixels to bypass file loading
    image_with_layout_anns.image = np.zeros(
        (int(image_with_layout_anns.height), int(image_with_layout_anns.width), 3),
        dtype=np.uint8,
    )

    output = image_to_hf_detr_training(category_names=["text"])(image_with_layout_anns)
    assert output is not None
    assert "annotations" in output
    assert len(output["annotations"]) == 8
