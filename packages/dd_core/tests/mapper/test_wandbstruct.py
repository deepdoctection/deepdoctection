# -*- coding: utf-8 -*-
# File: test_wandbstruct.py

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
Unit tests for WandB integration utilities and functions.

This module contains tests for verifying the correct functionality
of utilities used for generating WandB-compatible objects from
application-specific data structures. It includes checks for
ensuring compatibility with WandB's Python API and verifying the
output structure and data integrity.

"""

# -*- coding: utf-8 -*-
# File: test_wandbstruct.py

import pytest

from dd_core.datapoint.annotation import ImageAnnotation
from dd_core.mapper.wandbstruct import to_wandb_image
from dd_core.utils.file_utils import wandb_available
from dd_core.utils.object_types import ObjectTypes

if wandb_available():
    from wandb import Image as Wbimage


def _build_categories(anns: ImageAnnotation) -> dict[int, ObjectTypes]:
    names = sorted({ann.category_name for ann in anns})  # type: ignore
    return {idx + 1: name for idx, name in enumerate(names, start=1)}


@pytest.mark.skipif(not wandb_available(), reason="W&B is not installed")
def test_to_wandb_image_builds_boxes_with_stubbed_wandb(annotations):  # type: ignore
    """Test that to_wandb_image builds boxes with stubbed W&B."""

    dp = annotations(True, False)

    all_annotations = dp.get_annotation()
    categories = _build_categories(all_annotations)

    image_id, wb_image = to_wandb_image(categories)(dp) # pylint:disable=E1102

    assert isinstance(wb_image, Wbimage)
    assert image_id == "6ea00afa-ba27-382a-8952-51f8e6ce16b7"
