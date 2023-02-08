# -*- coding: utf-8 -*-
# File: test_hfstruct.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Testing module hfstruct
"""

from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.datapoint.image import Image
from deepdoctection.utils import transformers_available

from .data import DatapointImage

if transformers_available():
    from deepdoctection.mapper.hfstruct import image_to_hf_detr_training


@mark.requires_pt
@patch("deepdoctection.mapper.hfstruct.os.path.isfile", MagicMock(return_value=True))
def test_image_to_hf_detr_training(datapoint_image: Image, image_results: DatapointImage) -> None:
    """
    testing image_to_hf_detr_training is mapping correctly
    """

    # Act
    img_to_hf_tr_mapper = image_to_hf_detr_training(add_mask=False)  # pylint: disable=E1120
    output = img_to_hf_tr_mapper(datapoint_image)

    # Assert
    expected_output = image_results.get_hf_detr_training_anns()

    assert output
    assert output.keys() == expected_output.keys()
    assert output["width"] == output["width"]

    first_annotation = output["annotations"][0]
    expected_first_annotation = expected_output["annotations"][0]

    assert first_annotation["bbox"] == expected_first_annotation["bbox"]
    assert first_annotation["category_id"] == expected_first_annotation["category_id"]
    assert first_annotation["image_id"] == expected_first_annotation["image_id"]
    assert first_annotation["id"] == expected_first_annotation["id"]
