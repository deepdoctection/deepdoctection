# -*- coding: utf-8 -*-
# File: test_tpstruct.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
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
Testing module mapper.tpstruct
"""
from unittest.mock import MagicMock, patch

from numpy.testing import assert_allclose, assert_array_equal

from deepdoctection.datapoint.image import Image
from deepdoctection.mapper.tpstruct import image_to_tp_frcnn_training

from .data import DatapointImage


@patch("deepdoctection.mapper.tpstruct.os.path.isfile", MagicMock(return_value=True))
def test_image_to_tp_frcnn_training(datapoint_image: Image, image_results: DatapointImage) -> None:
    """
    testing image_to_tp_frcnn_training is mapping correctly
    """

    # Act
    img_to_tp_tr_mapper = image_to_tp_frcnn_training(add_mask=False)  # type: ignore # pylint: disable=E1120
    output = img_to_tp_tr_mapper(datapoint_image)  # pylint: disable=E1102

    # Assert
    expected_output = image_results.get_tp_frcnn_training_anns()

    assert output.keys() == expected_output.keys()
    assert_allclose(output["gt_boxes"], expected_output["gt_boxes"])
    assert_array_equal(output["gt_labels"], expected_output["gt_labels"])
    assert_array_equal(output["image"], expected_output["image"])
    assert output["file_name"] == expected_output["file_name"]
