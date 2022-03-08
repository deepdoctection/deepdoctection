# -*- coding: utf-8 -*-
# File: test_iiitar13k.py

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
Testing module mapper.iiitarstruct
"""

from math import isclose
from typing import Dict
from unittest.mock import MagicMock, patch

from deepdoctection.mapper.pascalstruct import pascal_voc_dict_to_image
from deepdoctection.utils.detection_types import JsonDict

from .data import IIITar13KJson


@patch("deepdoctection.mapper.pascalstruct.load_image_from_file", MagicMock())
def test_pascal_voc_dict_to_image(
    datapoint_iiitar13kjson: JsonDict,
    iiitar13k_categories_name_as_keys: Dict[str, str],
    iiitar13k_category_names_mapping: Dict[str, str],
    iiitar13k_results: IIITar13KJson,
) -> None:

    """
    testing iiitar13k_to_image is mapping correctly
    """

    # Act
    iiitar13k_to_image_mapper = pascal_voc_dict_to_image(  # type: ignore
        iiitar13k_categories_name_as_keys, False, False, False, iiitar13k_category_names_mapping
    )
    dp = iiitar13k_to_image_mapper(datapoint_iiitar13kjson)

    # Assert
    datapoint = iiitar13k_results
    assert dp is not None
    test_anns = dp.get_annotation()
    assert len(test_anns) == datapoint.get_number_anns()
    assert dp.width == datapoint.get_width()
    assert dp.height == datapoint.get_height()
    assert test_anns[0].category_name == datapoint.get_first_ann_category_name()
    assert isclose(test_anns[0].bounding_box.ulx, datapoint.get_first_ann_box().ulx, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.uly, datapoint.get_first_ann_box().uly, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.width, datapoint.get_first_ann_box().w, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.height, datapoint.get_first_ann_box().h, rel_tol=1e-15)
