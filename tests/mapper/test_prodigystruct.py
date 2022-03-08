# -*- coding: utf-8 -*-
# File: test_prodigystruct.py

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
Testing the module mapper.prodigystruct
"""


from math import isclose
from typing import Dict

from deepdoctection.datapoint.image import Image
from deepdoctection.mapper import image_to_prodigy, prodigy_to_image
from deepdoctection.utils.detection_types import JsonDict

from .data import DatapointImage, DatapointProdigy


def test_prodigy_to_image(
    datapoint_prodigy: JsonDict, categories_prodigy: Dict[str, str], prodigy_results: DatapointProdigy
) -> None:
    """
    testing prodigy_to_image is mapping correctly
    """

    load_image = True
    # Act
    prodigy_to_image_mapper = prodigy_to_image(categories_prodigy, load_image, False)  # type: ignore  # pylint: disable=E1120  # 259
    dp = prodigy_to_image_mapper(datapoint_prodigy)

    # Assert
    datapoint = prodigy_results
    assert dp is not None
    test_anns = dp.get_annotation()
    assert len(test_anns) == datapoint.get_number_anns()
    assert dp.width == datapoint.get_width(load_image)
    assert dp.height == datapoint.get_height(load_image)
    assert test_anns[0].category_name == datapoint.get_first_ann_category(False)
    assert test_anns[0].category_id == datapoint.get_first_ann_category(True)
    assert isclose(test_anns[0].bounding_box.ulx, datapoint.get_first_ann_box().ulx, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.uly, datapoint.get_first_ann_box().uly, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.width, datapoint.get_first_ann_box().w, rel_tol=1e-15)
    assert isclose(test_anns[0].bounding_box.height, datapoint.get_first_ann_box().h, rel_tol=1e-15)


def test_image_to_prodigy(datapoint_image: Image, image_results: DatapointImage) -> None:
    """
    testing image_to_prodigy is mapping correctly
    """

    # Act
    output = image_to_prodigy(datapoint_image)

    # Assert
    datapoint = image_results
    assert output["image"] == datapoint.get_image_str()
    assert output["text"] == datapoint.get_text()
    assert len(output["spans"]) == datapoint.get_len_spans()
    assert output["spans"][0] == datapoint.get_first_span()
    assert output["spans"][1] == datapoint.get_second_span()
