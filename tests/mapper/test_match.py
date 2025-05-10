# -*- coding: utf-8 -*-
# File: test_match.py
# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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
Testing the module mapper.match
"""
from pytest import mark

from deepdoctection import LayoutType
from deepdoctection.datapoint.image import Image
from deepdoctection.mapper.match import match_anns_by_distance

@mark.basic
def test_match_anns_by_intersection(dp_image_with_layout_and_caption_anns: Image) -> None:
    """
    Test match_anns_by_intersection
    """

    dp = dp_image_with_layout_and_caption_anns
    output = match_anns_by_distance(dp, LayoutType.TABLE, LayoutType.CAPTION)
    print(output)

    table_anns = dp.get_annotation(category_names=LayoutType.TABLE)
    caption_anns = dp.get_annotation(category_names=LayoutType.CAPTION)
    output_ids = {(anns[0].annotation_id, anns[1].annotation_id) for anns in output}
    expected_output_ids = {
        (table_anns[0].annotation_id, caption_anns[0].annotation_id),
        (table_anns[1].annotation_id, caption_anns[0].annotation_id),
    }

    assert output_ids == expected_output_ids
