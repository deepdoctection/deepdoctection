# -*- coding: utf-8 -*-
# File: test_prodigystruct.py
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

from math import isclose

from dd_core.mapper.prodigystruct import prodigy_to_image, image_to_prodigy
from dd_core.utils.object_types import get_type


def test_prodigy_to_image_basic(prodigy_datapoint):
    """
    Basic mapping from prodigy datapoint to Image: file_name, size and bounding box.
    """
    categories = {get_type("table"): 1, get_type("title"): 2}

    image = prodigy_to_image(categories, True, False)(prodigy_datapoint)  # load_image=True, fake_score=False

    assert image is not None
    assert image.file_name == prodigy_datapoint["text"]
    assert image.width == 17
    assert image.height == 34

    anns = image.get_annotation()
    assert len(anns) == len(prodigy_datapoint["spans"])

    # check first annotation bbox coordinates (derived from points in fixture)
    bbox = anns[0].get_bounding_box(image.image_id)
    assert isclose(bbox.ulx, 1.0, rel_tol=1e-9)
    assert isclose(bbox.uly, 3.0, rel_tol=1e-9)
    assert isclose(bbox.lrx, 15.0, rel_tol=1e-9)
    assert isclose(bbox.lry, 29.0, rel_tol=1e-9)



def test_prodigy_to_image_accept_only_filters(prodigy_datapoint):
    """
    When accept_only_answer=True and fixture has answer 'reject', mapper returns None.
    """
    categories = {get_type("table"): 1, get_type("title"): 2}
    image = prodigy_to_image(categories, True, False, accept_only_answer=True)(prodigy_datapoint)
    assert image is None



def test_image_to_prodigy_roundtrip(prodigy_datapoint):
    """
    Create Image from prodigy datapoint then map back to prodigy format and check image & meta.
    """
    categories = {get_type("table"): 1, get_type("title"): 2}
    image = prodigy_to_image(categories, True, False)(prodigy_datapoint)
    assert image is not None

    output = image_to_prodigy()(image)
    assert output["text"] == prodigy_datapoint["text"]
    assert output["meta"]["file_name"] == image.file_name
    assert len(output["spans"]) == len(prodigy_datapoint["spans"])



def test_image_to_prodigy_points_and_labels(prodigy_datapoint):
    """
    Ensure spans keep their point coordinates and present expected keys.
    """
    categories = {get_type("table"): 1, get_type("title"): 2}
    mapper = prodigy_to_image(categories, True, False)
    image = mapper(prodigy_datapoint)
    assert image is not None

    output = image_to_prodigy()(image)
    span0 = output["spans"][0]
    # points should reflect the fixture values (as floats)
    assert isclose(float(span0["points"][0][0]), 1.0, rel_tol=1e-9)
    assert isclose(float(span0["points"][0][1]), 3.0, rel_tol=1e-9)
    # check expected keys exist
    assert "label" in span0 and "points" in span0 and "type" in span0 and "score" in span0
