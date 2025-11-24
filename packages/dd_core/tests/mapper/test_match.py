# -*- coding: utf-8 -*-
# File: test_match.py
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
import pytest

from dd_core.datapoint.annotation import ImageAnnotation
from dd_core.mapper.match import match_anns_by_intersection, match_anns_by_distance
from dd_core.utils.object_types import LayoutType


def test_ioa_threshold_monotonicity(annotations):
    """
    Lower threshold should yield at least as many ioa matches as a higher threshold.
    Uses annotations(use_layout=True, use_captions=True).
    """
    dp = annotations(use_layout=True, use_captions=True)
    child_idx_low, _, _, _ = match_anns_by_intersection(dp, matching_rule="ioa", threshold=0.0)
    child_idx_high, _, _, _ = match_anns_by_intersection(dp, matching_rule="ioa", threshold=0.5)
    assert len(child_idx_low) >= len(child_idx_high)

def test_iou_table_caption_intersection(annotations):
    dp = annotations(use_layout=True, use_captions=True)
    child_idx, parent_idx, child_anns, parent_anns = match_anns_by_intersection(dp, matching_rule="ioa", parent_ann_category_names="table",child_ann_category_names="caption",threshold=0.8)
    assert child_anns[0].annotation_id=='89a1de97-ac04-30c4-9c07-5b88c7a0485c'
    assert parent_anns[parent_idx[0]].annotation_id=='773eb5ea-1757-3f18-88f3-fdffebe771cc'

def test_ioa_max_parent_only_uniqueness(annotations):
    """
    With max_parent_only=True each child index should appear at most once and parent/child arrays align.
    """
    dp = annotations(use_layout=True, use_captions=True)
    child_idx, parent_idx, child_anns, parent_anns = match_anns_by_intersection(
        dp, matching_rule="ioa",
        threshold=0.0,
        max_parent_only=True
    )
    assert len(child_idx) == len(parent_idx)
    assert len(set(child_idx.tolist())) == len(child_idx)


def test_iou_threshold_monotonicity(annotations):
    """
    Lower threshold should yield at least as many iou matches as a higher threshold.
    """
    dp = annotations(use_layout=True, use_captions=True)
    child_idx_low, _, _, _ = match_anns_by_intersection(dp, matching_rule="iou", threshold=0.0)
    child_idx_high, _, _, _ = match_anns_by_intersection(dp, matching_rule="iou", threshold=0.5)
    assert len(child_idx_low) >= len(child_idx_high)


def test_distance_assigned_child_is_closest(annotations):
    """
    For each returned (parent, child) pair the child must be the nearest among all children.
    """
    dp = annotations(use_layout=True, use_captions=True)
    output = match_anns_by_distance(dp, LayoutType.TABLE, LayoutType.CAPTION)

    table_anns = dp.get_annotation(category_names=LayoutType.TABLE)
    caption_anns = dp.get_annotation(category_names=LayoutType.CAPTION)
    output_ids = {(anns[0].annotation_id, anns[1].annotation_id) for anns in output}
    expected_output_ids = {
        (table_anns[0].annotation_id, caption_anns[0].annotation_id),
        (table_anns[1].annotation_id, caption_anns[0].annotation_id),
    }

    assert output_ids == expected_output_ids
