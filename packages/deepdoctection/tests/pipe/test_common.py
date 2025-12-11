# -*- coding: utf-8 -*-
# File: test_common.py

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

from deepdoctection.pipe.common import (
    AnnotationNmsService,
    FamilyCompound,
    IntersectionMatcher,
    MatchingService,
    NeighbourMatcher,
    PageParsingService,
)

from dd_core.utils.object_types import LayoutType, Relationships, ObjectTypes

from dd_core.datapoint.image import Image
from dd_core.datapoint.view import Page
from dd_core.datapoint.box import BoundingBox
from dd_core.datapoint.annotation import ImageAnnotation
from dd_core.utils.file_utils import pytorch_available



def make_ann(category: ObjectTypes, box, score=0.9) -> ImageAnnotation:
    ann = ImageAnnotation(category_name=category, score=score, bounding_box=BoundingBox(**box))
    assert ann.get_defining_attributes() == ["category_name", "bounding_box"]
    return ann

@pytest.mark.skipif(not pytorch_available(), reason="Pytorch not installed")
@pytest.mark.parametrize(
    "pairs,thresh,prio",
    [
        ([[LayoutType.TITLE, LayoutType.TEXT]], [0.5], [LayoutType.TITLE]),
        ([[LayoutType.TABLE, LayoutType.TABLE_ROTATED]], [0.3], [LayoutType.TABLE]),
    ],
)
def test_annotation_nms_service_serves(dp_image, pairs, thresh, prio):
    a1 = make_ann(LayoutType.TEXT, {"ulx": 10, "uly": 10, "width": 100, "height": 20, "absolute_coords": True})
    a2 = make_ann(LayoutType.TEXT, {"ulx": 15, "uly": 12, "width": 100, "height": 20, "absolute_coords": True})
    a3 = make_ann(LayoutType.TITLE, {"ulx": 12, "uly": 9, "width": 100, "height": 20, "absolute_coords": True})

    dp_image.dump(a1)
    dp_image.dump(a2)
    dp_image.dump(a3)

    svc = AnnotationNmsService(nms_pairs=pairs, thresholds=thresh, priority=prio)
    out = svc.pass_datapoint(dp_image)

    anns_text = out.get_annotation(category_names=[LayoutType.TEXT])
    assert len(anns_text) <= 2
    assert isinstance(out, Image)


def test_matching_service_child_relationships(dp_image):
    parent = make_ann(LayoutType.LIST, {"ulx": 50, "uly": 50, "width": 200, "height": 200, "absolute_coords": True})
    child1 = make_ann(LayoutType.LIST_ITEM, {"ulx": 60, "uly": 60, "width": 50, "height": 20, "absolute_coords": True})
    child2 = make_ann(LayoutType.LIST_ITEM, {"ulx": 300, "uly": 300, "width": 50, "height": 20, "absolute_coords": True})  # outside

    dp_image.dump(parent)
    dp_image.dump(child1)
    dp_image.dump(child2)

    matcher = IntersectionMatcher(matching_rule="iou", threshold=0.01, max_parent_only=False)
    fam = FamilyCompound(
        parent_categories=[LayoutType.LIST],
        child_categories=[LayoutType.LIST_ITEM],
        relationship_key=Relationships.CHILD,
    )
    svc = MatchingService(family_compounds=[fam], matcher=matcher)

    _ = svc.pass_datapoint(dp_image)

    rels = parent.get_relationship(Relationships.CHILD)
    assert child1.annotation_id in rels
    assert child2.annotation_id not in rels


def test_matching_service_synthetic_parent_creation(dp_image):
    child1 = make_ann(LayoutType.LIST_ITEM, {"ulx": 60, "uly": 60, "width": 50, "height": 20, "absolute_coords": True})
    child2 = make_ann(LayoutType.LIST_ITEM, {"ulx": 80, "uly": 100, "width": 50, "height": 20, "absolute_coords": True})

    dp_image.dump(child1)
    dp_image.dump(child2)

    matcher = IntersectionMatcher(matching_rule="iou", threshold=0.0, max_parent_only=False)
    fam = FamilyCompound(
        parent_categories=[LayoutType.LIST],
        child_categories=[LayoutType.LIST_ITEM],
        relationship_key=Relationships.CHILD,
        create_synthetic_parent=True,
        synthetic_parent=LayoutType.LIST,
    )
    svc = MatchingService(family_compounds=[fam], matcher=matcher)
    out = svc.pass_datapoint(dp_image)

    parents = out.get_annotation(category_names=[LayoutType.LIST])
    assert len(parents) >= 1
    assert (child1.annotation_id in parents[0].get_relationship(Relationships.CHILD) and
            child2.annotation_id in parents[1].get_relationship(Relationships.CHILD))


def test_neighbour_matcher_layout_link(dp_image):
    # Two text blocks near each other should be linked via NeighbourMatcher
    a = make_ann(LayoutType.TEXT, {"ulx": 100, "uly": 100, "width": 80, "height": 20, "absolute_coords": True})
    b = make_ann(LayoutType.CAPTION, {"ulx": 190, "uly": 105, "width": 80, "height": 20, "absolute_coords": True})

    dp_image.dump(a)
    dp_image.dump(b)

    neighbor = NeighbourMatcher()
    fam = FamilyCompound(
        parent_categories=[LayoutType.TEXT],
        child_categories=[LayoutType.CAPTION],
        relationship_key=Relationships.LAYOUT_LINK,
    )
    svc = MatchingService(family_compounds=[fam], matcher=neighbor)
    _ = svc.pass_datapoint(dp_image)

    rels_a = a.get_relationship(Relationships.LAYOUT_LINK)
    assert b.annotation_id in rels_a


def test_page_parsing_service_basic(dp_image):
    container = make_ann(LayoutType.TEXT, {"ulx": 0, "uly": 0, "width": 500, "height": 500, "absolute_coords": True})
    line1 = make_ann(LayoutType.LINE, {"ulx": 10, "uly": 20, "width": 100, "height": 15, "absolute_coords": True})
    line2 = make_ann(LayoutType.LINE, {"ulx": 12, "uly": 40, "width": 100, "height": 15, "absolute_coords": True})

    dp_image.dump(container)
    dp_image.dump(line1)
    dp_image.dump(line2)

    svc = PageParsingService(
        text_container=LayoutType.WORD,
        floating_text_block_categories=[LayoutType.TEXT],
        include_residual_text_container=True,
    )
    out = svc.pass_datapoint(dp_image)

    assert isinstance(out, Page)
