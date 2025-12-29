# -*- coding: utf-8 -*-
# File: test_anngen.py

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
Unit tests for annotation generation with DatapointManager.

This module contains a suite of test cases for verifying the behavior
of the DatapointManager from the deepdoctection library. It ensures
correct functionality for adding, updating, removing, and managing
image annotations, as well as handling their relationships and
categories.

"""

import pytest

from dd_core.datapoint import ContainerAnnotation, Image
from dd_core.utils.object_types import Relationships, get_type
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.anngen import DatapointManager


def _detection_result(box: list[float], name: str = "text", cid: int = 1, score: float = 0.9) -> DetectionResult:
    return DetectionResult(box=box, class_name=get_type(name), class_id=cid, score=score, absolute_coords=True)


def test_set_image_annotation_and_cache(dp_image: Image) -> None:
    """test set_image_annotation and cache"""
    mgr = DatapointManager(service_id="ibw2e9f9", model_id="idc8e9f3")
    mgr.datapoint = dp_image

    ann_id = mgr.set_image_annotation(_detection_result([10, 10, 30, 30]))
    assert ann_id is not None
    assert ann_id in mgr._cache_anns
    ann = mgr.get_annotation(ann_id)
    assert ann.category_name == "text"
    bbox = ann.bounding_box
    assert bbox is not None
    assert bbox.ulx == 10


def test_set_image_annotation_with_image_and_child_relationship(dp_image: Image) -> None:
    """test set_image_annotation with image and child relationship"""
    mgr = DatapointManager(service_id="svc", model_id="m")
    mgr.datapoint = dp_image

    parent_id = mgr.set_image_annotation(_detection_result([5, 5, 50, 50]), to_image=True)
    child_id = mgr.set_image_annotation(_detection_result([7, 7, 20, 20]), to_annotation_id=parent_id)

    parent = mgr.datapoint.get_annotation(annotation_ids=parent_id)[0]
    child = mgr.datapoint.get_annotation(annotation_ids=child_id)[0]

    assert parent.image is not None
    assert Relationships.CHILD in parent.relationships
    assert child_id in parent.get_relationship(Relationships.CHILD)
    assert child.annotation_id == child_id


def test_category_and_container_annotations(dp_image: Image) -> None:
    """test set_category_annotation and set_container_annotation"""
    mgr = DatapointManager(service_id="svc", model_id="m")
    mgr.datapoint = dp_image
    ann_id = mgr.set_image_annotation(_detection_result([0, 0, 10, 10]))
    assert ann_id is not None

    mgr.set_category_annotation(get_type("test_cat_1"), 7, get_type("sub_cat_1"), ann_id, 0.5)
    mgr.set_container_annotation(get_type("test_cat_2"), 9, get_type("sub_cat_2"), ann_id, "value_x", 0.6)

    ann = mgr.get_annotation(ann_id)
    cat_ann = ann.get_sub_category(get_type("sub_cat_1"))
    cont_ann = ann.get_sub_category(get_type("sub_cat_2"))

    assert cat_ann.category_id == 7
    assert cat_ann.score == 0.5
    # cont_ann is a ContainerAnnotation, not CategoryAnnotation
    assert isinstance(cont_ann, ContainerAnnotation)
    assert cont_ann.value == "value_x"
    assert cont_ann.category_id == 9


def test_summary_annotation(dp_image: Image) -> None:
    """test set_summary_annotation"""
    mgr = DatapointManager(service_id="svc", model_id="m")
    mgr.datapoint = dp_image
    ann_id = mgr.set_image_annotation(_detection_result([0, 0, 20, 20]), to_image=True)
    assert ann_id is not None

    summ_global_id = mgr.set_summary_annotation(get_type("sub_cat_1"), get_type("test_cat_1"), 1)
    summ_local_id = mgr.set_summary_annotation(get_type("sub_cat_2"), get_type("test_cat_2"), 2, annotation_id=ann_id)

    global_summ = mgr.datapoint.summary.get_sub_category(get_type("sub_cat_1"))
    local_summ = mgr.get_annotation(ann_id).get_summary(get_type("sub_cat_2"))

    assert global_summ.annotation_id == summ_global_id
    assert local_summ.annotation_id == summ_local_id
    assert global_summ.category_id == 1
    assert local_summ.category_id == 2


def test_remove_and_deactivate_annotations(dp_image: Image) -> None:
    """test remove_annotations and deactivate_annotation"""
    mgr = DatapointManager(service_id="svc", model_id="m")
    mgr.datapoint = dp_image
    a1 = mgr.set_image_annotation(_detection_result([0, 0, 10, 10]))
    a2 = mgr.set_image_annotation(_detection_result([20, 20, 40, 40]))
    assert a1 is not None
    assert a2 is not None

    mgr.deactivate_annotation(a1)
    active = mgr.datapoint.get_annotation()
    assert all(ann.annotation_id != a1 for ann in active)

    mgr.remove_annotations([a2])
    assert a2 not in mgr._cache_anns  # noqa: SLF001


def test_errors_assert_and_type() -> None:
    """test errors assert and type"""
    mgr = DatapointManager(service_id="svc", model_id="m")
    with pytest.raises(AssertionError, match="Pass datapoint"):
        mgr.set_image_annotation(_detection_result([0, 0, 10, 10]))

    # Proper setup
    img = Image(file_name="dummy.jpg")
    mgr.datapoint = img
    bad_dr = DetectionResult(
        box=("not", "a", "list"),  # type: ignore[arg-type]
        class_name=get_type("test_cat_1"),
        class_id=1,
        score=0.1,
        absolute_coords=True,
    )
    with pytest.raises(TypeError, match="must be of type list or np.ndarray"):
        mgr.set_image_annotation(bad_dr)
