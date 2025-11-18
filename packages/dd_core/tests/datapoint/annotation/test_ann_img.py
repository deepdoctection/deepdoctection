# -*- coding: utf-8 -*-
# File: test_img_ann.py

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

from dd_core.datapoint.annotation import (
    ImageAnnotation,
)
from dd_core.datapoint.box import BoundingBox
from dd_core.utils.error import AnnotationError
from dd_core.utils.object_types import get_type


class TestImageAnnotation:
    """Tests for ImageAnnotation class"""

    def test_image_annotation_creation_basic(self):
        """Test basic ImageAnnotation creation"""
        img_ann = ImageAnnotation(
            category_name="test_cat_1",
            category_id=1,
            bounding_box=BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        )
        assert img_ann.category_name == get_type("test_cat_1")
        assert img_ann.bounding_box is not None

    def test_image_annotation_bounding_box_from_dict(self):
        """Test creating ImageAnnotation with bounding_box from dict"""
        data = {
            "category_name": "test_cat_1",
            "bounding_box": {
                "ulx": 10,
                "uly": 20,
                "width": 30,
                "height": 40,
                "absolute_coords": True
            }
        }
        img_ann = ImageAnnotation(**data)
        assert img_ann.bounding_box.ulx == 10
        assert img_ann.bounding_box.uly == 20

    def test_image_annotation_get_bounding_box_basic(self):
        """Test getting bounding box from ImageAnnotation"""
        bbox = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        img_ann = ImageAnnotation(category_name="test_cat_1", bounding_box=bbox)
        retrieved_bbox = img_ann.get_bounding_box()
        assert retrieved_bbox == bbox

    def test_image_annotation_get_bounding_box_none_raises_error(self):
        """Test that getting bounding box when None raises error"""
        img_ann = ImageAnnotation(category_name="test_cat_1", external_id="test")
        with pytest.raises(AnnotationError):
            img_ann.get_bounding_box()


    def test_image_annotation_annotation_id_determinism(self):
        """Test that ImageAnnotation annotation_id is deterministic based on bounding_box"""
        bbox1 = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        bbox2 = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        img_ann1 = ImageAnnotation(category_name="test_cat_1", bounding_box=bbox1)
        img_ann2 = ImageAnnotation(category_name="test_cat_1", bounding_box=bbox2)
        ann_id_1 = ImageAnnotation.set_annotation_id(img_ann1)
        ann_id_2 = ImageAnnotation.set_annotation_id(img_ann2)
        assert ann_id_1 == ann_id_2

    def test_image_annotation_annotation_id_different_boxes(self):
        """Test that different bounding boxes produce different annotation_ids"""
        bbox1 = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        bbox2 = BoundingBox(ulx=15, uly=25, width=35, height=45, absolute_coords=True)
        img_ann1 = ImageAnnotation(category_name="test_cat_1", bounding_box=bbox1)
        img_ann2 = ImageAnnotation(category_name="test_cat_1", bounding_box=bbox2)
        ann_id_1 = ImageAnnotation.set_annotation_id(img_ann1)
        ann_id_2 = ImageAnnotation.set_annotation_id(img_ann2)
        assert ann_id_1 != ann_id_2


class TestBoundingBoxIntegration:
    """Tests for BoundingBox integration with ImageAnnotation"""


    def test_bounding_box_from_dict_coercion(self):
        """Test BoundingBox coercion from dict in ImageAnnotation"""
        bbox_dict = {
            "ulx": 5.0,
            "uly": 10.0,
            "width": 20.0,
            "height": 30.0,
            "absolute_coords": True
        }
        img_ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=bbox_dict
        )
        assert isinstance(img_ann.bounding_box, BoundingBox)
        assert img_ann.bounding_box.ulx == 5
        assert img_ann.bounding_box.uly == 10