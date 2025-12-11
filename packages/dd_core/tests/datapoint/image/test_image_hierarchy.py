# -*- coding: utf-8 -*-
# File: test_image_hierarchy.py

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
Testing Image hierarchy operations (image_ann_to_image, maybe_ann_to_sub_image, etc.)
"""

import numpy as np
from numpy import float32, ones
from pytest import mark, raises

from dd_core.datapoint import BoundingBox, Image, ImageAnnotation
from dd_core.utils.error import ImageError

from ..conftest import WhiteImage


class TestImageHierarchy:
    """Test Image hierarchical operations"""

    def test_image_ann_to_image_creates_image_attribute(self, white_image: WhiteImage):
        """image_ann_to_image creates image attribute in annotation"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=False)

        assert ann.image is not None
        assert isinstance(ann.image, Image)

    def test_image_ann_to_image_sets_correct_dimensions(self, white_image: WhiteImage):
        """image_ann_to_image sets correct dimensions for sub-image"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

        assert ann.image.width == 10
        assert ann.image.height == 4  # Intersects with image bounds

    def test_image_ann_to_image_creates_embedding(self, white_image: WhiteImage):
        """image_ann_to_image creates embedding in parent image"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=False)

        assert img.image_id in ann.image.embeddings

    def test_image_ann_to_image_crop_image_creates_pixels(self, white_image: WhiteImage):
        """image_ann_to_image with crop_image=True creates pixel data"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

        assert ann.image.image is not None
        assert ann.image.image.shape == (4, 10, 3)

    def test_image_ann_to_image_no_crop_leaves_no_pixels(self, white_image: WhiteImage):
        """image_ann_to_image with crop_image=False doesn't create pixels"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=False)

        assert ann.image.image is None

    def test_image_ann_to_image_requires_bbox(self, white_image: WhiteImage):
        """image_ann_to_image requires bounding box to be set"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)

        with raises(ImageError, match="Bounding box for image and ImageAnnotation"):
            img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=False)

    def test_image_ann_to_image_crop_requires_image_data(self, white_image: WhiteImage):
        """image_ann_to_image with crop_image=True requires image data"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.set_width_height(100, 100)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)

        with raises(ImageError, match="crop_image = True requires self.image to be not None"):
            img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

    def test_remove_image_from_lower_hierarchy_removes_sub_images(self, white_image: WhiteImage):
        """remove_image_from_lower_hierarchy removes annotation images"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

        assert ann.image is not None
        img.remove_image_from_lower_hierarchy()

        assert ann.image is None

    def test_remove_image_from_lower_hierarchy_preserves_bbox(self, white_image: WhiteImage):
        """remove_image_from_lower_hierarchy preserves bounding box"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)
        original_bbox = ann.get_bounding_box(img.image_id)

        img.remove_image_from_lower_hierarchy()

        assert ann.bounding_box is not None
        assert ann.bounding_box == original_bbox

    def test_remove_image_pixel_values_only(self, white_image: WhiteImage):
        """remove_image_from_lower_hierarchy with pixel_values_only=True"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = ones((24, 85, 3), dtype=float32)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

        img.remove_image_from_lower_hierarchy(pixel_values_only=True)

        # Image object exists but pixels are cleared
        assert ann.image is not None
        assert ann.image.image is None

    def test_get_categories_from_current_state(self, white_image: WhiteImage):
        """get_categories_from_current_state returns active category names"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann1)
        img.dump(ann2)

        categories = img.get_categories_from_current_state()

        assert "test_cat_1" in categories
        assert "test_cat_2" in categories
        assert len(categories) == 2

    def test_get_categories_excludes_inactive(self, white_image: WhiteImage):
        """get_categories_from_current_state excludes inactive annotations"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann1)
        img.dump(ann2)
        ann2.active = False

        categories = img.get_categories_from_current_state()

        assert "test_cat_1" in categories
        assert "test_cat_2" not in categories
