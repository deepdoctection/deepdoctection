# -*- coding: utf-8 -*-
# File: test_image_annotations.py

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
Testing Image annotation dump and retrieval operations
"""

from pytest import raises

from dd_core.datapoint import BoundingBox, Image, ImageAnnotation
from dd_core.utils.error import ImageError

from ..conftest import WhiteImage


class TestImageAnnotations:
    """Test Image annotation operations"""

    def test_dump_assigns_annotation_id(self, white_image: WhiteImage) -> None:
        """dump() assigns annotation_id if not set"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )

        assert ann._annotation_id is None
        img.dump(ann)
        assert ann.annotation_id is not None

    def test_dump_tracks_annotation_ids(self, white_image: WhiteImage) -> None:
        """dump() tracks annotation_id in internal list"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann)
        assert ann.annotation_id in img._annotation_ids

    def test_dump_rejects_duplicate_annotation(self, white_image: WhiteImage) -> None:
        """dump() raises error for duplicate annotation"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann)

        with raises(ImageError, match="Cannot dump annotation with existing id"):
            img.dump(ann)

    def test_dump_multiple_annotations(self, white_image: WhiteImage) -> None:
        """Multiple annotations can be dumped"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=50, uly=50, width=30, height=30, absolute_coords=True),
        )
        img.dump(ann1)
        img.dump(ann2)

        assert len(img.annotations) == 2
        assert len(img._annotation_ids) == 2

    def test_get_annotation_returns_all_by_default(self, white_image: WhiteImage) -> None:
        """get_annotation() returns all active annotations by default"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=50, uly=50, width=30, height=30, absolute_coords=True),
        )
        img.dump(ann1)
        img.dump(ann2)

        result = img.get_annotation()
        assert len(result) == 2

    def test_get_annotation_filters_by_category_name(self, white_image: WhiteImage) -> None:
        """get_annotation() filters by category_name"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=50, uly=50, width=30, height=30, absolute_coords=True),
        )

        img.dump(ann1)
        img.dump(ann2)

        result = img.get_annotation(category_names="test_cat_1")
        assert len(result) == 1
        assert result[0].category_name.value == "test_cat_1"  # type: ignore

    def test_get_annotation_filters_by_multiple_category_names(self, white_image: WhiteImage) -> None:
        """get_annotation() filters by multiple category names"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
        )
        ann3 = ImageAnnotation(
            category_name="test_cat_3",
            bounding_box=BoundingBox(ulx=50, uly=50, width=20, height=20, absolute_coords=True),
        )

        img.dump(ann1)
        img.dump(ann2)
        img.dump(ann3)

        result = img.get_annotation(category_names=["test_cat_1", "test_cat_2"])
        assert len(result) == 2

    def test_get_annotation_filters_by_annotation_id(self, white_image: WhiteImage) -> None:
        """get_annotation() filters by annotation_id"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=50, uly=50, width=30, height=30, absolute_coords=True),
        )
        img.dump(ann1)
        img.dump(ann2)

        result = img.get_annotation(annotation_ids=ann1.annotation_id)
        assert len(result) == 1
        assert result[0].annotation_id == ann1.annotation_id

    def test_get_annotation_filters_by_service_id(self, white_image: WhiteImage) -> None:
        """get_annotation() filters by service_id"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
            service_id="service_a",
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=50, uly=50, width=30, height=30, absolute_coords=True),
            service_id="service_b",
        )
        img.dump(ann1)
        img.dump(ann2)

        result = img.get_annotation(service_ids="service_a")
        assert len(result) == 1
        assert result[0].service_id == "service_a"

    def test_get_annotation_ignores_inactive_by_default(self, white_image: WhiteImage) -> None:
        """get_annotation() ignores inactive annotations by default"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann)
        ann.deactivate()

        result = img.get_annotation()
        assert len(result) == 0

    def test_get_annotation_can_include_inactive(self, white_image: WhiteImage) -> None:
        """get_annotation() can include inactive annotations"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann)
        ann.deactivate()

        result = img.get_annotation(ignore_inactive=False)
        assert len(result) == 1

    def test_get_annotation_combined_filters(self, white_image: WhiteImage) -> None:
        """get_annotation() applies multiple filters correctly"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
            service_id="service_a",
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
            service_id="service_b",
        )
        ann3 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=50, uly=50, width=20, height=20, absolute_coords=True),
            service_id="service_a",
        )
        img.dump(ann1)
        img.dump(ann2)
        img.dump(ann3)

        # Filter by both category and service_id
        result = img.get_annotation(category_names="test_cat_1", service_ids="service_a")
        assert len(result) == 1
        assert result[0].annotation_id == ann1.annotation_id

    def test_annotation_id_uniqueness(self, white_image: WhiteImage) -> None:
        """Each annotation gets a unique annotation_id"""
        img = Image(file_name=white_image.file_name)
        annotations = []

        for i in range(5):
            ann = ImageAnnotation(
                category_name="test_cat_1",
                bounding_box=BoundingBox(ulx=i * 10, uly=i * 10, width=20, height=20, absolute_coords=True),
            )
            img.dump(ann)
            annotations.append(ann)

        annotation_ids = [ann.annotation_id for ann in annotations]
        assert len(set(annotation_ids)) == 5  # All IDs should be unique
