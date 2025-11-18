# -*- coding: utf-8 -*-
# File: test_image_removal.py

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
Testing Image remove operations
"""

from pytest import mark

from dd_core.datapoint import BoundingBox, Image, ImageAnnotation

import shared_test_utils as stu


class TestImageRemoval:
    """Test Image annotation removal operations"""

    @staticmethod
    def test_remove_by_annotation_id_removes_annotation(white_image: stu.WhiteImage):
        """remove() by annotation_id removes annotation"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann)

        img.remove(annotation_ids=ann.annotation_id)

        result = img.get_annotation(annotation_ids=ann.annotation_id)
        assert len(result) == 0

    @staticmethod
    def test_remove_by_annotation_id_list(white_image: stu.WhiteImage):
        """remove() by list of annotation_ids removes all"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann1)
        img.dump(ann2)

        img.remove(annotation_ids=[ann1.annotation_id, ann2.annotation_id])

        assert len(img.get_annotation()) == 0

    @staticmethod
    def test_remove_by_service_id_removes_matching(white_image: stu.WhiteImage):
        """remove() by service_id removes matching annotations"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
            service_id="service_a"
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
            service_id="service_b"
        )
        img.dump(ann1)
        img.dump(ann2)

        img.remove(service_ids="service_a")

        result = img.get_annotation()
        assert len(result) == 1
        assert result[0].service_id == "service_b"

    @staticmethod
    def test_remove_by_multiple_service_ids(white_image: stu.WhiteImage):
        """remove() by list of service_ids removes all matching"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
            service_id="service_a"
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
            service_id="service_b"
        )
        ann3 = ImageAnnotation(
            category_name="test_cat_3",
            bounding_box=BoundingBox(ulx=50, uly=50, width=20, height=20, absolute_coords=True),
            service_id="service_c"
        )
        img.dump(ann1)
        img.dump(ann2)
        img.dump(ann3)

        img.remove(service_ids=["service_a", "service_b"])

        result = img.get_annotation()
        assert len(result) == 1
        assert result[0].service_id == "service_c"

    @staticmethod
    def test_remove_updates_annotation_ids_list(white_image: stu.WhiteImage):
        """remove() updates internal _annotation_ids list"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann)
        assert ann.annotation_id in img._annotation_ids

        img.remove(annotation_ids=ann.annotation_id)

        assert ann.annotation_id not in img._annotation_ids

    @staticmethod
    def test_remove_preserves_other_annotations(white_image: stu.WhiteImage):
        """remove() preserves annotations not matching criteria"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True)
        )
        ann3 = ImageAnnotation(
            category_name="test_cat_3",
            bounding_box=BoundingBox(ulx=50, uly=50, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann1)
        img.dump(ann2)
        img.dump(ann3)

        img.remove(annotation_ids=ann2.annotation_id)

        result = img.get_annotation()
        assert len(result) == 2
        result_ids = [r.annotation_id for r in result]
        assert ann1.annotation_id in result_ids
        assert ann3.annotation_id in result_ids

    @staticmethod
    def test_get_service_id_to_annotation_id_mapping(white_image: stu.WhiteImage):
        """get_service_id_to_annotation_id returns correct mapping"""
        img = Image(file_name=white_image.file_name)
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
            service_id="service_a"
        )
        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=20, height=20, absolute_coords=True),
            service_id="service_a"
        )
        ann3 = ImageAnnotation(
            category_name="test_cat_3",
            bounding_box=BoundingBox(ulx=50, uly=50, width=20, height=20, absolute_coords=True),
            service_id="service_b"
        )
        img.dump(ann1)
        img.dump(ann2)
        img.dump(ann3)

        mapping = img.get_service_id_to_annotation_id()

        assert "service_a" in mapping
        assert "service_b" in mapping
        assert len(mapping["service_a"]) == 2
        assert len(mapping["service_b"]) == 1

    @staticmethod
    def test_get_annotation_id_to_annotation_maps(white_image: stu.WhiteImage):
        """get_annotation_id_to_annotation_maps returns correct structure"""
        img = Image(file_name=white_image.file_name)
        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann)

        maps = img.get_annotation_id_to_annotation_maps()

        assert ann.annotation_id in maps
        assert len(maps[ann.annotation_id]) > 0

