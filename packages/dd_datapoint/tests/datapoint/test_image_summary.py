# -*- coding: utf-8 -*-
# File: test_image_summary.py

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
Testing Image summary operations
"""

from pytest import mark, raises

from dd_datapoint.datapoint import CategoryAnnotation, Image
from dd_datapoint.utils.error import ImageError
from dd_datapoint.utils.object_types import SummaryType


class TestImageSummary:
    """Test Image summary property"""

    @staticmethod
    @mark.basic
    def test_summary_created_on_first_access():
        """Summary is created on first access"""
        img = Image(file_name="test.png")

        summary = img.summary

        assert summary is not None
        assert isinstance(summary, CategoryAnnotation)

    @staticmethod
    @mark.basic
    def test_summary_has_correct_category():
        """Summary has SUMMARY category"""
        img = Image(file_name="test.png")

        summary = img.summary

        assert summary.category_name == SummaryType.SUMMARY

    @staticmethod
    @mark.basic
    def test_summary_has_annotation_id():
        """Summary gets annotation_id on creation"""
        img = Image(file_name="test.png")

        summary = img.summary

        assert summary.annotation_id is not None

    @staticmethod
    @mark.basic
    def test_summary_returns_same_instance():
        """Multiple accesses return same summary instance"""
        img = Image(file_name="test.png")

        summary1 = img.summary
        summary2 = img.summary

        assert summary1 is summary2

    @staticmethod
    @mark.basic
    def test_summary_can_be_set():
        """Summary can be set with custom CategoryAnnotation"""
        img = Image(file_name="test.png")
        custom_summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)

        img.summary = custom_summary

        assert img.summary is custom_summary

    @staticmethod
    @mark.basic
    def test_summary_sets_annotation_id_if_missing():
        """Setting summary assigns annotation_id if missing"""
        img = Image(file_name="test.png")
        custom_summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)

        assert custom_summary._annotation_id is None
        img.summary = custom_summary
        assert custom_summary.annotation_id is not None

    @staticmethod
    @mark.basic
    def test_summary_cannot_be_reset():
        """Summary cannot be reassigned once set"""
        img = Image(file_name="test.png")
        _ = img.summary  # Access to create

        new_summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)

        with raises(ImageError, match="Image.summary already defined and cannot be reset"):
            img.summary = new_summary

    @staticmethod
    @mark.basic
    def test_summary_preserves_annotation_id():
        """Summary with existing annotation_id preserves it"""
        img = Image(file_name="test.png")
        custom_summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)
        custom_id = "custom-annotation-id"
        custom_summary.annotation_id = custom_id

        img.summary = custom_summary

        assert img.summary.annotation_id == custom_id

    @staticmethod
    @mark.basic
    def test_summary_annotation_id_deterministic():
        """Summary annotation_id is deterministic based on image_id"""
        img1 = Image(file_name="test.png", location="/path")
        img2 = Image(file_name="test.png", location="/path")

        summary1 = img1.summary
        summary2 = img2.summary

        # Same image_id should produce same summary annotation_id
        assert summary1.annotation_id == summary2.annotation_id

