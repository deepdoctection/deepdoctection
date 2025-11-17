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

from dd_datapoint.datapoint import CategoryAnnotation, Image
from dd_datapoint.utils.object_types import SummaryType


class TestImageSummary:
    """Test Image summary property"""

    @staticmethod
    def test_summary_created_on_first_access():
        """Summary is created on first access"""
        img = Image(file_name="test.png")

        summary = img.summary

        assert summary is not None
        assert isinstance(summary, CategoryAnnotation)

    @staticmethod
    def test_summary_has_correct_category():
        """Summary has SUMMARY category"""
        img = Image(file_name="test.png")

        summary = img.summary

        assert summary.category_name == SummaryType.SUMMARY

    @staticmethod
    def test_summary_has_annotation_id():
        """Summary gets annotation_id on creation"""
        img = Image(file_name="test.png")

        summary = img.summary

        assert summary.annotation_id is not None
