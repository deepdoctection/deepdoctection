# -*- coding: utf-8 -*-
# File: test_image_annotation_base_view.py

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
Testing ImageAnnotationBaseView class methods
"""

from dd_datapoint.datapoint.view import Page
from dd_datapoint.utils.error import AnnotationError


class TestImageAnnotationBaseView:
    """Test ImageAnnotationBaseView functionality"""


    def test_category_name_property(self, page: Page):
        """Layout has category_name property"""

        layouts = page.layouts
        if layouts:
            assert hasattr(layouts[0], "category_name")

    def test_annotation_id_property(self, page: Page):
        """Layout has annotation_id property"""

        layouts = page.layouts
        if layouts:
            assert isinstance(layouts[0].annotation_id, str)

    def test_base_page_property(self, page: Page):
        """Layout has base_page attribute"""

        layouts = page.layouts
        if layouts:
            assert hasattr(layouts[0], "base_page")
            assert layouts[0].base_page == page

    def test_service_id_property(self, page: Page):
        """service_id property exists and can be None or str"""
        layouts = page.layouts
        if layouts:
            service_id = layouts[0].service_id
            assert service_id is None or isinstance(service_id, str)

    def test_model_id_property(self, page: Page):
        """model_id property exists and can be None or str"""
        layouts = page.layouts
        if layouts:
            model_id = layouts[0].model_id
            assert model_id is None or isinstance(model_id, str)

    def test_session_id_property(self, page: Page):
        """session_id property exists and can be None or str"""
        layouts = page.layouts
        if layouts:
            session_id = layouts[0].session_id
            assert session_id is None or isinstance(session_id, str)

    def test_b64_image_returns_string_or_none(self, page: Page):
        """b64_image property returns string or None"""
        layouts = page.layouts
        if layouts:
            b64_img = layouts[0].b64_image
            assert b64_img is None or isinstance(b64_img, str)

    def test_getattr_with_unregistered_attribute_raises_error(self, page: Page):
        """__getattr__ raises error for unregistered attributes"""
        layouts = page.layouts
        if layouts:
            try:
                _ = layouts[0].non_existent_attribute_xyz
                assert False, "Expected AttributeError"
            except AnnotationError as e:
                # Should raise some kind of error
                assert True


