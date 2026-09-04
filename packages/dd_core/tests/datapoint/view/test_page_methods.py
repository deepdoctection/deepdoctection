# -*- coding: utf-8 -*-
# File: test_page_methods.py

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
Testing Page class methods
"""

import pytest

from dd_core.datapoint.annotation import AnnotationRef, ContainerAnnotation, ReferencePayload
from dd_core.datapoint.view import Page
from dd_core.utils.error import AnnotationError
from dd_core.utils.object_types import SummaryKey


class TestPageMethods:
    """Test Page class methods"""

    def test_get_layout_context_returns_list(self, page: Page) -> None:
        """get_layout_context() returns a list"""
        layouts = page.layouts
        if layouts and len(layouts) > 0:
            layout = layouts[0]
            if layout.reading_order is not None:
                context = page.get_layout_context(layout.annotation_id, context_size=1)
                assert {context.reading_order for context in context} == {1, 2}

    def test_get_layout_context_includes_target(self, page: Page) -> None:
        """get_layout_context() includes the target annotation"""
        layouts = page.layouts
        ordered = [l for l in layouts if l.reading_order is not None]
        if ordered:
            layout = ordered[0]
            context = page.get_layout_context(layout.annotation_id, context_size=1)
            ann_ids = [c.annotation_id for c in context]
            assert layout.annotation_id in ann_ids

    def test_get_layout_context_respects_context_size(self, page: Page) -> None:
        """get_layout_context() respects context_size parameter"""
        layouts = page.layouts
        ordered = [l for l in layouts if l.reading_order is not None]
        if len(ordered) >= 3:
            # Get middle element
            middle_idx = len(ordered) // 2
            layout = ordered[middle_idx]
            context = page.get_layout_context(layout.annotation_id, context_size=1)
            # Should return at most 3 items (1 before, target, 1 after)
            assert len(context) <= 3

    def test_save_returns_dict_when_dry(self, page: Page) -> None:
        """save() returns dict when dry=True"""
        result = page.save(dry=True)
        assert isinstance(result, dict)


class TestPageResolveReferencePayload:
    """Test resolving a `ReferencePayload` on page level"""

    def test_resolve_reference_payload_returns_text(self, page: Page) -> None:
        """resolve_reference_payload resolves AnnotationRef leaves to the text of the page annotation"""
        word = page.words[0]

        payload = ReferencePayload(
            content={"some_word": AnnotationRef(annotation_id=word.annotation_id, image_id=page.image_id)}
        )

        assert page.resolve_reference_payload(payload) == {"some_word": word.characters}

    def test_resolve_reference_payload_without_image_id(self, page: Page) -> None:
        """An AnnotationRef without image_id is resolved on the page itself"""
        word = page.words[0]

        payload = ReferencePayload(
            content={"some_word": AnnotationRef(annotation_id=word.annotation_id, image_id=None)}
        )

        assert page.resolve_reference_payload(payload) == {"some_word": word.characters}

    def test_resolve_reference_payload_of_foreign_image_raises(self, page: Page) -> None:
        """An AnnotationRef pointing to another image cannot be resolved on this page"""
        word = page.words[0]

        payload = ReferencePayload(
            content={"some_word": AnnotationRef(annotation_id=word.annotation_id, image_id="some_other_image")}
        )

        with pytest.raises(AnnotationError):
            page.resolve_reference_payload(payload)

    def test_structured_output_is_empty_without_summary_sub_category(self, page: Page) -> None:
        """structured_output returns an empty dict, if the page summary has no structured_output"""
        assert "structured_output" not in page.summary.sub_categories
        assert page.structured_output == {}

    def test_structured_output_resolves_summary_payload(self, page: Page) -> None:
        """structured_output resolves the ReferencePayload dumped into the page summary"""
        word = page.words[0]
        page.summary.dump_sub_category(
            SummaryKey.STRUCTURED_OUTPUT,
            ContainerAnnotation(
                category_name=SummaryKey.STRUCTURED_OUTPUT,
                value=ReferencePayload(
                    content={"some_word": AnnotationRef(annotation_id=word.annotation_id, image_id=page.image_id)}
                ),
            ),
        )

        assert page.structured_output == {"some_word": word.characters}

    def test_resolve_mixed_payload_passes_non_reference_leaves_through(self, page: Page) -> None:
        """Only AnnotationRef leaves are resolved, every other leaf is returned unchanged"""
        word = page.words[0]

        payload = ReferencePayload(
            content={
                "quoted": [AnnotationRef(annotation_id=word.annotation_id, image_id=page.image_id)],
                "quoted_unmatched": [],
                "quoted_null": None,
                "inferred_str": "not written on the page",
                "inferred_int": 42,
                "inferred_bool": True,
                "inferred_obj": {"flag": False, "arr": [1, "two", None]},
            }
        )

        assert page.resolve_reference_payload(payload) == {
            "quoted": [word.characters],
            "quoted_unmatched": [],
            "quoted_null": None,
            "inferred_str": "not written on the page",
            "inferred_int": 42,
            "inferred_bool": True,
            "inferred_obj": {"flag": False, "arr": [1, "two", None]},
        }

    def test_structured_output_resolves_mixed_summary_payload(self, page: Page) -> None:
        """The same holds for a mixed payload dumped into the page summary"""
        word = page.words[0]
        page.summary.dump_sub_category(
            SummaryKey.STRUCTURED_OUTPUT,
            ContainerAnnotation(
                category_name=SummaryKey.STRUCTURED_OUTPUT,
                value=ReferencePayload(
                    content={
                        "quoted": [AnnotationRef(annotation_id=word.annotation_id, image_id=page.image_id)],
                        "inferred": {"is_signed": True, "pages": 3},
                    }
                ),
            ),
        )

        assert page.structured_output == {"quoted": [word.characters], "inferred": {"is_signed": True, "pages": 3}}
