# -*- coding: utf-8 -*-
# File: test_structured_output.py

# Copyright 2026 Dr. Janis Meyer. All rights reserved.
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
Unit tests for `dd_core.mapper.structured_output.image_or_doc_to_structured_output`.
"""

import pytest

import shared_test_utils as stu
from dd_core.datapoint.annotation import AnnotationRef, ContainerAnnotation, ReferencePayload
from dd_core.datapoint.image import Image
from dd_core.doc import Document
from dd_core.mapper.structured_output import image_or_doc_to_structured_output
from dd_core.utils.object_types import SummaryKey


def test_image_or_doc_to_structured_output_image_plain_dict(image: Image) -> None:
    """Image input whose ContainerAnnotation.value is already a plain dict is returned unresolved."""
    plain_value = {"some_field": "some_value"}
    image.summary.dump_sub_category(
        SummaryKey.STRUCTURED_OUTPUT,
        ContainerAnnotation(category_name=SummaryKey.STRUCTURED_OUTPUT, value=plain_value),
    )

    result, img_id = image_or_doc_to_structured_output(summary_sub_category_names="structured_output")(image)

    assert result == {SummaryKey.STRUCTURED_OUTPUT: [plain_value]}
    assert img_id == image.image_id


def test_image_or_doc_to_structured_output_image_reference_payload(image: Image) -> None:
    """Image input whose ContainerAnnotation.value is a ReferencePayload is resolved via the wrapped Page."""
    word_ann = image.get_annotation(category_names="word")[0]
    image.summary.dump_sub_category(
        SummaryKey.STRUCTURED_OUTPUT,
        ContainerAnnotation(
            category_name=SummaryKey.STRUCTURED_OUTPUT,
            value=ReferencePayload(
                content={"some_word": AnnotationRef(annotation_id=word_ann.annotation_id, image_id=image.image_id)}
            ),
        ),
    )

    result, img_id = image_or_doc_to_structured_output(summary_sub_category_names="structured_output")(image)

    assert list(result[SummaryKey.STRUCTURED_OUTPUT][0].keys()) == ["some_word"]
    assert img_id == image.image_id


def test_image_or_doc_to_structured_output_document_plain_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document input whose ContainerAnnotation.value is already a plain dict is returned unresolved -
    proves the already-resolved case is handled at document level too, not only page level."""
    monkeypatch.setenv("ENABLE_DYNAMIC_OBJECT_TYPES", "True")
    doc = Document(file_name="plain", location=stu.asset_path("eval_doc").parent, compute_metadata=False)
    plain_value = {"some_field": "some_value"}
    doc.summary.dump_sub_category(
        SummaryKey.STRUCTURED_OUTPUT,
        ContainerAnnotation(category_name=SummaryKey.STRUCTURED_OUTPUT, value=plain_value),
    )

    result, doc_id = image_or_doc_to_structured_output(summary_sub_category_names="structured_output")(doc)

    assert result == {SummaryKey.STRUCTURED_OUTPUT: [plain_value]}
    assert doc_id == doc.document_id


def test_image_or_doc_to_structured_output_document_resolves_reference_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document input with a genuine ReferencePayload resolves through Document.resolve_reference_payload."""
    monkeypatch.setenv("ENABLE_DYNAMIC_OBJECT_TYPES", "True")
    doc = Document.from_json(stu.asset_path("eval_doc"))

    result, doc_id = image_or_doc_to_structured_output(summary_sub_category_names="structured_output")(doc)

    assert doc_id == doc.document_id
    assert result[SummaryKey.STRUCTURED_OUTPUT][0] == doc.structured_output


def test_image_or_doc_to_structured_output_raises_for_non_container_annotation(table_image: Image) -> None:
    """ValueError when the requested summary sub category is not a ContainerAnnotation."""
    with pytest.raises(ValueError):
        image_or_doc_to_structured_output(summary_sub_category_names="number_of_rows")(table_image)


def test_image_or_doc_to_structured_output_raises_without_summary_sub_category_names(image: Image) -> None:
    """ValueError when summary_sub_category_names is not given - there is no default."""
    with pytest.raises(ValueError):
        image_or_doc_to_structured_output()(image)