# -*- coding: utf-8 -*-
# File: test_doc.py

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
Testing module for doc.py
"""

import os
from pathlib import Path

import pytest

from dd_core.datapoint.annotation import (
    DEFAULT_CATEGORY_ID,
    AnnotationRef,
    CategoryAnnotation,
    ContainerAnnotation,
    ReferencePayload,
)
from dd_core.datapoint.image import Image
from dd_core.datapoint.view import Page
from dd_core.doc import Document, PageReference, re_assign_document_summary_cat_ids
from dd_core.utils import file_utils as fu
from dd_core.utils.object_types import DocumentFileLabel, DocumentKey, DocumentLabel, SummaryKey, get_type

from .conftest import ObjectTestType  # pylint: disable=E0611


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_reports_number_of_pages(pdf_file_path_two_pages: Path) -> None:
    """test that pdf_reports_number_of_pages works"""

    doc = Document(location=pdf_file_path_two_pages)
    assert doc.number_of_pages == 2


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_get_page_reference_returns_valid_objects(pdf_file_path_two_pages: Path) -> None:
    """test that PageReference has all attributes filled"""

    doc = Document(location=pdf_file_path_two_pages)
    ref = doc.get_page_reference(1)

    assert isinstance(ref, PageReference)
    assert ref.image_id == "682a88af-630a-3160-b89b-4d8c2e0472f5"
    assert ref.source_path == os.fspath(pdf_file_path_two_pages)


def test_from_json_restores_internal_structures(sample_document_json: Path) -> None:
    """document returns internal structures"""
    doc = Document.from_json(sample_document_json)

    assert doc.document_id == "108e9e00-58cd-3c19-a900-38177f66fd87"
    assert isinstance(doc.get_page_reference(1), PageReference)


def test_len_equals_number_of_pages(sample_document_json: Path) -> None:
    """test document length equals number of pages"""
    doc = Document.from_json(sample_document_json)
    assert len(doc) == doc.number_of_pages


def test_doc_returns_structured_output(sample_document_json: Path) -> None:
    """test returns structured outputs"""
    doc = Document.from_json(sample_document_json)
    structured_output = doc.structured_output
    assert len(structured_output["buyer"]["buyerReference"]) == 11
    assert structured_output["buyer"]["contact"]["contactName"] is None


def test_resolve_reference_payload_returns_text(sample_document_json: Path) -> None:
    """resolve_reference_payload resolves AnnotationRef leaves to their text or characters value"""
    doc = Document.from_json(sample_document_json)

    reference = ReferencePayload(
        content={
            "some_cell": AnnotationRef(
                annotation_id="54339b9d-1571-3211-abe1-898aed89bfad",
                image_id="72a354a5-6a4b-3240-8310-4cfd9dd2c264",
            ),
            "some_word": AnnotationRef(
                annotation_id="b61e3b00-85cc-3cb4-9cef-cc00f54b1823",
                image_id="72a354a5-6a4b-3240-8310-4cfd9dd2c264",
            ),
        }
    )

    out = doc.resolve_reference_payload(reference)

    assert out == {"some_cell": "78", "some_word": "Kunden-No"}


def test_get_page_and_get_image_return_types(sample_document_json: Path) -> None:
    """test get page by given page number"""
    doc = Document.from_json(sample_document_json)
    page1 = doc.get_page(1)
    assert isinstance(page1, Page)


def test_get_page_by_image_id_returns_page(sample_document_json: Path) -> None:
    """test get page by given image id"""
    image_id = "7e154965-1250-3f4f-b1c2-a6e822f0aaa5"
    doc = Document.from_json(sample_document_json)
    page0 = doc.get_page(image_id=image_id)
    assert page0.image_id == image_id


def test_get_image_dataflow(sample_document_json: Path) -> None:
    """test get document dataflow"""
    doc = Document.from_json(sample_document_json)
    df = doc.get_image_dataflow()
    df.reset_state()
    assert len(list(df)) == 6


def test_set_image_updates_references_and_images() -> None:
    """
    Create a plain Document without metadata computation, create an Image,
    add it via set_image and verify internal mappings and retrieval.
    """
    doc = Document(file_name="plain", location=Path(), compute_metadata=False)
    img = Image(file_name="test.png", location="/fake/location", page_number=5)

    doc.set_image(img, page_number=1)

    assert img.image_id in doc._images
    assert doc._images[img.image_id] is img

    ref = doc.get_page_reference(1)
    assert isinstance(ref, PageReference)

    fetched = doc.get_image(image_id=img.image_id)
    assert fetched is img


def test_save_dry_returns_export_dict_with_expected_keys(sample_document_json: Path) -> None:
    """test save dry returns export dict"""
    doc = Document.from_json(sample_document_json)
    exported = doc.save(dry=True)
    assert isinstance(exported, dict)


def test_save_load_round_trip_preserves_reference_payload(sample_document_json: Path, tmp_path: Path) -> None:
    """Document save -> from_json must preserve ReferencePayload values so structured_output still resolves.

    Before the ContainerAnnotation.value field serializer fix, save() stripped the ``_ref_type`` markers and
    the value came back as a plain dict, breaking resolve_reference_payload/structured_output on reload.
    """
    doc = Document.from_json(sample_document_json)
    expected = doc.structured_output
    assert expected  # sanity: the sample document carries a ReferencePayload summary value

    # document-level summary value must be a ReferencePayload in memory
    doc_summary_value = doc.summary.get_sub_category(get_type("structured_output")).value  # type:ignore
    assert isinstance(doc_summary_value, ReferencePayload)

    saved_path = doc.save(path=tmp_path)
    assert isinstance(saved_path, str)

    reloaded = Document.from_json(saved_path)

    # value survives the round trip as a ReferencePayload (not a plain dict)
    reloaded_value = reloaded.summary.get_sub_category(get_type("structured_output")).value  # type:ignore
    assert isinstance(reloaded_value, ReferencePayload)

    # AnnotationRef leaves are reconstructed as AnnotationRef instances, not plain dicts
    def _has_annotation_ref_leaf(node: object) -> bool:
        if isinstance(node, AnnotationRef):
            return True
        if isinstance(node, dict):
            return any(_has_annotation_ref_leaf(v) for v in node.values())
        if isinstance(node, list):
            return any(_has_annotation_ref_leaf(v) for v in node)
        return False

    assert _has_annotation_ref_leaf(reloaded_value.content)
    assert reloaded.structured_output == expected


def test_document_extras_without_persist_do_not_survive_save_load(tmp_path: Path) -> None:
    """Document-level extras registered without persist=True stay transient across save -> from_json."""
    doc = Document(file_name="plain", location=Path(), compute_metadata=False)
    doc.configure_extras("scratch", "str")
    doc.dump_extra("scratch", "not kept")

    exported = doc.save(dry=True)
    assert isinstance(exported, dict)
    assert "_extras" not in exported

    saved_path = doc.save(path=tmp_path)
    assert isinstance(saved_path, str)
    reloaded = Document.from_json(saved_path)

    with pytest.raises(AttributeError):
        _ = reloaded.extras.scratch


def test_document_extras_with_persist_survive_save_load(tmp_path: Path) -> None:
    """Document-level extras registered with persist=True survive a full save -> from_json round trip."""
    doc = Document(file_name="plain", location=Path(), compute_metadata=False)
    doc.configure_extras("message", "str", persist=True)
    doc.dump_extra("message", "e-mail body text")

    saved_path = doc.save(path=tmp_path)
    assert isinstance(saved_path, str)
    reloaded = Document.from_json(saved_path)

    assert reloaded.extras.message == "e-mail body text"


def test_image_persisted_extras_survive_document_save_load(tmp_path: Path) -> None:
    """A persist=True extras key configured on a page Image survives a Document save -> from_json round trip."""
    doc = Document(file_name="plain", location=Path(), compute_metadata=False)
    img = Image(file_name="test.png", location="/fake/location", page_number=1)
    img.configure_extras("subject", "str", persist=True)
    img.dump_extra("subject", "Re: invoice")
    img.configure_extras("scratch", "str")
    img.dump_extra("scratch", "not kept")
    doc.set_image(img, page_number=1)

    saved_path = doc.save(path=tmp_path)
    assert isinstance(saved_path, str)
    reloaded = Document.from_json(saved_path)

    reloaded_img = reloaded.get_image(image_id=img.image_id)
    assert reloaded_img.extras.subject == "Re: invoice"
    with pytest.raises(AttributeError):
        _ = reloaded_img.extras.scratch


def test_get_annotation_id_with_given_image_id(sample_document_json: Path) -> None:
    """test get annotation id with given image id"""
    doc = Document.from_json(sample_document_json)
    text = doc.get_annotation(image_id="7e154965-1250-3f4f-b1c2-a6e822f0aaa5", category_names="table")
    page = doc.get_page(image_id="7e154965-1250-3f4f-b1c2-a6e822f0aaa5")
    text_from_page = page.get_annotation(category_names="table")
    assert len(text) == len(text_from_page)


def test_export_annotation_with_given_annotation_id(sample_document_json: Path) -> None:
    """test export annotation with given annotation id"""
    doc = Document.from_json(sample_document_json)
    # exporting CategoryAnnotation(annotation_id='c7d70ce6-d01d-3fbe-9b99-7f488554c538') on document.summary -level
    # and CategoryAnnotation(annotation_id='f6f2b661-1e9c-38be-be7d-213d125f5558') as sub category of an ImageAnnotation
    output = doc.export_annotations(
        annotation_ids=["c7d70ce6-d01d-3fbe-9b99-7f488554c538", "f6f2b661-1e9c-38be-be7d-213d125f5558"]
    )
    assert len(output) == 2

    assert len(output["c7d70ce6-d01d-3fbe-9b99-7f488554c538"][0]) == 1
    assert isinstance(output["c7d70ce6-d01d-3fbe-9b99-7f488554c538"][1], ContainerAnnotation)
    assert output["c7d70ce6-d01d-3fbe-9b99-7f488554c538"][1].annotation_id == "c7d70ce6-d01d-3fbe-9b99-7f488554c538"

    assert len(output["f6f2b661-1e9c-38be-be7d-213d125f5558"][0]) == 1
    assert isinstance(output["f6f2b661-1e9c-38be-be7d-213d125f5558"][1], CategoryAnnotation)
    assert output["f6f2b661-1e9c-38be-be7d-213d125f5558"][1].annotation_id == "f6f2b661-1e9c-38be-be7d-213d125f5558"

    annotation = doc.get_annotation(
        image_id="7e154965-1250-3f4f-b1c2-a6e822f0aaa5", annotation_ids="518264e3-98a8-350f-9e01-4344e35937f2"
    )[0]
    assert get_type("reading_order") not in annotation.sub_categories


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_first_page_number_is_1(pdf_file_path_two_pages: Path) -> None:
    """first page of a PDF document has page_number == 1"""
    doc = Document(location=pdf_file_path_two_pages)
    ref = doc.get_page_reference(1)
    assert ref.page_number == 1


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_first_page_file_name_suffix_is_1(pdf_file_path_two_pages: Path) -> None:
    """first PDF page file_name ends with _1"""
    doc = Document(location=pdf_file_path_two_pages)
    img = doc.get_image(page_number=1)
    assert img.file_name.endswith("_1.pdf")


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_get_image_page_number_1_returns_first_page(pdf_file_path_two_pages: Path) -> None:
    """get_image(page_number=1) returns the first page image"""
    doc = Document(location=pdf_file_path_two_pages)
    img = doc.get_image(page_number=1)
    assert img.page_number == 1


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_get_image_page_number_0_raises(pdf_file_path_two_pages: Path) -> None:
    """get_image(page_number=0) raises IndexError"""
    doc = Document(location=pdf_file_path_two_pages)
    with pytest.raises(IndexError):
        doc.get_image(page_number=0)


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_get_image_out_of_range_raises(pdf_file_path_two_pages: Path) -> None:
    """get_image(page_number=N+1) raises IndexError"""
    doc = Document(location=pdf_file_path_two_pages)
    with pytest.raises(IndexError):
        doc.get_image(page_number=doc.number_of_pages + 1)


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_iteration_yields_pages_in_order(pdf_file_path_two_pages: Path) -> None:
    """iterating a PDF Document yields N pages with page_numbers 1..N in order"""

    doc = Document(location=pdf_file_path_two_pages)
    pages = list(doc)
    assert len(pages) == doc.number_of_pages
    for i, page in enumerate(pages, start=1):
        assert isinstance(page, Page)
        assert page.page_number == i


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_images_dict_non_empty_after_init(pdf_file_path_two_pages: Path) -> None:
    """_images is populated after Document init for a PDF"""
    doc = Document(location=pdf_file_path_two_pages)
    assert len(doc._images) > 0


def test_document_get_attribute_names_contains_defaults() -> None:
    """Document.get_attribute_names returns number_of_pages/structured_output/document summary keys"""
    doc = Document(compute_metadata=False)
    attr_names = doc.get_attribute_names()
    assert "number_of_pages" in attr_names
    assert "structured_output" in attr_names
    assert SummaryKey.DOCUMENT_SUMMARY.value in attr_names
    assert SummaryKey.DOCUMENT_MAPPING.value in attr_names


def test_document_get_attribute_names_includes_custom_summary_sub_category() -> None:
    """A custom key dumped into doc.summary is picked up by get_attribute_names"""
    doc = Document(compute_metadata=False)
    doc.summary.dump_sub_category(
        ObjectTestType.SUMMARY_1, CategoryAnnotation(category_name=ObjectTestType.SUMMARY_1, category_id=1)
    )
    assert ObjectTestType.SUMMARY_1.value in doc.get_attribute_names()


def test_document_custom_summary_sub_category_accessible_via_getattr() -> None:
    """doc.my_custom_key resolves to the category_id of a custom summary sub category"""
    doc = Document(compute_metadata=False)
    doc.summary.dump_sub_category(
        ObjectTestType.SUMMARY_1, CategoryAnnotation(category_name=ObjectTestType.SUMMARY_1, category_id=1)
    )
    assert doc.summary_1 == 1


def test_document_custom_summary_sub_category_container_value_accessible_via_getattr() -> None:
    """doc.my_custom_key resolves to the value of a custom summary ContainerAnnotation"""
    doc = Document(compute_metadata=False)
    doc.summary.dump_sub_category(
        ObjectTestType.SUMMARY_1,
        ContainerAnnotation(category_name=ObjectTestType.SUMMARY_1, value="my_value"),
    )
    assert doc.summary_1 == "my_value"


def test_document_getattr_raises_for_unregistered_attribute() -> None:
    """Attributes that are neither properties nor summary sub categories still raise"""
    doc = Document(compute_metadata=False)
    with pytest.raises(AttributeError):
        _ = doc.some_completely_unknown_attribute


@pytest.mark.parametrize(
    "suffix,expected_file_type",
    [
        (".png", DocumentFileLabel.PNG),
        (".jpg", DocumentFileLabel.JPG),
        (".jpeg", DocumentFileLabel.JPEG),
        (".tif", DocumentFileLabel.TIFF),
        (".tiff", DocumentFileLabel.TIFF),
    ],
)
def test_single_image_resolves_file_type_from_suffix(
    sample_document_image_path: Path, tmp_path: Path, suffix: str, expected_file_type: DocumentFileLabel
) -> None:
    """test that a document backed by a single image file resolves its file type from the suffix"""

    location = tmp_path / f"sample{suffix}"
    location.write_bytes(sample_document_image_path.read_bytes())

    doc = Document(location=location)

    assert doc.file_type == expected_file_type
    assert doc.file_name == location.name


def test_single_image_reports_one_page(sample_document_image_path: Path) -> None:
    """test that a document backed by a single image file has exactly one page"""

    doc = Document(location=sample_document_image_path)

    assert doc.number_of_pages == 1
    assert doc.get_page_reference(1).source_path == os.fspath(sample_document_image_path)


def test_single_image_get_image_loads_pixels(sample_document_image_path: Path) -> None:
    """test that a document backed by a single image file reloads its pixels"""

    doc = Document(location=sample_document_image_path)

    image = doc.get_image(page_number=1, load_pixels=True)

    assert image.page_number == 1
    assert image.image is not None


def test_single_image_save_load_round_trip(sample_document_image_path: Path, tmp_path: Path) -> None:
    """test that a document backed by a single image file survives a save/load round trip"""

    location = tmp_path / "sample.png"
    location.write_bytes(sample_document_image_path.read_bytes())
    doc = Document(location=location)
    saved_path = doc.save(image_to_json=False, path=tmp_path)

    doc = Document.from_json(str(saved_path))

    assert doc.file_type == DocumentFileLabel.PNG
    assert doc.number_of_pages == 1
    assert doc.get_image(page_number=1, load_pixels=True).image is not None


def test_legacy_document_type_key_still_loads() -> None:
    """test that a document saved before `document_type` was renamed to `file_type` still loads"""

    inputs = Document(file_name="sample.png", file_type=DocumentFileLabel.PNG, compute_metadata=False).as_dict()
    inputs["document_type"] = inputs.pop("file_type")

    doc = Document.from_dict(inputs)

    assert doc.file_type == DocumentFileLabel.PNG


def test_re_assign_document_summary_cat_ids() -> None:
    """test that summary sub categories of a document are assigned the ids of the categories passed"""

    doc = Document(compute_metadata=False)
    doc.summary.dump_sub_category(DocumentKey.DOCUMENT_TYPE, CategoryAnnotation(category_name=DocumentLabel.INVOICE))

    doc = re_assign_document_summary_cat_ids(  # pylint: disable=E1102
        {DocumentKey.DOCUMENT_TYPE: {DocumentLabel.LETTER: 1, DocumentLabel.INVOICE: 2}}
    )(doc)

    assert doc.summary.get_sub_category(DocumentKey.DOCUMENT_TYPE).category_id == 2


def test_re_assign_document_summary_cat_ids_with_unknown_category() -> None:
    """test that an unknown category name falls back to the default category id"""

    doc = Document(compute_metadata=False)
    doc.summary.dump_sub_category(DocumentKey.DOCUMENT_TYPE, CategoryAnnotation(category_name=DocumentLabel.INVOICE))

    doc = re_assign_document_summary_cat_ids(  # pylint: disable=E1102
        {DocumentKey.DOCUMENT_TYPE: {DocumentLabel.LETTER: 1}}
    )(doc)

    assert doc.summary.get_sub_category(DocumentKey.DOCUMENT_TYPE).category_id == DEFAULT_CATEGORY_ID


def test_re_assign_document_summary_cat_ids_skips_missing_keys() -> None:
    """test that a summary sub category key that is not dumped is skipped"""

    doc = Document(compute_metadata=False)

    doc = re_assign_document_summary_cat_ids(  # pylint: disable=E1102
        {DocumentKey.DOCUMENT_TYPE: {DocumentLabel.LETTER: 1}}
    )(doc)

    assert DocumentKey.DOCUMENT_TYPE not in doc.summary.sub_categories
