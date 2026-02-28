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

from dd_core.datapoint.image import Image
from dd_core.datapoint.view import Page
from dd_core.doc import Document, PageReference
from dd_core.utils import file_utils as fu


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_reports_number_of_pages(pdf_file_path_two_pages: Path) -> None:
    """test that pdf_reports_number_of_pages works"""

    doc = Document(location=pdf_file_path_two_pages)
    assert doc.number_of_pages == 2


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_pdf_get_page_reference_returns_valid_objects(pdf_file_path_two_pages: Path) -> None:
    """test that PageReference has all attributes filled"""

    doc = Document(location=pdf_file_path_two_pages)
    ref = doc.get_page_reference(0)

    assert isinstance(ref, PageReference)
    assert ref.image_id == "f05529d3-5d55-3f99-866d-311f5f539358"
    assert ref.source_path == os.fspath(pdf_file_path_two_pages)


def test_from_json_restores_internal_structures(sample_document_json: Path) -> None:
    """document returns internal structures"""
    doc = Document.from_json(sample_document_json)

    assert doc.document_id == "108e9e00-58cd-3c19-a900-38177f66fd87"
    assert isinstance(doc.get_page_reference(0), PageReference)


def test_len_equals_number_of_pages(sample_document_json: Path) -> None:
    """test document length equals number of pages"""
    doc = Document.from_json(sample_document_json)
    assert len(doc) == doc.number_of_pages


def test_doc_returns_structured_output(sample_document_json: Path) -> None:
    """test returns structured outputs"""
    doc = Document.from_json(sample_document_json)
    structured_output = doc.structured_output
    assert len(structured_output["buyer"]["buyerReference"]) == 11
    assert structured_output["buyer"]["contact"]["contractName"] is None


def test_get_page_and_get_image_return_types(sample_document_json: Path) -> None:
    """test get page by given page number"""
    doc = Document.from_json(sample_document_json)
    page0 = doc.get_page(0)
    assert isinstance(page0, Page)


def test_get_page_by_image_id_returns_page(sample_document_json: Path) -> None:
    """test get page by given image id"""
    image_id = "7e154965-1250-3f4f-b1c2-a6e822f0aaa"
    doc = Document.from_json(sample_document_json)
    page0 = doc.get_page(image_id=image_id)
    assert page0.image_id == image_id


def test_get_image_dataflow(sample_document_json: Path) -> None:
    """test get document dataflow"""
    doc = Document.from_json(sample_document_json)
    df = doc.get_image_dataflow()
    df.reset_state()
    assert list(df) == 6


def test_set_image_updates_references_and_images() -> None:
    """
    Create a plain Document without metadata computation, create an Image,
    add it via set_image and verify internal mappings and retrieval.
    """
    doc = Document(file_name="plain", location=Path(), compute_metadata=False)
    img = Image(file_name="test.png", location="/fake/location", page_number=5)

    doc.set_image(img, page_number=0)

    assert img.image_id in doc._images
    assert doc._images[img.image_id] is img

    ref = doc.get_page_reference(0)
    assert isinstance(ref, PageReference)

    fetched = doc.get_image(image_id=img.image_id)
    assert fetched is img


def test_save_dry_returns_export_dict_with_expected_keys(sample_document_json: Path) -> None:
    """test save dry returns export dict"""
    doc = Document.from_json(sample_document_json)
    exported = doc.save(dry=True)
    assert isinstance(exported, dict)


def test_get_annotation_id_with_given_image_id(sample_document_json: Path) -> None:
    """test get annotation id with given image id"""
    doc = Document.from_json(sample_document_json)
    text = doc.get_annotation(image_id="7e154965-1250-3f4f-b1c2-a6e822f0aaa5", category_names="table")
    page = doc.get_page(image_id="7e154965-1250-3f4f-b1c2-a6e822f0aaa5")
    text_from_page = page.get_annotation(category_names="table")
    assert len(text) == len(text_from_page)
