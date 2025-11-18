# -*- coding: utf-8 -*-
# File: test_custom_serialize.py

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
Testing the module dataflow.custom_serialize
"""
import tempfile
from pathlib import Path
from typing import Any

import pytest


from dd_core.utils import file_utils as fu

import shared_test_utils as stu
from dd_core.dataflow import (
    CocoParser,
    CustomDataFromList,
    SerializerCoco,
    SerializerFiles,
    SerializerJsonlines,
    SerializerPdfDoc,
    SerializerTabsepFiles,
)



@pytest.fixture(name="temp_dir")
def fixture_temp_dir():
    """
    Temporary directory fixture
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def test_file_closing_iterator_closes_file(temp_dir: str, simple_dict_list: list[dict[str, Any]]) -> None:
    """
    Test that FileClosingIterator properly closes the file after iteration completes
    """
    # Arrange
    import json
    
    from dd_core.dataflow.custom_serialize import FileClosingIterator
    
    file_path = Path(temp_dir) / "test.json"
    with open(file_path, "w") as f:
        for item in simple_dict_list:
            f.write(json.dumps(item) + "\n")
    
    file_obj = open(file_path, "r")
    iterator = FileClosingIterator(file_obj, iter(file_obj))
    
    # Act - consume the iterator
    result = list(iterator)
    
    # Assert
    assert len(result) == 3
    assert file_obj.closed


def test_serializer_jsonlines_load(temp_dir: str, simple_dict_list: list[dict[str, Any]]) -> None:
    """
    Test SerializerJsonlines loading from jsonlines file
    """
    # Arrange
    import jsonlines
    
    file_path = Path(temp_dir) / "test.jsonl"
    with jsonlines.open(file_path, "w") as writer:
        for item in simple_dict_list:
            writer.write(item)
    
    # Act
    df = SerializerJsonlines.load(file_path)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 3
    assert result[0]["key1"] == "a"
    assert result[1]["key2"] == 2
    assert result[2]["key1"] == "c"


def test_serializer_jsonlines_save(temp_dir: str, simple_dict_list: list[dict[str, Any]]) -> None:
    """
    Test SerializerJsonlines saving to jsonlines file
    """
    # Arrange
    import jsonlines
    
    df = CustomDataFromList(simple_dict_list)
    file_name = "output.jsonl"
    
    # Act
    SerializerJsonlines.save(df, temp_dir, file_name, max_datapoints=2)
    
    # Assert
    output_path = Path(temp_dir) / file_name
    assert output_path.exists()
    
    with jsonlines.open(output_path, "r") as reader:
        saved_data = list(reader)
    
    assert len(saved_data) == 2
    assert saved_data[0]["key1"] == "a"


def test_serializer_tabsep_files_load(text_file: Path) -> None:
    """
    Test SerializerTabsepFiles loading from text file
    """
    # Act
    df = SerializerTabsepFiles.load(text_file)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 5
    assert "imagesg/g/t/h/gth35e00/2024525661.tif" in result[0]
    assert "11" in result[0]


def test_serializer_tabsep_files_load_with_max_datapoints(text_file: Path) -> None:
    """
    Test SerializerTabsepFiles with max_datapoints limit
    """
    # Act
    df = SerializerTabsepFiles.load(text_file, max_datapoints=3)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 3


def test_serializer_files_load(temp_dir: str) -> None:
    """
    Test SerializerFiles loading files of specific type from directory
    """
    # Arrange - create test files
    test_dir = Path(temp_dir)
    (test_dir / "file1.pdf").touch()
    (test_dir / "file2.pdf").touch()
    (test_dir / "file3.txt").touch()
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file4.pdf").touch()
    
    # Act
    df = SerializerFiles.load(test_dir, file_type=".pdf", sort=True)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 3
    # Verify all results are pdf files
    for file_path in result:
        assert file_path.endswith(".pdf")


def test_coco_parser_loads_annotations(coco_file_path: Path) -> None:
    """
    Test CocoParser loads and indexes COCO annotations correctly
    """
    # Act
    parser = CocoParser(coco_file_path)
    
    # Assert
    assert len(parser.imgs) > 0
    assert len(parser.anns) > 0
    assert len(parser.cats) > 0
    
    # Test getting image ids
    img_ids = parser.get_image_ids()
    assert len(img_ids) == 20


def test_serializer_coco_load(coco_file_path: Path) -> None:
    """
    Test SerializerCoco loads COCO format annotations
    """
    # Act
    df = SerializerCoco.load(coco_file_path)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 20
    # Check structure of first datapoint
    assert "file_name" in result[0]
    assert "annotations" in result[0]
    assert isinstance(result[0]["annotations"], list)


def test_serializer_pdf_doc_load(pdf_file_path_two_pages: Path) -> None:
    """
    Test SerializerPdfDoc loads PDF pages correctly
    """
    # Act
    df = SerializerPdfDoc.load(pdf_file_path_two_pages)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 2
    
    # Check structure of first page
    first_page = result[0]
    assert "path" in first_page
    assert "file_name" in first_page
    assert "pdf_bytes" in first_page
    assert "page_number" in first_page
    assert "document_id" in first_page
    assert isinstance(first_page["pdf_bytes"], bytes)
    assert first_page["page_number"] == 0


@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_serializer_pdf_doc_with_max_datapoints(pdf_file_path_two_pages: Path) -> None:
    """
    Test SerializerPdfDoc respects max_datapoints limit
    """
    # Act
    df = SerializerPdfDoc.load(pdf_file_path_two_pages, max_datapoints=1)
    result = stu.collect_datapoint_from_dataflow(df)
    
    # Assert
    assert len(result) == 1

@pytest.mark.skipif(not fu.pypdf_available(), reason="Pypdf is not installed")
def test_serializer_pdf_doc_split(pdf_file_path_two_pages: Path, temp_dir: str) -> None:
    """
    Test SerializerPdfDoc.split() creates separate PDF files for each page
    """
    # Act
    SerializerPdfDoc.split(pdf_file_path_two_pages, temp_dir, max_datapoint=2)
    
    # Assert
    output_files = list(Path(temp_dir).glob("*.pdf"))
    assert len(output_files) == 2
    
    # Verify files contain PDF data
    for pdf_file in output_files:
        assert pdf_file.stat().st_size > 0

