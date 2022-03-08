# -*- coding: utf-8 -*-
# File: test_serialize.py

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
Testing the module dataflow.custom_serialize
"""
import os

from deepdoctection.dataflow import (
    CustomDataFromList,
    SerializerCoco,
    SerializerFiles,
    SerializerJsonlines,
    SerializerPdfDoc,
)

from ..test_utils import collect_datapoint_from_dataflow, get_test_path


class TestSerializerJsonlines:
    """
    Testing SerializerJsonlines loading and saving functions
    """

    @staticmethod
    def test_loading() -> None:
        """
        Testing the loading of a .jsonl file
        """

        # Arrange
        test_file = os.path.join(get_test_path(), "test_file.jsonl")

        # Act
        df = SerializerJsonlines.load(test_file)
        output = collect_datapoint_from_dataflow(df=df)

        # Assert
        assert len(output) == 3

    @staticmethod
    def test_saving() -> None:
        """
        Testing the saving of a .jsonl file
        """

        # Arrange
        for_saving = [{"foo": "bak"}, {"fok": "bah"}]
        df = CustomDataFromList(for_saving)

        file_name = "test_file_saved.jsonl"
        test_file = os.path.join(get_test_path(), file_name)

        # Act
        SerializerJsonlines.save(df, get_test_path(), file_name)

        # Assert
        assert os.path.isfile(test_file)

        # Clean-up
        os.remove(test_file)


class TestSerializerFiles:  # pylint: disable=R0903
    """
    Testing SerializerFiles loading function
    """

    @staticmethod
    def test_loading() -> None:
        """
        Test the loading of elements of a file directory
        """

        # Arrange and Act
        df = SerializerFiles.load(get_test_path(), file_type=".jsonl")
        output = collect_datapoint_from_dataflow(df=df)

        # Assert
        assert len(output) == 3


class TestSerializerCoco:  # pylint: disable=R0903
    """
    Testing SerializerCoco loading function
    """

    @staticmethod
    def test_loading() -> None:
        """
        Test the loading of a .json file
        """

        # Arrange
        test_file = os.path.join(get_test_path(), "test_file.json")

        # Act
        df = SerializerCoco.load(test_file)
        output = collect_datapoint_from_dataflow(df=df)

        # Assert
        assert len(output) == 20

        # Act
        df = SerializerCoco.load(test_file, max_datapoints=5)
        output = collect_datapoint_from_dataflow(df=df)

        # Assert
        assert len(output) == 5


class TestSerializerPdfDoc:  # pylint: disable=R0903
    """
    Testing SerializerPdfDoc loading function
    """

    @staticmethod
    def test_loading() -> None:
        """
        Test the loading of a .pdf file
        """

        # Arrange
        test_file = os.path.join(get_test_path(), "test_file.pdf")

        # Act
        df = SerializerPdfDoc.load(test_file)
        output = collect_datapoint_from_dataflow(df=df)
        first_image = output[0]

        # Assert
        assert len(output) == 2
        assert first_image["path"] == test_file
        assert first_image["file_name"] == "test_file_0.pdf"
        assert isinstance(first_image["pdf_bytes"], bytes)
