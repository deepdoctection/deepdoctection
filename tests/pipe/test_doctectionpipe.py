# -*- coding: utf-8 -*-
# File: test_doctectionpipe.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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
Testing the module pipe.doctectionpipe
"""

import os

from pytest import mark, raises

from deepdoctection.dataflow.serialize import DataFromList
from deepdoctection.datapoint.image import Image
from deepdoctection.pipe.doctectionpipe import DoctectionPipe

from ..test_utils import collect_datapoint_from_dataflow, get_integration_test_path


class TestDoctectionPipe:
    """
    Test DoctectionPipe
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.pipe = DoctectionPipe(pipeline_component_list=[])

    @mark.basic
    def test_analyze_dataset_dataflow(self, dp_image: Image) -> None:
        """
        Analyze a single image passed as a dataset_dataflow

        :param dp_image: test image
        """

        # Arrange
        df = DataFromList(lst=[dp_image])
        df = self.pipe.analyze(dataset_dataflow=df) # type: ignore

        # Act
        output = collect_datapoint_from_dataflow(df)

        # Assert
        assert len(output) == 1

    @mark.basic
    def test_analyze_folder(self)-> None:
        """
        Analyze a folder of images

        """

        # Arrange
        df = self.pipe.analyze(path=get_integration_test_path())

        # Act
        output = collect_datapoint_from_dataflow(df)

        # Assert
        assert len(output) == 1

    @mark.basic
    def test_analyze_bytes(self)-> None:
        """
        Analyze bytes of an .png

        """

        # Arrange
        path = get_integration_test_path() / "sample_2.png"
        with open(path, "rb") as f:
            image_bytes = f.read()

        # Act
        df = self.pipe.analyze(bytes=image_bytes, path=path, output="image")
        output = collect_datapoint_from_dataflow(df)

        # Assert
        assert len(output) == 1
        image = output[0]
        assert image.location == os.fspath(path)
        assert image.file_name == "sample_2.png"

    @mark.basic
    def test_analyze_bytes_fails_without_path_argument(self)-> None:
        """
        If path argument is not provided, analyze bytes should raise a ValueError
        """

        # Arrange
        path = get_integration_test_path() / "sample_2.png"
        with open(path, "rb") as f:
            image_bytes = f.read()

        # Act/Assert
        with raises(ValueError):
            self.pipe.analyze(bytes=image_bytes, output="image")
