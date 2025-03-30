# -*- coding: utf-8 -*-
# File: test_base.py

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
Testing module extern.base
"""

import uuid
from types import MappingProxyType
from unittest.mock import MagicMock

import numpy as np
from pytest import mark

from deepdoctection.extern.base import (
    DetectionResult,
    DeterministicImageTransformer,
    ModelCategories,
    NerModelCategories,
)
from deepdoctection.utils.settings import get_type
from deepdoctection.utils.transform import BaseTransform


class TestModelCategories:
    """
    Test ModelCategories
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.init_categories = {1: "word", 2: "line", 3: "table", 4: "figure", 5: "header", 6: "footnote"}
        self.categories = ModelCategories(init_categories=self.init_categories)

    @mark.basic
    def test_get_categories(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType(
            {
                1: get_type("word"),
                2: get_type("line"),
                3: get_type("table"),
                4: get_type("figure"),
                5: get_type("header"),
                6: get_type("footnote"),
            }
        )
        assert categories == expected_categories

    @mark.basic
    def test_get_categories_name_as_keys(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.get_categories(name_as_key=False)

        # Assert
        expected_categories = MappingProxyType(
            {
                1: get_type("word"),
                2: get_type("line"),
                3: get_type("table"),
                4: get_type("figure"),
                5: get_type("header"),
                6: get_type("footnote"),
            }
        )
        assert categories == expected_categories

    @mark.basic
    def test_get_categories_as_tuple(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.get_categories(False)

        # Assert
        expected_categories = (
            get_type("word"),
            get_type("line"),
            get_type("table"),
            get_type("figure"),
            get_type("header"),
            get_type("footnote"),
        )
        assert categories == expected_categories

    @mark.basic
    def test_filter_categories(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        self.categories.filter_categories = (
            get_type("word"),
            get_type("header"),
        )
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType(
            {2: get_type("line"), 3: get_type("table"), 4: get_type("figure"), 6: get_type("footnote")}
        )
        assert categories == expected_categories

    @mark.basic
    def test_shift_category_ids(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.shift_category_ids(-1)

        # Assert
        expected_categories = MappingProxyType(
            {
                0: get_type("word"),
                1: get_type("line"),
                2: get_type("table"),
                3: get_type("figure"),
                4: get_type("header"),
                5: get_type("footnote"),
            }
        )
        assert categories == expected_categories


class TestNerModelCategories:
    """TestNerModelCategories"""

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.categories_semantics = (
            "question",
            "answer",
        )
        self.categories_bio = (
            "B",
            "I",
        )

    def test_get_categories(self) -> None:
        """
        Test NerModelCategories
        """
        # Arrange
        self.categories = NerModelCategories(
            init_categories=None, categories_semantics=self.categories_semantics, categories_bio=self.categories_bio
        )

        # Act
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType(
            {
                1: get_type("B-answer"),
                2: get_type("B-question"),
                3: get_type("I-answer"),
                4: get_type("I-question"),
            }
        )
        assert categories == expected_categories

    def test_categories_does_not_overwrite_consolidated_categories(self) -> None:
        """
        Test NerModelCategories
        """
        # Arrange
        self.categories = NerModelCategories(
            init_categories={1: get_type("B-answer"), 2: get_type("B-question")},
            categories_semantics=self.categories_semantics,
            categories_bio=self.categories_bio,
        )

        # Act
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType({1: get_type("B-answer"), 2: get_type("B-question")})
        assert categories == expected_categories


class TestDeterministicImageTransformer:
    """Test class for DeterministicImageTransformer"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        # Create a mock BaseTransform
        self.mock_base_transform = MagicMock(spec=BaseTransform)
        self.mock_base_transform.get_init_args.return_value = ["angle"]
        self.mock_base_transform.get_category_names.return_value = ("foo",)
        self.mock_base_transform.angle = 90

        # Mock the apply_image, apply_coords, and inverse_apply_coords methods
        self.mock_base_transform.apply_image.return_value = np.ones((10, 10, 3))
        self.mock_base_transform.apply_coords.return_value = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
        self.mock_base_transform.inverse_apply_coords.return_value = np.array([[5, 5, 15, 15], [25, 25, 35, 35]])

        # Create the transformer with mocked base_transform
        self.transformer = DeterministicImageTransformer(self.mock_base_transform)

        # Create test detection results with proper uuid4 and class_name
        self.detection_result1 = DetectionResult(
            box=[1, 1, 2, 2],
            class_id=1,
            class_name=get_type("report_date"),
            score=0.9,
            absolute_coords=True,
            uuid=str(uuid.uuid4()),
        )

        self.detection_result2 = DetectionResult(
            box=[3, 3, 4, 4],
            class_id=2,
            class_name=get_type("umbrella"),
            score=0.8,
            absolute_coords=True,
            uuid=str(uuid.uuid4()),
        )

        self.detect_results = [self.detection_result1, self.detection_result2]

    def test_transform_image(self) -> None:
        """Test transform_image method"""
        img = np.zeros((10, 10, 3))
        specification = DetectionResult()

        result = self.transformer.transform_image(img, specification) # type: ignore

        # Check if base_transform.apply_image was called with correct args
        self.mock_base_transform.apply_image.assert_called_once_with(img)
        # Check if result is what we expect from our mock
        assert np.array_equal(result, np.ones((10, 10, 3)))

    def test_transform_coords(self) -> None:
        """Test transform_coords method"""
        result = self.transformer.transform_coords(self.detect_results)

        # Verify the base_transform.apply_coords was called
        self.mock_base_transform.apply_coords.assert_called_once()

        # Verify results maintained the correct properties
        assert len(result) == 2
        assert result[0].uuid == self.detection_result1.uuid
        assert result[1].uuid == self.detection_result2.uuid
        assert result[0].box == [10, 10, 20, 20]
        assert result[1].box == [30, 30, 40, 40]
        assert result[0].class_name == "report_date"
        assert result[1].class_name == "umbrella"

    def test_inverse_transform_coords(self) -> None:
        """Test inverse_transform_coords method"""
        result = self.transformer.inverse_transform_coords(self.detect_results)

        # Verify the base_transform.inverse_apply_coords was called
        self.mock_base_transform.inverse_apply_coords.assert_called_once()

        # Verify results maintained the correct properties
        assert len(result) == 2
        assert result[0].uuid == self.detection_result1.uuid
        assert result[1].uuid == self.detection_result2.uuid
        assert result[0].box == [5, 5, 15, 15]
        assert result[1].box == [25, 25, 35, 35]
        assert result[0].class_id == 1
        assert result[1].class_id == 2

    def test_predict(self)  -> None:
        """Test predict method"""
        img = np.zeros((10, 10, 3))
        result = self.transformer.predict(img) # type: ignore

        # Check that attributes from base_transform were copied to DetectionResult
        assert result.angle == 90
