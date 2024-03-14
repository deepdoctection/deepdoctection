# -*- coding: utf-8 -*-
# File: test_text_refine.py

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
Testing module pipe.text_refine
"""


import sys
sys.path.insert(0, '/home/ubuntu/deepdoctection')

from unittest.mock import MagicMock, patch
import pytest

from deepdoctection.pipe.text_refine import TextRefinementService
from deepdoctection.datapoint.image import Image
from deepdoctection.utils.settings import LayoutType, WordType

class MockNLPRefinement:
    """
    Mock class for NLP-based refinement pipeline.
    """
    def __call__(self, text):
        # Mock NLP-based refinement, returning a dictionary with 'token_str' and 'score' keys.
        if text == "hello [MASK]":
            return [{"token_str": "world", "score": 0.99}]
        return [{"token_str": text, "score": 1.0}]

@pytest.fixture
def text_refinement_service():
    """
    Fixture for TextRefinementService.
    """
    service = TextRefinementService(
        use_spellcheck_refinement=True,
        use_nlp_refinement=True,
        nlp_refinement_model_name="bert-base-uncased",
        text_refinement_threshold=0.95
    )
    return service

# This fixture is meant to mock pre-refinement state
@pytest.fixture
def mock_image_annotation():
    annotation = MagicMock()
    annotation.category_name = LayoutType.text
    annotation.value = "hello wrld"  # Misspelled on purpose
    annotation.score = 0.8
    # Mock sub_categories to return a mock with value and score properties
    mock_sub_categories = {'get.return_value.value': 'hello wrld', 'get.return_value.score': 0.8}
    annotation.sub_categories.configure_mock(**mock_sub_categories)
    return annotation
    """
    Fixture for mock ImageAnnotation.
    """
    annotation = MagicMock()
    annotation.category_name = LayoutType.text
    annotation.value = "hello wrld"
    annotation.score = 0.8
    return annotation

@pytest.fixture
def refined_text_annotation():
    """
    Fixture for refined text annotation with a MagicMock.
    """
    annotation = MagicMock()
    # Configure mock to return specific values for nested calls
    annotation.sub_categories.get.return_value.value = "hello world"
    annotation.sub_categories.get.return_value.score = 0.99
    return annotation


@pytest.fixture
def image_datapoint(mock_image_annotation):
    dp = Image(file_name="dummy.jpg")
    dp.annotations = [mock_image_annotation]
    return dp


def test_text_refinement_spell_checking(text_refinement_service, image_datapoint, mock_image_annotation):
    # Mock enchant.Dict().suggest() to simulate spell checking.
    with patch("enchant.Dict") as mock_spell_checker:
        mock_spell_checker.return_value.check.return_value = False
        mock_spell_checker.return_value.suggest.return_value = ["world"]
        
        # Execute the service
        text_refinement_service.serve(image_datapoint)
        
        # Assuming your service modifies mock_image_annotation in place,
        # or you capture the refined result differently
        # This assertion needs to reflect the actual logic of how text_refinement_service updates annotations
        refined_value = mock_image_annotation.sub_categories.get(WordType.characters).return_value.value
        assert refined_value == "hello world"    """
    Revised test to use the refined_text_annotation fixture.
    """
    # Assuming serve method or similar logic updates annotations as intended
    text_refinement_service.serve(image_datapoint)
    
    # Direct assertion on the mock's configured return value
    assert refined_text_annotation.sub_categories.get(WordType.characters).value == "hello world"
    assert refined_text_annotation.sub_categories.get(WordType.characters).score == 0.99    
    """
    Test that text refinement service correctly refines text using spell checking.
    """
    # Mock enchant.Dict().suggest() to simulate spell checking.
    with patch("enchant.Dict") as mock_spell_checker:
        mock_spell_checker.return_value.check.return_value = False
        mock_spell_checker.return_value.suggest.return_value = ["world"]
        
        text_refinement_service.serve(image_datapoint)
        
        # Check if the misspelled word was corrected.
        assert mock_image_annotation.sub_categories.get(WordType.characters).value == "hello world"

def test_text_refinement_nlp_based(text_refinement_service, image_datapoint, mock_image_annotation):
    """
    Test that text refinement service correctly refines text using NLP-based approach.
    """
    # Mock NLP-based refinement pipeline.
    with patch.object(text_refinement_service, "nlp_pipeline", new_callable=MockNLPRefinement):
        text_refinement_service.serve(image_datapoint)
        
        # Check if the NLP-based refinement was applied correctly.
        assert mock_image_annotation.sub_categories.get(WordType.characters).value == "hello world"
        assert mock_image_annotation.sub_categories.get(WordType.characters).score == 0.99

def test_text_refinement_no_action_needed(text_refinement_service, image_datapoint, mock_image_annotation):
    """
    Test that text refinement service does not modify correctly spelled and confident text.
    """
    mock_image_annotation.value = "hello world"
    mock_image_annotation.score = 0.99  # High confidence, correct spelling
    
    text_refinement_service.serve(image_datapoint)
    
    # Check if the text was left unchanged.
    assert mock_image_annotation.sub_categories.get(WordType.characters).value == "hello world"
    assert mock_image_annotation.sub_categories.get(WordType.characters).score == 0.99