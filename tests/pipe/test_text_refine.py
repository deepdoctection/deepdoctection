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


# Adjusting the setup for the mock_image_annotation to more accurately reflect changes that would
# occur in a real scenario. Also, refining the way we assert these changes.
import sys
sys.path.insert(0, '/home/ubuntu/deepdoctection')

import pytest
from unittest.mock import MagicMock, patch

from deepdoctection.pipe.text_refine import TextRefinementService
from deepdoctection.utils.settings import LayoutType, WordType

class MockNLPRefinement:
    def __call__(self, text):
        if "hello [MASK]" in text:
            return [{"token_str": "world", "score": 0.99}]
        return [{"token_str": text.replace("[MASK]", ""), "score": 1.0}]

@pytest.fixture
def text_refinement_service():
    return TextRefinementService(
        use_spellcheck_refinement=True,
        use_nlp_refinement=True,
        nlp_refinement_model_name="bert-base-uncased",
        text_refinement_threshold=0.95
    )

@pytest.fixture
def mock_image_annotation():
    annotation = MagicMock()
    annotation.category_name = LayoutType.text
    annotation.value = "hello wrld"
    annotation.score = 0.8
    return annotation

@pytest.fixture
def image_datapoint(mock_image_annotation):
    dp = MagicMock()
    dp.get_annotation.return_value = [mock_image_annotation]
    return dp

def test_spell_checking_refinement(text_refinement_service, image_datapoint, mock_image_annotation):
    with patch("enchant.Dict") as mock_spell_checker, \
         patch.object(text_refinement_service.dp_manager, 'update_annotation') as mock_update_annotation:
        mock_spell_checker.return_value.check.return_value = False
        mock_spell_checker.return_value.suggest.return_value = ["world"]

        text_refinement_service.serve(image_datapoint)

        mock_update_annotation.assert_called_once_with(
            annotation_id=mock_image_annotation.annotation_id,
            new_value="hello world"
        )

def test_nlp_based_refinement(text_refinement_service, image_datapoint, mock_image_annotation):
    text_refinement_service.nlp_pipeline = MockNLPRefinement()
    text_refinement_service.serve(image_datapoint)
    
    # Assuming the NLP refinement directly updates the annotation
    assert mock_image_annotation.value == "hello world"

def test_no_action_needed(text_refinement_service, image_datapoint, mock_image_annotation):
    mock_image_annotation.value = "hello world"
    mock_image_annotation.score = 0.99
    
    text_refinement_service.serve(image_datapoint)