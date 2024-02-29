# -*- coding: utf-8 -*-
# File: text_refine.py

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
Module for refining methods of text. The refining methods lead ultimately to a better extracted text, with fewer errors
"""

# from ..extern.base import ObjectDetector, PdfMiner, TextRecognizer
# from ..extern.tessocr import TesseractOcrDetector

from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..utils.detection_types import ImageType, JsonDict
from ..utils.settings import PageType, TypeOrStr, WordType, get_type
from .base import PipelineComponent
from .registry import pipeline_component_registry
from ..datapoint.image import Image

import enchant
from transformers import BertTokenizer, BertForMaskedLM
import torch

from nltk.tokenize import word_tokenize
from typing import List

__all__ = ["TextRefinementService", "Resources"]


class Resources:
    def __init__(self):
        # Load English dictionary
        self.spell_checker = enchant.Dict("en_US")
        
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        
    def check_word(self, word: str) -> bool:
        return self.spell_checker.check(word)
    
    def suggest_corrections(self, word: str) -> list:
        return self.spell_checker.suggest(word)
        
    def predict_missing_word(self, text: str) -> str:
        """
        Use BERT to predict the missing word in a text with a [MASK] token.
        """
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        
        with torch.no_grad():
            predict = self.model(input_ids)[0]
        
        predicted_token_id = predict[0, mask_token_index, :].argmax(axis=-1)
        predicted_word = self.tokenizer.decode(predicted_token_id)
        return predicted_word


@pipeline_component_registry.register("TextRefinementService")
class TextRefinementService(PipelineComponent):
    """
    Pipeline component for refining text extraction results using dictionaries, lexicons, or language models.
    """

    def __init__(self, 
                 resources: Resources
                #  path_to_dictionary: str = None,
                #  path_to_lexicon: str = None,
                #  path_to_language_model: str = None
                 ):
        """
        Initializes the TextRefinementService with necessary resources such as dictionaries, lexicons, and/or language models.
        
        :param resources: A dictionary containing resources for text refinement.
        """
        super().__init__("text_refinement")
        self.resources = resources

    # @staticmethod
    # def load_dictionary(path_to_dictionary: str) -> set:
        """
        Loads a dictionary from a file where each line contains a word.
        
        :param path_to_dictionary: Path to the dictionary file.
        :return: A set containing all the words in the dictionary.
        """
        with open(path_to_dictionary, 'r', encoding='utf-8') as file:
            dictionary = {line.strip() for line in file}
        return dictionary
    
    def refine_text(self, text: str) -> str:
        """
        Refine the given text using spell checking and BERT model for context-aware corrections.
        """
        refined_text = []
        for word in text.split():
            # If the word is misspelled, suggest corrections
            if not self.resources.check_word(word):
                suggestions = self.resources.suggest_corrections(word)
                if suggestions:
                    word = suggestions[0]  # Take the first suggestion
            refined_text.append(word)
        
        # Use BERT to predict and refine further based on context (optional, for demonstration)
        # refined_text_str = " ".join(refined_text)
        # refined_text_str = refined_text_str.replace('___', '[MASK]') # Assume '___' indicates where to predict
        # predicted_word = self.resources.predict_missing_word(refined_text_str)
        # refined_text_str = refined_text_str.replace('[MASK]', predicted_word)
        
        return " ".join(refined_text)

    
    
    def serve(self, dp: Image) -> None:
        """
        Refines the text extraction results for the given document page (Image).

        :param dp: The document page as an Image object containing text extraction results.
        """
        # TODO: Implement the logic for refining text extraction results using the provided resources.
        pass

    def clone(self) -> "PipelineComponent":
        """
        Creates a copy of this TextRefinementService instance.
        """
        return self.__class__(self.resources)
