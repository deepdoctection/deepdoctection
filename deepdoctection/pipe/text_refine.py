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
Refines extracted text using both spell checking and NLP-based (BERT) approaches for improved accuracy.
This module provides a TextRefinementService as part of a text processing pipeline,
which can be configured to use spell checking, NLP-based refinement, or both, and supports multiple languages.
"""

from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import torch
from langdetect import detect, LangDetectException
import enchant

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.settings import Languages, WordType, get_type, get_language_enum_from_enchant_code, get_language_enum_from_langdetect_code
from .base import PipelineComponent
from .registry import pipeline_component_registry

# requires C library for PyEnchant, 
# sudo apt-get install -y libenchant1c2a

# PyEnchant requires additional language dictionaries for the spell checker backend. 
# for example, huntspell adds french and german
# sudo apt-get install hunspell-fr hunspell-de-de


__all__ = ["TextRefinementService"]

class TextRefinementService(PipelineComponent):
    """
    Pipeline component for refining text extraction results using spell checking and NLP-based approaches.
    """
    
    def __init__(self,
                 use_spellcheck_refinement: bool = True,
                 use_nlp_refinement: bool = True,
                 nlp_refinement_model_name: str = "bert-base-multilingual-cased",
                 text_refinement_threshold: float = 0.8,
                 categories_to_refine: Optional[List[str]] = None):
        
        """
        Initializes the TextRefinementService with the necessary configurations.

        Parameters:
        - use_spellcheck_refinement (bool): Whether to use spell checking for refinement.
        - use_nlp_refinement (bool): Whether to use NLP-based refinement.
        - nlp_refinement_model_name (str): The name of the NLP model to use for refinement.
        - text_refinement_threshold (float): The confidence threshold for accepting NLP-based refinements.
        - categories_to_refine (Optional[Union[List[str], str]]): Categories of text annotations to refine.
        """
                
        self.use_spellcheck_refinement = use_spellcheck_refinement
        self.use_nlp_refinement = use_nlp_refinement
        self.text_refinement_threshold = text_refinement_threshold
        self.categories_to_refine = [get_type(cat) for cat in (categories_to_refine if categories_to_refine else [])]
        self.spell_checker = None  # Dynamically initialized
        self.nlp_pipeline = self.init_nlp_pipeline(nlp_refinement_model_name)
        super().__init__("text_refinement")

    def init_nlp_pipeline(self, model_name: str):
        """
        Initializes the NLP pipeline for text refinement.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        return pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)

    def detect_and_set_language(self, text: str):
        """
        Detects the language and initializes resources for spell checking if available.
        Disables spell checking and logs a warning if the dictionary for the detected language is not found.
        """
        try:
            detected_lang = detect(text)
            lang_code = detected_lang[:2]
            self.spell_checker = enchant.Dict(lang_code)
        except LangDetectException as e:
            self.spell_checker = None
        except enchant.errors.DictNotFoundError:
            self.spell_checker = None

    def refine_text(self, texts: List[str], word_scores: List[Optional[float]]):
        """
        Refines texts using spell checking and NLP-based approaches, returning refined texts and their scores.
        """
        refined_texts = []
        new_scores = []
        for i, text in enumerate(texts):
            if not text.strip():  # Skip refinement for empty texts
                refined_texts.append(text)
                new_scores.append(word_scores[i])
                continue

            refined_text, new_score = self.refine_individual_text(text, word_scores[i])
            refined_texts.append(refined_text)
            new_scores.append(new_score)

        return refined_texts, new_scores

    def refine_individual_text(self, text: str, score: Optional[float]):
        """
        Refines an individual text, handling spell checking and NLP-based refinement.
        """
        core_word, prefix, suffix = self.extract_core_word(text)
        if self.use_spellcheck_refinement and self.spell_checker:
            core_word = self.refine_with_spellchecker(core_word)
        if self.use_nlp_refinement and score is not None and score < self.text_refinement_threshold:
            core_word, score = self.refine_with_nlp(core_word)

        refined_text = prefix + core_word + suffix
        return refined_text, score

    def extract_core_word(self, text: str):
        """
        Extracts the core word from the text, preserving surrounding punctuation.
        """
        core_word = ''.join(filter(str.isalnum, text))
        prefix = text[:text.find(core_word)]
        suffix = text[text.find(core_word) + len(core_word):]
        return core_word, prefix, suffix

    def refine_with_spellchecker(self, word: str):
        """
        Refines a word using the spell checker, returning the first suggestion if available.
        """
        
        # Check if the word is empty or consists only of whitespace
        if not word.strip():
            return word
            
        if not self.spell_checker.check(word):
            suggestions = self.spell_checker.suggest(word)
            if suggestions:  # Check if the suggestions list is not empty
                return suggestions[0]
        return word  # Ensure a string is always returned, even if no correction is needed.

    def refine_with_nlp(self, word: str):
        """
        Refines a word using the NLP pipeline, returning the refined word and its score.
        """
        masked_text = self.create_masked_text([word], 0)
        prediction = self.nlp_pipeline(masked_text)[0]
        return prediction['token_str'], prediction['score']

        """
        Refines a list of texts using spell checking and NLP-based approaches.

        Parameters:
        - texts (List[str]): The texts to refine.
        - word_scores (List[Optional[float]]): The scores of the words to refine.

        Returns:
        - (List[str], List[Optional[float]]): The refined texts and their new scores.
        """
        refined_texts = []
        new_scores = []
        for i, text in enumerate(texts):
            
            # Separate the core word from its surrounding punctuation
            core_word = ''.join(filter(str.isalnum, text))
            prefix = text[:text.find(core_word)]
            suffix = text[text.find(core_word) + len(core_word):]
            
            # Skip spell checking and NLP-based refinement for empty core words
            if not core_word:  # core_word is empty
                refined_texts.append(text)
                new_scores.append(word_scores[i])
                continue

            if self.use_spellcheck_refinement and self.spell_checker and not self.spell_checker.check(core_word):
                suggestions = self.spell_checker.suggest(core_word)
                core_word = suggestions[0] if suggestions else core_word

            if self.use_nlp_refinement and word_scores[i] is not None and word_scores[i] < self.text_refinement_threshold:
                masked_text = self.create_masked_text([core_word], 0)  # Mask only the core_word for prediction
                prediction = self.nlp_pipeline(masked_text)[0]
                core_word = prediction['token_str']
                new_scores.append(prediction['score'])
            else:
                new_scores.append(word_scores[i])
            
            # Reassemble the word with its original punctuation
            refined_text = prefix + core_word + suffix
            refined_texts.append(refined_text)

        return refined_texts, new_scores
    
    def create_masked_text(self, texts: List[str], masked_index: int) -> str:
        """
        Creates a text string with a specified index masked, for use in NLP-based refinement.

        Parameters:
        - texts (List[str]): The list of texts, one of which will be masked.
        - masked_index (int): The index of the text to mask.

        Returns:
        - str: The text string with the specified text masked.
        """
        texts_copy = deepcopy(texts)
        texts_copy[masked_index] = self.tokenizer.mask_token
        return " ".join(texts_copy)

    def get_child(self, dp: Image, child_id: str) -> str:
        """
        Retrieve the text of a child word annotation by its ID.
        """
        child_annotation = dp.get_annotation(annotation_ids=child_id)
        child_annotation = child_annotation[0].sub_categories.get(WordType.characters) # TODO: Figure out how/if to incorporate layout element here.
        return child_annotation
    
    def update_child_annotations(self, dp: Image, child_ids: List[str], refined_texts: List[str], word_scores: List[Optional[float]]):
        """
        Update child word annotations with refined text and potentially updated scores.

        Parameters:
        dp (Image): The current image datapoint being processed.
        child_ids (List[str]): List of child annotation IDs to update.
        refined_texts (List[str]): List of refined texts corresponding to each child ID.
        word_scores (List[Optional[float]]): List of new scores corresponding to each refined text.
        """
        for child_id, new_text, new_score in zip(child_ids, refined_texts, word_scores):
            # Update the annotation with the new value and score
            print(f"DEBUG: Updating annotation. Annotation ID: {child_id}, New Value: {new_text}, New Score: {new_score}")  # Debug 3
            self.dp_manager.update_annotation(annotation_id=child_id, new_value=new_text, new_score=new_score, sub_category_key=WordType.characters)
    
    def serve(self, dp: Image) -> None:
        """
        The service method that refines text annotations within the given image datapoint.

        Parameters:
        - dp (Image): The image datapoint containing text annotations to refine.
        """
        for annotation in dp.get_annotation():
            if annotation.category_name in [cat.value for cat in self.categories_to_refine]:
                child_ids = annotation.relationships.get("child", [])
                children = [(self.get_child(dp, child_id).value, self.get_child(dp, child_id).score) for child_id in child_ids]
                if children:
                    children_texts, children_scores = zip(*children)
                    self.detect_and_set_language(" ".join(children_texts))
                    refined_texts, new_scores = self.refine_text(children_texts, children_scores)
                    self.update_child_annotations(dp, child_ids, refined_texts, new_scores)


    def clone(self) -> "TextRefinementService":
        """
        Creates a deep copy of the current TextRefinementService instance.

        Returns:
        - TextRefinementService: A new instance of TextRefinementService with the same configuration.
        """
        return deepcopy(self)
    

    def get_meta_annotation(self) -> Dict:
        """
        Provides metadata annotations related to text refinement.

        Returns:
        - Dict: A dictionary of metadata annotations.
        """
        return {
            "image_annotations": [cat.value for cat in self.categories_to_refine],
            "sub_categories": {},
            "relationships": {},
            "summaries": []
        }

# Register the TextRefinementService component to the pipeline
pipeline_component_registry.register("TextRefinementService")(TextRefinementService)

