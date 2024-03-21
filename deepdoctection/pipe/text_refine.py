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
from langdetect import detect
import enchant

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.settings import Languages, WordType, get_type
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
                 default_language: str = 'en',
                 categories_to_refine: Optional[Union[List[str], str]] = None):
        
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
        self.default_language = default_language
        self.categories_to_refine = [get_type(cat) for cat in 
                                     (categories_to_refine if isinstance(categories_to_refine, list) else [categories_to_refine])] if categories_to_refine else []
       
        # Spell checker is initialized dynamically based on detected language. 
        # TODO: Do the same with NLP pipeline
        self.spell_checker = None  # Initialized dynamically based on detected language
        
        if self.use_nlp_refinement:
            # TODO: consider using language-specific models instead of multilingual BERT
            self.tokenizer = AutoTokenizer.from_pretrained(nlp_refinement_model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(nlp_refinement_model_name)
            self.nlp_pipeline = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)

        super().__init__("text_refinement")

    
    def detect_and_set_language(self, text: str):
        """
        Detects the language of the provided text and initializes spell checking resources for that language.
        Tries to gracefully handle cases where the dictionary for the detected language is not available.
        """
        try:
            detected_lang = detect(text)
            # Map detected language codes to those defined in the Languages enum
            lang_code = Languages[detected_lang].value if detected_lang in Languages.__members__ else "en"
        except:
            lang_code = "en"  # Default to English if detection fails or mapping fails

        try:
            if self.use_spellcheck_refinement:
                self.spell_checker = enchant.Dict(lang_code)
        except enchant.errors.DictNotFoundError:
            print(f"Dictionary for language '{lang_code}' could not be found. Falling back to English.")
            self.spell_checker = enchant.Dict("en")
       
    def refine_text(self, texts: List[str], word_scores: List[Optional[float]]):
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
        print("DEBUG: Entering serve method.")  # Debug 1
        for annotation in dp.get_annotation():
            if annotation.category_name in [cat.value for cat in self.categories_to_refine]:
                child_ids = annotation.relationships.get("child", [])
                print(f"DEBUG: Found child IDs: {child_ids}")  # Debug to check child IDs
                children = [(self.get_child(dp, child_id).value, self.get_child(dp, child_id).score) for child_id in child_ids]
                if children:
                    children_texts, children_scores = zip(*children)
                    self.detect_and_set_language(" ".join(children_texts))
                    refined_texts, new_scores = self.refine_text(children_texts, children_scores)
                    print(f"DEBUG: About to update child annotations. Child IDs: {child_ids}, Refined Texts: {refined_texts}, New Scores: {new_scores}")  # Debug 2
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

