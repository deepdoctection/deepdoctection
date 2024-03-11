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
This script provides a TextRefinementService as part of a text processing pipeline, 
which can be configured to use spell checking, NLP-based refinement, or both, and supports multiple languages.
"""

from copy import deepcopy
from typing import List, Dict, Optional, Sequence, Union
from transformers import pipeline, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
import torch
import enchant

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..utils.detection_types import ImageType, JsonDict
from ..utils.settings import PageType, TypeOrStr, WordType, get_type
from .base import PipelineComponent
from .registry import pipeline_component_registry

# requires C library for PyEnchant, 
# sudo apt-get install -y libenchant1c2a


__all__ = ["TextRefinementService"]

class TextRefinementService(PipelineComponent):
    """
    Pipeline component for refining text extraction results using dictionaries, lexicons, or language models.
    """
    
    def __init__(self, 
                 language: str = "en",
                 use_spellcheck_refinement: bool = True,
                 use_nlp_refinement: bool = True,
                 nlp_refinement_model_name = f"bert-base-multilingual-cased",
                 text_refinement_threshold: float = 0.8,
                 categories_to_refine: Optional[Union[Sequence[TypeOrStr], TypeOrStr]] = None):
        """
        Initializes the TextRefinementService with necessary resources.
        """
                
        self.language = language
        self.use_spellcheck_refinement = use_spellcheck_refinement
        self.use_nlp_refinement = use_nlp_refinement
        self.text_refinement_threshold = text_refinement_threshold
        self.categories_to_refine = [get_type(cat) for cat in categories_to_refine] if categories_to_refine else []
        
        if self.use_spellcheck_refinement:
            self.spell_checker = enchant.Dict(language)
        
        if self.use_nlp_refinement:
            # TODO: consider using language-specific models instead of multilingual BERT
            self.tokenizer = AutoTokenizer.from_pretrained(nlp_refinement_model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(nlp_refinement_model_name)
            self.nlp_pipeline = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)

        super().__init__("text_refinement")

                
    def refine_with_spellchecker(self, text: str) -> str:
        words = text.split()
        refined_words = [self.spell_checker.suggest(word)[0] if not self.spell_checker.check(word) else word for word in words]
        return " ".join(refined_words)
    
    
    def refine_with_nlp(self, text: str) -> str:
        # This function needs to be carefully designed to maintain the original structure and punctuation
        # Here's a simplified example that doesn't handle all cases
        refined_text = text
        for masked_index in range(len(refined_text.split())):
            masked_text = self.create_masked_text(refined_text, masked_index)
            prediction = self.nlp_pipeline(masked_text)[0]
            refined_text = self.replace_masked_word(refined_text, masked_index, prediction['token_str'])
        return refined_text
    
    
    def create_masked_text(self, text: str, masked_index: int) -> str:
        words = text.split()
        words[masked_index] = self.tokenizer.mask_token
        return " ".join(words)

    def replace_masked_word(self, text: str, index: int, new_word: str) -> str:
        words = text.split()
        words[index] = new_word
        return " ".join(words)

    def get_child(self, dp: Image, child_id: str) -> str:
        """
        Retrieve the text of a child word annotation by its ID.
        """
        child_annotation = dp.get_annotation(annotation_ids=child_id)
        child_annotation = child_annotation[0].sub_categories.get(WordType.characters) # TODO: Figure out how/if to incorporate layout element here.
        return child_annotation
    
    def update_child_annotations(self, dp: Image, child_ids: List[str], refined_text: str):
        """
        Update child word annotations with refined text. This simplistic approach assigns the entire
        refined text back to each child, which may not be accurate. Further logic could be added to
        intelligently distribute the refined text among children based on their original content.
        """
        for child_id in child_ids:
            child_annotation = dp.get_annotation(annotation_ids=child_id)
            child_annotation.set_value(refined_text)  # This needs refinement for accurate text distribution
    
    
    def serve(self, dp: Image) -> None:
        """
        Refines the text extraction results for the given document page (Image).
        This includes individual word corrections and context-aware refinements for larger text blocks.
        """
        for annotation in dp.get_annotation():
            # Handle larger text blocks that contain child word annotations
            if annotation.category_name in [cat.value for cat in self.categories_to_refine]:
                # Collect text from child words for context-aware refinement
                child_ids = annotation.relationships.get("child", [])
                children = [self.get_child(dp, child_id) for child_id in child_ids]
                
                # Join child texts to form the complete text block
                complete_text = " ".join(child_texts)
                # Refine the complete text block
                refined_complete_text = self.refine_text(complete_text)
                # Update child annotations with refined text, this part might require custom logic
                # to distribute the refined text back to individual children if necessary
                self.update_child_annotations(dp, child_ids, refined_complete_text)

            # Direct refinement for individual words, if not part of a larger text block
            elif annotation.category_name == WordType.word.value:
                original_text = annotation.get_value()
                refined_text = self.refine_text(original_text)
                annotation.set_value(refined_text)            

    def clone(self) -> "PipelineComponent":
        """
        Creates a copy of this TextRefinementService instance.
        """
        return deepcopy(self)
    

    # TODO: Talk to Janis about this
    def get_meta_annotation(self) -> Dict:
        """
        Returns metadata annotations related to text refinement.
        """
        return {
            "image_annotations": [cat.value for cat in self.categories_to_refine], # Specify the primary focus on relevant types
            "sub_categories": {},  # Adjust if your service modifies or utilizes specific sub-categories
            "relationships": {},  # Adjust if your service establishes or modifies relationships between annotations
            "summaries": []  # Adjust if your service contributes to or modifies summary information
        }

# Register the TextRefinementService component to the pipeline
pipeline_component_registry.register("TextRefinementService")(TextRefinementService)

