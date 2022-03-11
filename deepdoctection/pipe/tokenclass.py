# -*- coding: utf-8 -*-
# File: tokenclass.py

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
Module for token classification pipeline
"""
from typing import List

from ..datapoint.image import Image
from ..utils.settings import names
from .base import LanguageModelPipelineComponent


class LMTokenClassifierService(LanguageModelPipelineComponent):
    """
    Pipeline component for token classification
    """

    def serve(self, dp: Image) -> None:
        image_to_lm_input = self.mapping_to_lm_input_func(tokenizer=self.tokenizer)  # type: ignore
        lm_input = image_to_lm_input(dp)
        lm_output = self.language_model.predict(**lm_input)

        # turn to word level predictions
        lm_output = [
            token
            for token in lm_output
            if token.token_id
            not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
            and not token.token.startswith("##")
        ]

        words_populated: List[str] = []
        for token in lm_output:
            if token.uuid not in words_populated:
                self.dp_manager.set_category_annotation(token.semantic_name, None, names.C.SE, token.uuid)
                self.dp_manager.set_category_annotation(token.bio_tag, None, names.NER.TAG, token.uuid)
                words_populated.append(token.uuid)
