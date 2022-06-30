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
from ..utils.detection_types import JsonDict
from ..utils.settings import names
from .base import LanguageModelPipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("LMTokenClassifierService")
class LMTokenClassifierService(LanguageModelPipelineComponent):
    """
    Pipeline component for token classification

    **Example**

        .. code-block:: python

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizer.from_pretrained("mrm8488/layoutlm-finetuned-funsd")
            layoutlm = HFLayoutLmTokenClassifier(categories_explicit= ['B-ANSWER', 'B-HEAD', 'B-QUESTION', 'E-ANSWER',
                                                               'E-HEAD', 'E-QUESTION', 'I-ANSWER', 'I-HEAD',
                                                               'I-QUESTION', 'O', 'S-ANSWER', 'S-HEAD', 'S-QUESTION'])

            # token classification service
            layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm,image_to_layoutlm)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
    """

    def serve(self, dp: Image) -> None:
        image_to_lm_input = self.mapping_to_lm_input_func(tokenizer=self.tokenizer)
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

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {names.C.WORD: {names.C.SE, names.NER.TAG}}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )
