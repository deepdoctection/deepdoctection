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

from copy import copy
from typing import Any, List, Literal, Optional, Sequence, Union

from ..datapoint.image import Image
from ..extern.hflayoutlm import HFLayoutLmSequenceClassifierBase, HFLayoutLmTokenClassifierBase
from ..mapper.laylmstruct import image_to_layoutlm_features
from ..utils.detection_types import JsonDict
from ..utils.file_utils import transformers_available
from ..utils.settings import BioTag, LayoutType, ObjectTypes, PageType, TokenClasses, WordType
from .base import LanguageModelPipelineComponent
from .registry import pipeline_component_registry

if transformers_available():
    from transformers import LayoutLMTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

    _ARCHITECTURES_TO_TOKENIZER = {
        ("LayoutLMForTokenClassification", False): LayoutLMTokenizerFast.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        ),
        ("LayoutLMForSequenceClassification", False): LayoutLMTokenizerFast.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        ),
        ("LayoutLMv2ForTokenClassification", False): LayoutLMTokenizerFast.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        ),
        ("LayoutLMv2ForSequenceClassification", False): LayoutLMTokenizerFast.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        ),
        ("LayoutLMv2ForTokenClassification", True): XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base"),
        ("LayoutLMv2ForSequenceClassification", True): XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base"),
        ("LayoutLMv3ForSequenceClassification", False): RobertaTokenizerFast.from_pretrained(
            "roberta-base", add_prefix_space=True
        ),
        ("LayoutLMv3ForTokenClassification", False): RobertaTokenizerFast.from_pretrained(
            "roberta-base", add_prefix_space=True
        ),
    }


def get_tokenizer_from_architecture(architecture_name: str, use_xlm_tokenizer: bool) -> Any:
    """
    We do not use the tokenizer for a particular model that the transformer library provides. Thie mapping therefore
    returns the tokenizer that should be used for a particular model.

    :param architecture_name: The model as stated in the transformer library.
    :param use_xlm_tokenizer: True if one uses the LayoutXLM. (The model cannot be distinguished from LayoutLMv2).
    :return: Tokenizer instance to use.
    """
    return _ARCHITECTURES_TO_TOKENIZER[(architecture_name, use_xlm_tokenizer)]


@pipeline_component_registry.register("LMTokenClassifierService")
class LMTokenClassifierService(LanguageModelPipelineComponent):
    """
    Pipeline component for token classification

    **Example**

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
            layoutlm = HFLayoutLmTokenClassifier(categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                               'E-header', 'E-question', 'I-answer', 'I-header',
                                                               'I-question', 'O', 'S-answer', 'S-header', 'S-question'])

            # token classification service
            layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
    """

    def __init__(
        self,
        tokenizer: Any,
        language_model: HFLayoutLmTokenClassifierBase,
        padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
        use_other_as_default_category: bool = False,
        segment_positions: Optional[Union[LayoutType, Sequence[LayoutType]]] = None,
        sliding_window_stride: int = 0,
    ) -> None:
        """
        :param tokenizer: Token classifier, typing allows currently anything. This will be changed in the future
        :param language_model: language model token classifier
        :param padding: A padding strategy to be passed to the tokenizer. Must bei either `max_length, longest` or
                        `do_not_pad`.
        :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                           maximum acceptable input length for the model if that argument is not provided. This will
                           truncate token by token, removing a token from the longest sequence in the pair if a pair of
                           sequences (or a batch of pairs) is provided.
                           If `False` then no truncation (i.e., can output batch with sequence lengths greater than the
                           model maximum admissible input size).
        :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens
                           can be returned as an additional batch element. Not that in this case, the number of input
                           batch samples will be smaller than the output batch samples.
        :param use_other_as_default_category: When predicting token classes, it might be possible that some words might
                                              not get sent to the model because they are categorized as not eligible
                                              token (e.g. empty string). If set to `True` it will assign all words
                                              without token the `BioTag.outside` token.
        :param segment_positions: Using bounding boxes of segment instead of words improves model accuracy significantly
                              for models that have been trained on segments rather than words.
                              Choose a single or a sequence of layout segments to use their bounding boxes. Note, that
                              the layout segments need to have a child-relationship with words. If a word does not
                              appear as child, it will use the word bounding box.
        :param sliding_window_stride: If the output of the tokenizer exceeds the max_length sequence length, a sliding
                              windows will be created with each window having max_length sequence input. When using
                              `sliding_window_stride=0` no strides will be created, otherwise it will create slides
                              with windows shifted `sliding_window_stride` to the right.
        """
        self.language_model = language_model
        self.padding = padding
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        self.use_other_as_default_category = use_other_as_default_category
        self.segment_positions = segment_positions
        self.sliding_window_stride = sliding_window_stride
        if self.use_other_as_default_category:
            categories_name_as_key = {val: key for key, val in self.language_model.categories.items()}
            self.default_key: ObjectTypes
            if BioTag.outside in categories_name_as_key:
                self.default_key = BioTag.outside
            else:
                self.default_key = TokenClasses.other
            self.other_name_as_key = {self.default_key: categories_name_as_key[self.default_key]}
        super().__init__(self._get_name(), tokenizer, image_to_layoutlm_features)
        self.required_kwargs = {
            "tokenizer": self.tokenizer,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_overflowing_tokens": self.return_overflowing_tokens,
            "return_tensors": "pt",
            "segment_positions": self.segment_positions,
            "sliding_window_stride": self.sliding_window_stride,
        }
        self.required_kwargs.update(self.language_model.default_kwargs_for_input_mapping())
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        lm_input = self.mapping_to_lm_input_func(**self.required_kwargs)(dp)
        if lm_input is None:
            return
        lm_output = self.language_model.predict(**lm_input)

        # turn to word level predictions and remove all special tokens
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
                if token.class_name == token.semantic_name:
                    token_class_name_id = token.class_id
                else:
                    token_class_name_id = None
                self.dp_manager.set_category_annotation(
                    token.semantic_name, token_class_name_id, WordType.token_class, token.uuid
                )
                self.dp_manager.set_category_annotation(token.bio_tag, None, WordType.tag, token.uuid)
                self.dp_manager.set_category_annotation(
                    token.class_name, token.class_id, WordType.token_tag, token.uuid
                )
                words_populated.append(token.uuid)

        if self.use_other_as_default_category:
            word_anns = dp.get_annotation(LayoutType.word)
            for word in word_anns:
                if WordType.token_class not in word.sub_categories:
                    self.dp_manager.set_category_annotation(
                        TokenClasses.other,
                        self.other_name_as_key[self.default_key],
                        WordType.token_class,
                        word.annotation_id,
                    )
                if WordType.tag not in word.sub_categories:
                    self.dp_manager.set_category_annotation(BioTag.outside, None, WordType.tag, word.annotation_id)
                if WordType.token_tag not in word.sub_categories:
                    self.dp_manager.set_category_annotation(
                        self.default_key,
                        self.other_name_as_key[self.default_key],
                        WordType.token_tag,
                        word.annotation_id,
                    )

    def clone(self) -> "LMTokenClassifierService":
        return self.__class__(
            copy(self.tokenizer),
            self.language_model.clone(),
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
            self.use_other_as_default_category,
            self.segment_positions,
            self.sliding_window_stride,
        )

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {LayoutType.word: {WordType.token_class, WordType.tag, WordType.token_tag}}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )

    def _get_name(self) -> str:
        return f"lm_token_class_{self.language_model.name}"

    def _init_sanity_checks(self) -> None:
        tokenizer_class = self.language_model.model.config.tokenizer_class
        use_xlm_tokenizer = False
        if tokenizer_class is not None:
            use_xlm_tokenizer = True
        tokenizer_reference = get_tokenizer_from_architecture(
            self.language_model.model.__class__.__name__, use_xlm_tokenizer
        )
        if not isinstance(self.tokenizer, type(tokenizer_reference)):
            raise ValueError(
                f"You want to use {type(self.tokenizer)} but you should use {type(tokenizer_reference)} "
                f"in this framework"
            )


@pipeline_component_registry.register("LMSequenceClassifierService")
class LMSequenceClassifierService(LanguageModelPipelineComponent):
    """
    Pipeline component for sequence classification

    **Example**

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
            layoutlm = HFLayoutLmSequenceClassifier("path/to/config.json","path/to/model.bin",
                                                     categories=["handwritten", "presentation", "resume"])

            # token classification service
            layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...

    """

    def __init__(
        self,
        tokenizer: Any,
        language_model: HFLayoutLmSequenceClassifierBase,
        padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
    ) -> None:
        """
        :param tokenizer: Tokenizer, typing allows currently anything. This will be changed in the future
        :param language_model: language model sequence classifier
        :param padding: A padding strategy to be passed to the tokenizer. Must bei either `max_length, longest` or
                        `do_not_pad`.
        :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                           maximum acceptable input length for the model if that argument is not provided. This will
                           truncate token by token, removing a token from the longest sequence in the pair if a pair of
                           sequences (or a batch of pairs) is provided.
                           If `False` then no truncation (i.e., can output batch with sequence lengths greater than the
                           model maximum admissible input size).
        :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens
                           can be returned as an additional batch element. Not that in this case, the number of input
                           batch samples will be smaller than the output batch samples.
        """
        self.language_model = language_model
        self.padding = padding
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        super().__init__(self._get_name(), tokenizer, image_to_layoutlm_features)
        self.required_kwargs = {
            "tokenizer": self.tokenizer,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_overflowing_tokens": self.return_overflowing_tokens,
            "return_tensors": "pt",
        }
        self.required_kwargs.update(self.language_model.default_kwargs_for_input_mapping())
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        lm_input = self.mapping_to_lm_input_func(**self.required_kwargs)(dp)
        if lm_input is None:
            return
        lm_output = self.language_model.predict(**lm_input)
        self.dp_manager.set_summary_annotation(
            PageType.document_type, lm_output.class_name, lm_output.class_id, None, lm_output.score
        )

    def clone(self) -> "LMSequenceClassifierService":
        return self.__class__(
            copy(self.tokenizer),
            self.language_model.clone(),
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
        )

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", [PageType.document_type]),
            ]
        )

    def _get_name(self) -> str:
        return f"lm_sequence_class_{self.language_model.name}"

    def _init_sanity_checks(self) -> None:
        tokenizer_class = self.language_model.model.config.tokenizer_class
        use_xlm_tokenizer = False
        if tokenizer_class is not None:
            use_xlm_tokenizer = True
        tokenizer_reference = get_tokenizer_from_architecture(
            self.language_model.model.__class__.__name__, use_xlm_tokenizer
        )
        if not isinstance(self.tokenizer, type(tokenizer_reference)):
            raise ValueError(
                f"You want to use {type(self.tokenizer)} but you should use {type(tokenizer_reference)} "
                f"in this framework"
            )
