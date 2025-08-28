# -*- coding: utf-8 -*-
# File: lm.py

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
from __future__ import annotations

import inspect
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, Union

from ..datapoint.image import Image, MetaAnnotation
from ..extern.base import SequenceClassResult
from ..mapper.laylmstruct import image_to_layoutlm_features, image_to_lm_features
from ..utils.settings import BioTag, LayoutType, ObjectTypes, PageType, TokenClasses, WordType
from .base import PipelineComponent
from .registry import pipeline_component_registry

if TYPE_CHECKING:
    from ..extern.hflayoutlm import LayoutSequenceModels, LayoutTokenModels
    from ..extern.hflm import LmSequenceModels, LmTokenModels


@pipeline_component_registry.register("LMTokenClassifierService")
class LMTokenClassifierService(PipelineComponent):
    """
    Module for token classification pipeline.

    This module provides pipeline components for token and sequence classification using language models.

    Example:
        ```python
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
        layoutlm_service = LMTokenClassifierService(tokenizer, layoutlm)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service, layoutlm_service])

        path = "path/to/some/form"
        df = pipe.analyze(path=path)

        for dp in df:
            ...
        ```
    """

    def __init__(
        self,
        tokenizer: Any,
        language_model: Union[LayoutTokenModels, LmTokenModels],
        padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
        use_other_as_default_category: bool = False,
        segment_positions: Optional[Union[LayoutType, Sequence[LayoutType]]] = None,
        sliding_window_stride: int = 0,
    ) -> None:
        """
        Pipeline component for token classification.

        Example:
            ```python
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
            layoutlm_service = LMTokenClassifierService(tokenizer, layoutlm)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service, layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
            ```

        Args:
            tokenizer: `Token classifier`, typing allows currently anything. This will be changed in the future.
            language_model: `language model token classifier`.
            padding: A padding strategy to be passed to the `tokenizer`. Must be either `max_length`, `longest` or
                    `do_not_pad`.
            truncation: If `True` will truncate to a maximum length specified with the argument `max_length` or to
                        the maximum acceptable input length for the model if that argument is not provided. This
                        will truncate token by token, removing a token from the longest sequence in the pair if a
                        pair of sequences (or a batch of pairs) is provided. If `False` then no truncation (i.e.,
                        can output batch with sequence lengths greater than the model maximum admissible input
                        size).
            return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens
                                       can be returned as an additional batch element. Note that in this case, the
                                       number of input batch samples will be smaller than the output batch samples.
            use_other_as_default_category: When predicting token classes, it might be possible that some words
                                           might not get sent to the model because they are categorized as not
                                           eligible token (e.g. empty string). If set to `True` it will assign all
                                           words without token the `BioTag.outside` token.
            segment_positions: Using bounding boxes of segments instead of words improves model accuracy
                               significantly for models that have been trained on segments rather than words.
                               Choose a single or a sequence of layout segments to use their bounding boxes. Note,
                               that the layout segments need to have a child-relationship with words. If a word
                               does not appear as child, it will use the word bounding box.
            sliding_window_stride: If the output of the `tokenizer` exceeds the `max_length` sequence length, a
                                   sliding window will be created with each window having `max_length` sequence
                                   input. When using `sliding_window_stride=0` no strides will be created,
                                   otherwise it will create slides with windows shifted `sliding_window_stride` to
                                   the right.

        Note:
            If `use_other_as_default_category` is set, words without eligible tokens will be assigned the
            `BioTag.outside` token.
        """
        self.language_model = language_model
        self.padding = padding
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        self.use_other_as_default_category = use_other_as_default_category
        self.segment_positions = segment_positions
        self.sliding_window_stride = sliding_window_stride
        if self.use_other_as_default_category:
            categories_name_as_key = {val: key for key, val in self.language_model.categories.categories.items()}
            self.default_key: ObjectTypes
            if BioTag.OUTSIDE in categories_name_as_key:
                self.default_key = BioTag.OUTSIDE
            else:
                self.default_key = TokenClasses.OTHER
            self.other_name_as_key = {self.default_key: categories_name_as_key[self.default_key]}
        self.tokenizer = tokenizer
        self.mapping_to_lm_input_func = self.image_to_features_func(self.language_model.image_to_features_mapping())
        super().__init__(self._get_name(), self.language_model.model_id)
        self.required_kwargs = {
            "tokenizer": self.tokenizer,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_overflowing_tokens": self.return_overflowing_tokens,
            "return_tensors": "pt",
            "segment_positions": self.segment_positions,
            "sliding_window_stride": self.sliding_window_stride,
        }
        self.required_kwargs.update(self.language_model.default_kwargs_for_image_to_features_mapping())
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        """
        Serve the token classification pipeline on a given `Image`.

        Args:
            dp: The `Image` to process.

        Returns:
            None
        """
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

        words_populated: list[str] = []
        for token in lm_output:
            if token.uuid not in words_populated:
                if token.class_name == token.semantic_name:
                    token_class_name_id = token.class_id
                else:
                    token_class_name_id = None
                self.dp_manager.set_category_annotation(
                    token.semantic_name, token_class_name_id, WordType.TOKEN_CLASS, token.uuid, token.score
                )
                self.dp_manager.set_category_annotation(token.bio_tag, None, WordType.TAG, token.uuid)
                self.dp_manager.set_category_annotation(
                    token.class_name, token.class_id, WordType.TOKEN_TAG, token.uuid, token.score
                )
                words_populated.append(token.uuid)

        if self.use_other_as_default_category:
            word_anns = dp.get_annotation(LayoutType.WORD)
            for word in word_anns:
                if WordType.TOKEN_CLASS not in word.sub_categories:
                    self.dp_manager.set_category_annotation(
                        TokenClasses.OTHER,
                        self.other_name_as_key[self.default_key],
                        WordType.TOKEN_CLASS,
                        word.annotation_id,
                    )
                if WordType.TAG not in word.sub_categories:
                    self.dp_manager.set_category_annotation(BioTag.OUTSIDE, None, WordType.TAG, word.annotation_id)
                if WordType.TOKEN_TAG not in word.sub_categories:
                    self.dp_manager.set_category_annotation(
                        self.default_key,
                        self.other_name_as_key[self.default_key],
                        WordType.TOKEN_TAG,
                        word.annotation_id,
                    )

    def clone(self) -> LMTokenClassifierService:
        # ToDo: replace copying of tokenizer with a proper clone method. Otherwise we cannot run the evaluation with
        # multiple threads
        return self.__class__(
            copy(self.tokenizer),
            self.language_model.clone(),  # type: ignore
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
            self.use_other_as_default_category,
            self.segment_positions,
            self.sliding_window_stride,
        )

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(
            image_annotations=(),
            sub_categories={
                LayoutType.WORD: {
                    WordType.TOKEN_CLASS: set(self.language_model.categories.categories_semantics)  # type: ignore
                    if self.language_model.categories.categories_semantics
                    else [],
                    WordType.TAG: set(self.language_model.categories.categories_bio)  # type: ignore
                    if self.language_model.categories.categories_bio
                    else [],
                    WordType.TOKEN_TAG: set(self.language_model.categories.get_categories(as_dict=False)),
                }
            },
            relationships={},
            summaries=(),
        )

    def _get_name(self) -> str:
        return f"lm_token_class_{self.language_model.name}"

    def _init_sanity_checks(self) -> None:
        tokenizer_class_name = self.language_model.model.config.tokenizer_class
        if tokenizer_class_name != self.tokenizer.__class__.__name__:
            raise TypeError(
                f"You want to use {type(self.tokenizer)} but you should use {tokenizer_class_name} "
                f"in this framework"
            )
        func_params = inspect.signature(self.mapping_to_lm_input_func).parameters
        self.required_kwargs = {k: v for k, v in self.required_kwargs.items() if k in func_params}

    @staticmethod
    def image_to_features_func(mapping_str: str) -> Callable[..., Callable[[Image], Optional[Any]]]:
        """Replacing eval functions"""
        return {"image_to_layoutlm_features": image_to_layoutlm_features, "image_to_lm_features": image_to_lm_features}[
            mapping_str
        ]

    def clear_predictor(self) -> None:
        self.language_model.clear_model()


@pipeline_component_registry.register("LMSequenceClassifierService")
class LMSequenceClassifierService(PipelineComponent):
    """
    Pipeline component for sequence classification.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
        layoutlm = HFLayoutLmSequenceClassifier("path/to/config.json", "path/to/model.bin",
                                                 categories=["handwritten", "presentation", "resume"])

        # token classification service
        layoutlm_service = LMSequenceClassifierService(tokenizer, layoutlm)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service, layoutlm_service])

        path = "path/to/some/form"
        df = pipe.analyze(path=path)

        for dp in df:
            ...
        ```


    """

    def __init__(
        self,
        tokenizer: Any,
        language_model: Union[LayoutSequenceModels, LmSequenceModels],
        padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
        use_other_as_default_category: bool = False,
    ) -> None:
        """
        Args:
            tokenizer: `Tokenizer`, typing allows currently anything. This will be changed in the future.
            language_model: `language model sequence classifier`.
            padding: A padding strategy to be passed to the `tokenizer`. Must be either `max_length`, `longest` or
                     `do_not_pad`.
            truncation: If `True` will truncate to a maximum length specified with the argument `max_length` or to the
                        maximum acceptable input length for the model if that argument is not provided. This will
                        truncate token by token, removing a token from the longest sequence in the pair if a pair of
                        sequences (or a batch of pairs) is provided. If `False` then no truncation (i.e., can output
                        batch with sequence lengths greater than the model maximum admissible input size).
            return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens
                                       can be returned as an additional batch element. Note that in this case, the
                                       number of input batch samples will be smaller than the output batch samples.
            use_other_as_default_category: When predicting document classes, it might be possible that some pages do
                                           not get sent to the model because they are empty. If set to `True` it will
                                           assign images with no features the category `TokenClasses.OTHER`.

        Note:
            If `use_other_as_default_category` is set, images with no features will be assigned the `TokenClasses.OTHER`
            category.

        """
        self.language_model = language_model
        self.padding = padding
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        self.use_other_as_default_category = use_other_as_default_category
        self.tokenizer = tokenizer
        self.mapping_to_lm_input_func = self.image_to_features_func(self.language_model.image_to_features_mapping())
        super().__init__(self._get_name(), self.language_model.model_id)
        self.required_kwargs = {
            "tokenizer": self.tokenizer,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_overflowing_tokens": self.return_overflowing_tokens,
            "return_tensors": "pt",
        }
        self.required_kwargs.update(self.language_model.default_kwargs_for_image_to_features_mapping())
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        """
        Serve the sequence classification pipeline on a given `Image`.

        Args:
            dp: The `Image` to process.

        Returns:
            None
        """
        lm_input = self.mapping_to_lm_input_func(**self.required_kwargs)(dp)
        lm_output = None
        if lm_input is None:
            if self.use_other_as_default_category:
                class_id = self.language_model.categories.get_categories(as_dict=True, name_as_key=True).get(
                    TokenClasses.OTHER, 1
                )
                lm_output = SequenceClassResult(class_name=TokenClasses.OTHER, class_id=class_id, score=-1.0)
        else:
            lm_output = self.language_model.predict(**lm_input)
        if lm_output:
            self.dp_manager.set_summary_annotation(
                PageType.DOCUMENT_TYPE, lm_output.class_name, lm_output.class_id, None, lm_output.score
            )

    def clone(self) -> LMSequenceClassifierService:
        return self.__class__(
            copy(self.tokenizer),
            self.language_model.clone(),  # type: ignore
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
        )

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(
            image_annotations=(), sub_categories={}, relationships={}, summaries=(PageType.DOCUMENT_TYPE,)
        )

    def _get_name(self) -> str:
        return f"lm_sequence_class_{self.language_model.name}"

    def _init_sanity_checks(self) -> None:
        tokenizer_class_name = self.language_model.model.config.tokenizer_class
        if tokenizer_class_name != self.tokenizer.__class__.__name__:
            raise TypeError(
                f"You want to use {type(self.tokenizer)} but you should use {tokenizer_class_name} "
                f"in this framework"
            )

    @staticmethod
    def image_to_features_func(mapping_str: str) -> Callable[..., Callable[[Image], Optional[Any]]]:
        """
        Get the function to map images to features for the language model.

        Args:
            mapping_str: The mapping function name as a string.

        Returns:
            A callable that maps an `Image` to features.
        """
        return {"image_to_layoutlm_features": image_to_layoutlm_features, "image_to_lm_features": image_to_lm_features}[
            mapping_str
        ]

    def clear_predictor(self) -> None:
        self.language_model.clear_model()
