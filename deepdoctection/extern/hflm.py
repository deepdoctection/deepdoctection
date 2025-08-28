# -*- coding: utf-8 -*-
# File: hfml.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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
Wrapper for the HF Language Model for sequence and token classification
"""
from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union

from lazy_imports import try_import
from typing_extensions import TypeAlias

from ..utils.file_utils import get_pytorch_requirement, get_transformers_requirement
from ..utils.settings import TypeOrStr
from ..utils.types import JsonDict, PathLikeOrStr, Requirement
from .base import (
    DetectionResult,
    LanguageDetector,
    LMSequenceClassifier,
    LMTokenClassifier,
    ModelCategories,
    NerModelCategories,
    SequenceClassResult,
    TokenClassResult,
)
from .hflayoutlm import get_tokenizer_from_model_class
from .pt.ptutils import get_torch_device

with try_import() as pt_import_guard:
    import torch
    import torch.nn.functional as F

with try_import() as tr_import_guard:
    from transformers import (
        PretrainedConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaTokenizerFast,
    )


def predict_token_classes_from_lm(
    uuids: list[list[str]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    tokens: list[list[str]],
    model: XLMRobertaForTokenClassification,
) -> list[TokenClassResult]:
    """
    Args:
        uuids: A list of uuids that correspond to a word that induces the resulting token
        input_ids: Token converted to ids to be taken from `LayoutLMTokenizer`
        attention_mask: The associated attention masks from padded sequences taken from `LayoutLMTokenizer`
        token_type_ids: Torch tensor of token type ids taken from `LayoutLMTokenizer`
        tokens: List of original tokens taken from `LayoutLMTokenizer`
        model: layoutlm model for token classification

    Returns:
        A list of `TokenClassResult`s
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    soft_max = F.softmax(outputs.logits, dim=2)
    score = torch.max(soft_max, dim=2)[0].tolist()
    token_class_predictions_ = outputs.logits.argmax(-1).tolist()
    input_ids_list = input_ids.tolist()

    all_results = defaultdict(list)
    for idx, uuid_list in enumerate(uuids):
        for pos, token in enumerate(uuid_list):
            all_results[token].append(
                (input_ids_list[idx][pos], token_class_predictions_[idx][pos], tokens[idx][pos], score[idx][pos])
            )
    all_token_classes = []
    for uuid, res in all_results.items():
        res.sort(key=lambda x: x[3], reverse=True)
        output = res[0]
        all_token_classes.append(
            TokenClassResult(uuid=uuid, token_id=output[0], class_id=output[1], token=output[2], score=output[3])
        )
    return all_token_classes


def predict_sequence_classes_from_lm(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    model: XLMRobertaForSequenceClassification,
) -> SequenceClassResult:
    """
    Args:
        input_ids: Token converted to ids to be taken from `XLMRobertaTokenizer`
        attention_mask: The associated attention masks from padded sequences taken from `XLMRobertaTokenizer`
        token_type_ids: Torch tensor of token type ids taken from `XLMRobertaTokenizer`
        model: `XLMRobertaForSequenceClassification` model for sequence classification

    Returns:
        `SequenceClassResult`
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    score = torch.max(F.softmax(outputs.logits)).tolist()
    sequence_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()

    return SequenceClassResult(class_id=sequence_class_predictions, score=float(score))  # type: ignore


class HFLmTokenClassifierBase(LMTokenClassifier, ABC):
    """
    Abstract base class for wrapping Bert-like models for token classification into the framework.
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[int, TypeOrStr]] = None,
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        use_xlm_tokenizer: bool = False,
    ):
        """
        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact
            categories_semantics: A dict with key (indices) and values (category names) for `NER` semantics, i.e. the
                                 entities self. To be consistent with detectors use only values `>0`. Conversion will
                                 be done internally.
            categories_bio: A dict with key (indices) and values (category names) for `NER` tags (i.e. `BIO`). To be
                           consistent with detectors use only `values>0`. Conversion will be done internally.
            categories: If you have a pre-trained model you can pass a complete dict of NER categories
            device: The device (cpu,"cuda"), where to place the model.
            use_xlm_tokenizer: True if one uses the `LayoutXLM` or a lilt model built with a xlm language model, e.g.
                              `info-xlm` or `roberta-xlm`. (`LayoutXLM` cannot be distinguished from LayoutLMv2).
        """

        if categories is None:
            if categories_semantics is None:
                raise ValueError("If categories is None then categories_semantics cannot be None")
            if categories_bio is None:
                raise ValueError("If categories is None then categories_bio cannot be None")

        self.path_config = Path(path_config_json)
        self.path_weights = Path(path_weights)
        self.categories = NerModelCategories(
            init_categories=categories, categories_semantics=categories_semantics, categories_bio=categories_bio
        )
        self.device = get_torch_device(device)
        self.use_xlm_tokenizer = use_xlm_tokenizer

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def _map_category_names(self, token_results: list[TokenClassResult]) -> list[TokenClassResult]:
        for result in token_results:
            result.class_name = self.categories.categories[result.class_id + 1]
            output = self.categories.disentangle_token_class_and_tag(result.class_name)
            if output is not None:
                token_class, tag = output
                result.semantic_name = token_class
                result.bio_tag = tag
            else:
                result.semantic_name = result.class_name
            result.class_id += 1
        return token_results

    def _validate_encodings(
        self, **encodings: Any
    ) -> tuple[list[list[str]], list[str], torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
        image_ids = encodings.get("image_ids", [])
        ann_ids = encodings.get("ann_ids")
        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        tokens = encodings.get("tokens")

        assert isinstance(ann_ids, list), type(ann_ids)
        if len(set(image_ids)) > 1:
            raise ValueError("HFLmTokenClassifier accepts for inference only one image.")
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.device)
        else:
            raise ValueError(f"input_ids must be list but is {type(input_ids)}")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(self.device)
        else:
            raise ValueError(f"attention_mask must be list but is {type(attention_mask)}")
        if isinstance(token_type_ids, torch.Tensor):
            token_type_ids = token_type_ids.to(self.device)
        else:
            raise ValueError(f"token_type_ids must be list but is {type(token_type_ids)}")
        if not isinstance(tokens, list):
            raise ValueError(f"tokens must be list but is {type(tokens)}")

        return ann_ids, image_ids, input_ids, attention_mask, token_type_ids, tokens

    def clone(self) -> HFLmTokenClassifierBase:
        return self.__class__(
            self.path_config,
            self.path_weights,
            self.categories.categories_semantics,
            self.categories.categories_bio,
            self.categories.get_categories(),
            self.device,
            self.use_xlm_tokenizer,
        )

    @staticmethod
    def get_name(path_weights: PathLikeOrStr, architecture: str) -> str:
        """Returns the name of the model"""
        return f"Transformers_{architecture}_" + "_".join(Path(path_weights).parts[-2:])

    @staticmethod
    def get_tokenizer_class_name(model_class_name: str, use_xlm_tokenizer: bool) -> str:
        """
        A refinement for adding the tokenizer class name to the model configs.

        Args:
            model_class_name: The model name, e.g. `model.__class__.__name__`
            use_xlm_tokenizer: Whether to use a `XLM` tokenizer.

        Returns:
            The name of the tokenizer class.
        """
        tokenizer = get_tokenizer_from_model_class(model_class_name, use_xlm_tokenizer)
        return tokenizer.__class__.__name__

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """Returns the mapping function to convert images into raw features."""
        return "image_to_raw_lm_features"

    @staticmethod
    def image_to_features_mapping() -> str:
        """Returns the mapping function to convert images into features."""
        return "image_to_lm_features"


class HFLmTokenClassifier(HFLmTokenClassifierBase):
    """
    A wrapper class for `transformers.XLMRobertaForTokenClassification` and similar models to use within a pipeline
    component. Check <https://huggingface.co/docs/transformers/model_doc/xlm-roberta> for documentation of the
    model itself.
    Note that this model is equipped with a head that is only useful for classifying the tokens. For sequence
    classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
        roberta = XLMRobertaForTokenClassification("path/to/config.json","path/to/model.bin",
                                              categories=["first_name", "surname", "street"])

        # token classification service
        roberta_service = LMTokenClassifierService(tokenizer,roberta)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,roberta_service])

        path = "path/to/some/form"
        df = pipe.analyze(path=path)

        for dp in df:
            ...
        ```
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[int, TypeOrStr]] = None,
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        use_xlm_tokenizer: bool = True,
    ):
        """
        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact
            categories_semantics: A dict with key (indices) and values (category names) for NER semantics, i.e. the
                                 entities self. To be consistent with detectors use only values `>0`. Conversion will
                                 be done internally.
            categories_bio: A dict with key (indices) and values (category names) for `NER` tags (i.e. BIO). To be
                           consistent with detectors use only values>0. Conversion will be done internally.
            categories: If you have a pre-trained model you can pass a complete dict of NER categories
            device: The device (cpu,"cuda"), where to place the model.
            use_xlm_tokenizer: Do not change this value unless you pre-trained a bert-like model with a different
                              Tokenizer.
        """
        super().__init__(
            path_config_json, path_weights, categories_semantics, categories_bio, categories, device, use_xlm_tokenizer
        )
        self.name = self.get_name(path_weights, "bert-like-token-classification")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Launch inference on bert-like models for token classification. Pass the following arguments

        Args:
            encodings: input_ids: Token converted to ids to be taken from `LayoutLMTokenizer`
                       attention_mask: The associated attention masks from padded sequences taken from
                                       `LayoutLMTokenizer`
                       token_type_ids: Torch tensor of token type ids taken from `LayoutLMTokenizer`
                       boxes: Torch tensor of bounding boxes of type `xyxy`
                       tokens: List of original tokens taken from `LayoutLMTokenizer`

        Returns:
            A list of `TokenClassResult`s
        """

        ann_ids, _, input_ids, attention_mask, token_type_ids, tokens = self._validate_encodings(**encodings)
        results = predict_token_classes_from_lm(ann_ids, input_ids, attention_mask, token_type_ids, tokens, self.model)
        return self._map_category_names(results)

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> XLMRobertaForTokenClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to .json config file
            path_weights: path to model artifact

        Returns:
            `nn.Module`
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return XLMRobertaForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLmSequenceClassifierBase(LMSequenceClassifier, ABC):
    """
    Abstract base class for wrapping Bert-type models for sequence classification into the deepdoctection framework.
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
    ):
        self.path_config = Path(path_config_json)
        self.path_weights = Path(path_weights)
        self.categories = ModelCategories(init_categories=categories)

        self.device = get_torch_device(device)

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> HFLmSequenceClassifierBase:
        return self.__class__(self.path_config, self.path_weights, self.categories.get_categories(), self.device)

    def _validate_encodings(
        self, **encodings: Union[list[list[str]], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.device)
        else:
            raise ValueError(f"input_ids must be list but is {type(input_ids)}")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(self.device)
        else:
            raise ValueError(f"attention_mask must be list but is {type(attention_mask)}")
        if isinstance(token_type_ids, torch.Tensor):
            token_type_ids = token_type_ids.to(self.device)
        else:
            raise ValueError(f"token_type_ids must be list but is {type(token_type_ids)}")

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        return input_ids, attention_mask, token_type_ids

    @staticmethod
    def get_name(path_weights: PathLikeOrStr, architecture: str) -> str:
        """
        Returns the name of the model

        Args:
            path_weights: Path to model weights
            architecture: Architecture name

        Returns:
            str: Model name
        """
        return f"Transformers_{architecture}_" + "_".join(Path(path_weights).parts[-2:])

    @staticmethod
    def get_tokenizer_class_name(model_class_name: str, use_xlm_tokenizer: bool) -> str:
        """
        A refinement for adding the tokenizer class name to the model configs.

        Args:
            model_class_name: The model name, e.g. `model.__class__.__name__`
            use_xlm_tokenizer: Whether to use a `XLM` tokenizer.

        Returns:
            str: Tokenizer class name
        """
        tokenizer = get_tokenizer_from_model_class(model_class_name, use_xlm_tokenizer)
        return tokenizer.__class__.__name__

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """
        Returns the mapping function to convert images into raw features.

        Returns:
            str: Name of the mapping function
        """
        return "image_to_raw_lm_features"

    @staticmethod
    def image_to_features_mapping() -> str:
        """
        Returns the mapping function to convert images into features.

        Returns:
            str: Name of the mapping function
        """
        return "image_to_lm_features"


class HFLmSequenceClassifier(HFLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.XLMRobertaForSequenceClassification` and similar models to use within a pipeline
    component. Check <https://huggingface.co/docs/transformers/model_doc/xlm-roberta> for documentation of the
    model itself.
    Note that this model is equipped with a head that is only useful for classifying the input sequence. For token
    classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
        roberta = HFLmSequenceClassifier("path/to/config.json","path/to/model.bin",
                                              categories=["handwritten", "presentation", "resume"])

        # token classification service
        roberta_service = LMSequenceClassifierService(tokenizer,roberta)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,roberta_service])

        path = "path/to/some/form"
        df = pipe.analyze(path=path)

        for dp in df:
            ...
        ```
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        use_xlm_tokenizer: bool = True,
    ):
        super().__init__(path_config_json, path_weights, categories, device)
        self.name = self.get_name(path_weights, "bert-like-sequence-classification")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.use_xlm_tokenizer = use_xlm_tokenizer
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids = self._validate_encodings(**encodings)

        result = predict_sequence_classes_from_lm(
            input_ids,
            attention_mask,
            token_type_ids,
            self.model,
        )

        result.class_id += 1
        result.class_name = self.categories.categories[result.class_id]
        return result

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> XLMRobertaForSequenceClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to .json config file
            path_weights: path to model artifact

        Returns:
            `XLMRobertaForSequenceClassification`
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        return XLMRobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.

        Returns:
            JsonDict: Dictionary with default arguments
        """
        return {}

    def clear_model(self) -> None:
        self.model = None


class HFLmLanguageDetector(LanguageDetector):
    """
    Language detector using HuggingFace's `XLMRobertaForSequenceClassification`.

    This class wraps a multilingual sequence classification model (XLMRobertaForSequenceClassification)
    for language detection tasks. Input text is tokenized and truncated/padded to a maximum length of 512 tokens.
    The prediction returns a `DetectionResult` containing the detected language code and its confidence score.
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        use_xlm_tokenizer: bool = True,
    ):
        super().__init__()
        self.path_config = Path(path_config_json)
        self.path_weights = Path(path_weights)
        self.categories = ModelCategories(init_categories=categories)
        self.device = get_torch_device(device)
        self.use_xlm_tokenizer = use_xlm_tokenizer
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
        self.name = self.get_name(path_weights, "bert-like-language-detection")
        self.model_id = self.get_model_id()

    def predict(self, text_string: str) -> DetectionResult:
        """
        Predict the language of the input sequence.

        Args:
            text_string: The input text sequence to classify.

        Returns:
            DetectionResult: The detected language and its confidence score.
        """
        encoding = self.tokenizer(
            text_string,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        else:
            token_type_ids = torch.zeros_like(input_ids)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            score, class_id_tensor = torch.max(probs, dim=-1)
            class_id = int(class_id_tensor.item() + 1)
            lang = self.categories.categories[class_id]

        return DetectionResult(class_name=lang, score=float(score.item()))

    def clear_model(self) -> None:
        self.model = None

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> XLMRobertaForSequenceClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to .json config file
            path_weights: path to model artifact

        Returns:
            `XLMRobertaForSequenceClassification`
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        return XLMRobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )

    def clone(self) -> HFLmLanguageDetector:
        return self.__class__(
            self.path_config, self.path_weights, self.categories.get_categories(), self.device, self.use_xlm_tokenizer
        )

    @staticmethod
    def get_name(path_weights: PathLikeOrStr, architecture: str) -> str:
        """
        Returns the name of the model

        Args:
            path_weights: Path to model weights
            architecture: Architecture name

        Returns:
            str: Model name
        """
        return f"Transformers_{architecture}_" + "_".join(Path(path_weights).parts[-2:])


if TYPE_CHECKING:
    LmTokenModels: TypeAlias = Union[HFLmTokenClassifier,]
    LmSequenceModels: TypeAlias = Union[HFLmSequenceClassifier,]
