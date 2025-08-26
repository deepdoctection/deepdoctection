# -*- coding: utf-8 -*-
# File: hflayoutlm.py

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
HF Layoutlm models.
"""
from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from lazy_imports import try_import
from typing_extensions import TypeAlias

from ..utils.file_utils import get_pytorch_requirement, get_transformers_requirement
from ..utils.settings import TypeOrStr
from ..utils.types import JsonDict, PathLikeOrStr, Requirement
from .base import (
    LMSequenceClassifier,
    LMTokenClassifier,
    ModelCategories,
    NerModelCategories,
    SequenceClassResult,
    TokenClassResult,
)
from .pt.ptutils import get_torch_device

with try_import() as pt_import_guard:
    import torch
    import torch.nn.functional as F

with try_import() as tr_import_guard:
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from transformers import (
        LayoutLMForSequenceClassification,
        LayoutLMForTokenClassification,
        LayoutLMTokenizerFast,
        LayoutLMv2Config,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv3Config,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3ForTokenClassification,
        LiltForSequenceClassification,
        LiltForTokenClassification,
        PretrainedConfig,
        RobertaTokenizerFast,
        XLMRobertaTokenizerFast,
    )

if TYPE_CHECKING:
    HfLayoutTokenModels: TypeAlias = Union[
        LayoutLMForTokenClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv3ForTokenClassification,
        LiltForTokenClassification,
    ]

    HfLayoutSequenceModels: TypeAlias = Union[
        LayoutLMForSequenceClassification,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv3ForSequenceClassification,
        LiltForSequenceClassification,
    ]


def get_tokenizer_from_model_class(model_class: str, use_xlm_tokenizer: bool) -> Any:
    """
    We do not use the tokenizer for a particular model that the transformer library provides. Thie mapping therefore
    returns the tokenizer that should be used for a particular model.

    Args:
        model_class: The model as stated in the transformer library.
        use_xlm_tokenizer: True if one uses the `LayoutXLM`. (The model cannot be distinguished from `LayoutLMv2`).

    Returns:
        Tokenizer instance to use.
    """
    return {
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
        ("LiltForTokenClassification", True): XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base"),
        ("LiltForTokenClassification", False): RobertaTokenizerFast.from_pretrained(
            "roberta-base", add_prefix_space=True
        ),
        ("LiltForSequenceClassification", True): XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base"),
        ("LiltForSequenceClassification", False): RobertaTokenizerFast.from_pretrained(
            "roberta-base", add_prefix_space=True
        ),
        ("XLMRobertaForSequenceClassification", True): XLMRobertaTokenizerFast.from_pretrained(
            "FacebookAI/xlm-roberta-base"
        ),
        ("XLMRobertaForTokenClassification", True): XLMRobertaTokenizerFast.from_pretrained(
            "FacebookAI/xlm-roberta-base"
        ),
    }[(model_class, use_xlm_tokenizer)]


def predict_token_classes_from_layoutlm(
    uuids: list[list[str]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    boxes: torch.Tensor,
    tokens: list[list[str]],
    model: HfLayoutTokenModels,
    images: Optional[torch.Tensor] = None,
) -> list[TokenClassResult]:
    """
    Args:
        uuids: A list of uuids that correspond to a word that induces the resulting token
        input_ids: Token converted to ids to be taken from `LayoutLMTokenizer`
        attention_mask: The associated attention masks from padded sequences taken from `LayoutLMTokenizer`
        token_type_ids: Torch tensor of token type ids taken from `LayoutLMTokenizer`
        boxes: Torch tensor of bounding boxes of type 'xyxy'
        tokens: List of original tokens taken from `LayoutLMTokenizer`
        model: layoutlm model for token classification
        images: A list of torch image tensors or None

    Returns:
        A list of `TokenClassResult`s
    """

    if images is None:
        outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    elif isinstance(model, LayoutLMv2ForTokenClassification):
        outputs = model(
            input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids, image=images
        )
    elif isinstance(model, LayoutLMv3ForTokenClassification):
        outputs = model(
            input_ids=input_ids,
            bbox=boxes,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=images,
        )
    else:
        raise ValueError(f"Cannot call model {type(model)}")

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


def predict_sequence_classes_from_layoutlm(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    boxes: torch.Tensor,
    model: HfLayoutSequenceModels,
    images: Optional[torch.Tensor] = None,
) -> SequenceClassResult:
    """
    Args:
        input_ids: Token converted to ids to be taken from `LayoutLMTokenizer`
        attention_mask: The associated attention masks from padded sequences taken from `LayoutLMTokenizer`
        token_type_ids: Torch tensor of token type ids taken from `LayoutLMTokenizer`
        boxes: Torch tensor of bounding boxes of type `xyxy`
        model: layoutlm model for sequence classification
        images: A list of torch image tensors or None

    Returns:
        SequenceClassResult
    """

    if images is None:
        outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    elif isinstance(model, LayoutLMv2ForSequenceClassification):
        outputs = model(
            input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids, image=images
        )
    elif isinstance(model, LayoutLMv3ForSequenceClassification):
        outputs = model(
            input_ids=input_ids,
            bbox=boxes,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=images,
        )
    else:
        raise ValueError(f"Cannot call model {type(model)}")

    score = torch.max(F.softmax(outputs.logits)).tolist()
    sequence_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()

    return SequenceClassResult(class_id=sequence_class_predictions, score=float(score))  # type: ignore


class HFLayoutLmTokenClassifierBase(LMTokenClassifier, ABC):
    """
    Abstract base class for wrapping `LayoutLM` models for token classification into the framework.
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
    ) -> tuple[list[list[str]], list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
        image_ids = encodings.get("image_ids", [])
        ann_ids = encodings.get("ann_ids")
        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        boxes = encodings.get("bbox")
        tokens = encodings.get("tokens")

        assert isinstance(ann_ids, list), type(ann_ids)
        if len(set(image_ids)) > 1:
            raise ValueError("HFLayoutLmTokenClassifier accepts for inference only one image.")
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
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.to(self.device)
        else:
            raise ValueError(f"boxes must be list but is {type(boxes)}")
        if not isinstance(tokens, list):
            raise ValueError(f"tokens must be list but is {type(tokens)}")

        return ann_ids, image_ids, input_ids, attention_mask, token_type_ids, boxes, tokens

    def clone(self) -> HFLayoutLmTokenClassifierBase:
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
        return "image_to_raw_layoutlm_features"

    @staticmethod
    def image_to_features_mapping() -> str:
        """Returns the mapping function to convert images into features."""
        return "image_to_layoutlm_features"


class HFLayoutLmTokenClassifier(HFLayoutLmTokenClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMForTokenClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/model_doc/layoutlm> for documentation of the model itself.
    Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
        layoutlm = HFLayoutLmTokenClassifier("path/to/config.json","path/to/model.bin",
                                              categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                           'E-header', 'E-question', 'I-answer', 'I-header',
                                                           'I-question', 'O', 'S-answer', 'S-header',
                                                           'S-question'])

        # token classification service
        layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

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
        use_xlm_tokenizer: bool = False,
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
            use_xlm_tokenizer: Do not change this value unless you pre-trained a LayoutLM model with a different
                              Tokenizer.
        """
        super().__init__(
            path_config_json, path_weights, categories_semantics, categories_bio, categories, device, use_xlm_tokenizer
        )
        self.name = self.get_name(path_weights, "LayoutLM")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

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

        ann_ids, _, input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        results = predict_token_classes_from_layoutlm(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, None
        )

        return self._map_category_names(results)

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> LayoutLMForTokenClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to .json config file
            path_weights: path to model artifact

        Returns:
            `nn.Module`
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return LayoutLMForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLayoutLmv2TokenClassifier(HFLayoutLmTokenClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv2ForTokenClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv2>  for documentation of the model
    itself. Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    Note, that you must use `LayoutLMTokenizerFast` as tokenizer. `LayoutLMv2TokenizerFast` will not be accepted.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
        layoutlm = HFLayoutLmv2TokenClassifier("path/to/config.json","path/to/model.bin",
                                              categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                           'E-header', 'E-question', 'I-answer', 'I-header',
                                                           'I-question', 'O', 'S-answer', 'S-header',
                                                           'S-question'])

        # token classification service
        layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

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
                           consistent with detectors use only values>0. Conversion will be done internally.
            categories: If you have a pre-trained model you can pass a complete dict of `NER` categories
            device: The device (cpu,"cuda"), where to place the model.
            use_xlm_tokenizer: Set to True if you use a LayoutXLM model. If you use a `LayoutLMv2` model keep the
                              default value.
        """
        super().__init__(
            path_config_json, path_weights, categories_semantics, categories_bio, categories, device, use_xlm_tokenizer
        )
        self.name = self.get_name(path_weights, "LayoutLMv2")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Launch inference on `LayoutLm` for token classification. Pass the following arguments

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

        ann_ids, _, input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        images = encodings.get("image")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")
        results = predict_token_classes_from_layoutlm(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, images
        )

        return self._map_category_names(results)

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {"image_width": 224, "image_height": 224}

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> LayoutLMv2ForTokenClassification:
        """
        Get the inner (wrapped) model.

        :param path_config_json: path to .json config file
        :param path_weights: path to model artifact
        :return: 'nn.Module'
        """
        config = LayoutLMv2Config.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return LayoutLMv2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLayoutLmv3TokenClassifier(HFLayoutLmTokenClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv3ForTokenClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv3>  for documentation of the model
    itself. Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    Note, that you must use `RobertaTokenizerFast` as tokenizer. `LayoutLMv3TokenizerFast` will not be accepted.

    **Example**

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            layoutlm = HFLayoutLmv3TokenClassifier("path/to/config.json","path/to/model.bin",
                                                  categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                               'E-header', 'E-question', 'I-answer', 'I-header',
                                                               'I-question', 'O', 'S-answer', 'S-header',
                                                               'S-question'])

            # token classification service
            layoutlm_service = LMTokenClassifierService(tokenizer, layoutlm)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
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
        :param path_config_json: path to .json config file
        :param path_weights: path to model artifact
        :param categories_semantics: A dict with key (indices) and values (category names) for NER semantics, i.e. the
                                     entities self. To be consistent with detectors use only values >0. Conversion will
                                     be done internally.
        :param categories_bio: A dict with key (indices) and values (category names) for NER tags (i.e. BIO). To be
                               consistent with detectors use only values>0. Conversion will be done internally.
        :param categories: If you have a pre-trained model you can pass a complete dict of NER categories
        :param device: The device (cpu,"cuda"), where to place the model.
        :param use_xlm_tokenizer: Do not change this value unless you pre-trained a LayoutLMv3 model with a different
                                  tokenizer.
        """
        super().__init__(
            path_config_json, path_weights, categories_semantics, categories_bio, categories, device, use_xlm_tokenizer
        )
        self.name = self.get_name(path_weights, "LayoutLMv3")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

        `input_ids:` Token converted to ids to be taken from `LayoutLMTokenizer`
        `attention_mask:` The associated attention masks from padded sequences taken from `LayoutLMTokenizer`
        `token_type_ids:` Torch tensor of token type ids taken from `LayoutLMTokenizer`
        `boxes:` Torch tensor of bounding boxes of type 'xyxy'
        `tokens:` List of original tokens taken from `LayoutLMTokenizer`

        :return: A list of TokenClassResults
        """

        ann_ids, _, input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        images = encodings.get("pixel_values")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")
        results = predict_token_classes_from_layoutlm(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, images
        )

        return self._map_category_names(results)

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {
            "image_width": 224,
            "image_height": 224,
            "color_mode": "RGB",
            "pixel_mean": np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32),
            "pixel_std": np.array(IMAGENET_DEFAULT_STD, dtype=np.float32),
        }

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> LayoutLMv3ForTokenClassification:
        """
        Get the inner (wrapped) model.

        :param path_config_json: path to .json config file
        :param path_weights: path to model artifact
        :return: 'nn.Module'
        """
        config = LayoutLMv3Config.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return LayoutLMv3ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLayoutLmSequenceClassifierBase(LMSequenceClassifier, ABC):
    """
    Abstract base class for wrapping LayoutLM models  for sequence classification into the deepdoctection framework.
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        use_xlm_tokenizer: bool = False,
    ):
        self.path_config = Path(path_config_json)
        self.path_weights = Path(path_weights)
        self.categories = ModelCategories(init_categories=categories)

        self.device = get_torch_device(device)
        self.use_xlm_tokenizer = use_xlm_tokenizer

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> HFLayoutLmSequenceClassifierBase:
        return self.__class__(
            self.path_config, self.path_weights, self.categories.get_categories(), self.device, self.use_xlm_tokenizer
        )

    def _validate_encodings(
        self, **encodings: Union[list[list[str]], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        boxes = encodings.get("bbox")

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
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.to(self.device)
        else:
            raise ValueError(f"boxes must be list but is {type(boxes)}")

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        boxes = boxes.to(self.device)
        return input_ids, attention_mask, token_type_ids, boxes

    @staticmethod
    def get_name(path_weights: PathLikeOrStr, architecture: str) -> str:
        """Returns the name of the model"""
        return f"Transformers_{architecture}_" + "_".join(Path(path_weights).parts[-2:])

    @staticmethod
    def get_tokenizer_class_name(model_class_name: str, use_xlm_tokenizer: bool) -> str:
        """A refinement for adding the tokenizer class name to the model configs.

        Args:
            model_class_name: The model name, e.g. `model.__class__.__name__`
            use_xlm_tokenizer: Whether to use a `XLM` tokenizer.
        """
        tokenizer = get_tokenizer_from_model_class(model_class_name, use_xlm_tokenizer)
        return tokenizer.__class__.__name__

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """Returns the mapping function to convert images into raw features."""
        return "image_to_raw_layoutlm_features"

    @staticmethod
    def image_to_features_mapping() -> str:
        """Returns the mapping function to convert images into features."""
        return "image_to_layoutlm_features"


class HFLayoutLmSequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/model_doc/layoutlm> for documentation of the model itself.
    Note that this model is equipped with a head that is only useful for classifying the input sequence. For token
    classification and other things please use another model of the family.

    Example:
        ```python
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
        ```
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        use_xlm_tokenizer: bool = False,
    ):
        """
        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact
            categories: A dict with key (indices) and values (category names) for sequence classification.
                        To be consistent with detectors use only values `>0`. Conversion will be done internally.
            device: The device ("cpu","cuda"), where to place the model.
            use_xlm_tokenizer: Do not change this value unless you pre-trained a `LayoutLM` model with a different
                              Tokenizer.
        """
        super().__init__(path_config_json, path_weights, categories, device, use_xlm_tokenizer)
        self.name = self.get_name(path_weights, "LayoutLM")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        """
        Launch inference on LayoutLm for sequence classification. Pass the following arguments

        Args:
            encodings: input_ids: Token converted to ids to be taken from `LayoutLMTokenizer`
                       attention_mask: The associated attention masks from padded sequences taken from
                                       `LayoutLMTokenizer`
                       token_type_ids: Torch tensor of token type ids taken from `LayoutLMTokenizer`
                       boxes: Torch tensor of bounding boxes of type `xyxy`
        """
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)

        result = predict_sequence_classes_from_layoutlm(
            input_ids,
            attention_mask,
            token_type_ids,
            boxes,
            self.model,
        )

        result.class_id += 1
        result.class_name = self.categories.categories[result.class_id]
        return result

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> LayoutLMForSequenceClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact

        Returns:
            'nn.Module'
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return LayoutLMForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLayoutLmv2SequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv2ForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv2> for documentation of the model
    itself. Note that this model is equipped with a head that is only useful for classifying the input sequence. For
    token classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
        layoutlm = HFLayoutLmv2SequenceClassifier("path/to/config.json","path/to/model.bin",
                                              categories=["handwritten", "presentation", "resume"])

        # token classification service
        layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

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
        use_xlm_tokenizer: bool = False,
    ):
        """
        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact
            categories: A dict with key (indices) and values (category names) for sequence classification.
                        To be consistent with detectors use only values `>0`. Conversion will be done internally.
            device: The device ("cpu","cuda"), where to place the model.
            use_xlm_tokenizer: Do not change this value unless you pre-trained a `LayoutLM` model with a different
                              Tokenizer.
        """
        super().__init__(path_config_json, path_weights, categories, device, use_xlm_tokenizer)
        self.name = self.get_name(path_weights, "LayoutLMv2")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        """
        Launch inference on LayoutLm for sequence classification. Pass the following arguments

        Args:
            encodings: input_ids: Token converted to ids to be taken from `LayoutLMTokenizer`
                       attention_mask: The associated attention masks from padded sequences taken from
                                       `LayoutLMTokenizer`
                       token_type_ids: Torch tensor of token type ids taken from `LayoutLMTokenizer`
                       boxes: Torch tensor of bounding boxes of type `xyxy`
        """
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)
        images = encodings.get("image")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")

        result = predict_sequence_classes_from_layoutlm(input_ids,
                                                        attention_mask,
                                                        token_type_ids,
                                                        boxes,
                                                        self.model,
                                                        images)

        result.class_id += 1
        result.class_name = self.categories.categories[result.class_id]
        return result

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {"image_width": 224, "image_height": 224}

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> LayoutLMv2ForSequenceClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact

        Returns:
            'nn.Module'
        """
        config = LayoutLMv2Config.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return LayoutLMv2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLayoutLmv3SequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv3ForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv3> for documentation of the model
    itself. Note that this model is equipped with a head that is only useful for classifying the input sequence. For
    token classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        layoutlm = HFLayoutLmv3SequenceClassifier("path/to/config.json","path/to/model.bin",
                                              categories=["handwritten", "presentation", "resume"])

        # token classification service
        layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

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
        use_xlm_tokenizer: bool = False,
    ):
        super().__init__(path_config_json, path_weights, categories, device, use_xlm_tokenizer)
        self.name = self.get_name(path_weights, "LayoutLMv3")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)
        images = encodings.get("pixel_values")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")

        result = predict_sequence_classes_from_layoutlm(input_ids,
                                                        attention_mask,
                                                        token_type_ids,
                                                        boxes,
                                                        self.model,
                                                        images)

        result.class_id += 1
        result.class_name = self.categories.categories[result.class_id]
        return result

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {
            "image_width": 224,
            "image_height": 224,
            "color_mode": "RGB",
            "pixel_mean": np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32),
            "pixel_std": np.array(IMAGENET_DEFAULT_STD, dtype=np.float32),
        }

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr
    ) -> LayoutLMv3ForSequenceClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact

        Returns:
            'nn.Module'
        """
        config = LayoutLMv3Config.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config_json))
        return LayoutLMv3ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=os.fspath(path_weights), config=config
        )

    def clear_model(self) -> None:
        self.model = None


class HFLiltTokenClassifier(HFLayoutLmTokenClassifierBase):
    """
    A wrapper class for `transformers.LiltForTokenClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/model_doc/lilt> for documentation of the model itself.
    Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and token classifier
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        lilt = HFLiltTokenClassifier("path/to/config.json","path/to/model.bin",
                                              categories= ['B-answer', 'B-header', 'B-question', 'E-answer',
                                                           'E-header', 'E-question', 'I-answer', 'I-header',
                                                           'I-question', 'O', 'S-answer', 'S-header',
                                                           'S-question'])

        # token classification service
        lilt_service = LMTokenClassifierService(tokenizer,lilt)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,lilt_service])

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
        use_xlm_tokenizer: bool = False,
    ):
        """
        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact
            categories_semantics: A dict with key (indices) and values (category names) for `NER` semantics, i.e. the
                                 entities self. To be consistent with detectors use only values `>0`. Conversion will
                                 be done internally.
            categories_bio: A dict with key (indices) and values (category names) for NER tags (i.e. `BIO`). To be
                           consistent with detectors use only values>0. Conversion will be done internally.
            categories: If you have a pre-trained model you can pass a complete dict of `NER` categories
            device: The device ("cpu","cuda"), where to place the model.
        """

        super().__init__(
            path_config_json, path_weights, categories_semantics, categories_bio, categories, device, use_xlm_tokenizer
        )
        self.name = self.get_name(path_weights, "LiLT")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

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

        ann_ids, _, input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        results = predict_token_classes_from_layoutlm(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, None
        )

        return self._map_category_names(results)

    @staticmethod
    def get_wrapped_model(path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr) -> LiltForTokenClassification:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact

        Returns:
            `nn.Module`
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        return LiltForTokenClassification.from_pretrained(pretrained_model_name_or_path=path_weights, config=config)

    def clear_model(self) -> None:
        self.model = None


class HFLiltSequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LiLTForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/model_doc/lilt> for documentation of the model itself.
    Note that this model is equipped with a head that is only useful for classifying the input sequence. For token
    classification and other things please use another model of the family.

    Example:
        ```python
        # setting up compulsory ocr service
        tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
        tess = TesseractOcrDetector(tesseract_config_path)
        ocr_service = TextExtractionService(tess)

        # hf tokenizer and sequence classifier
        tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
        lilt = HFLiltSequenceClassifier("path/to/config.json",
                                            "path/to/model.bin",
                                            categories=["handwritten", "presentation", "resume"])

        # sequence classification service
        lilt_service = LMSequenceClassifierService(tokenizer,lilt)

        pipe = DoctectionPipe(pipeline_component_list=[ocr_service,lilt_service])

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
        use_xlm_tokenizer: bool = False,
    ):
        super().__init__(path_config_json, path_weights, categories, device, use_xlm_tokenizer)
        self.name = self.get_name(path_weights, "LiLT")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, self.use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)

        result = predict_sequence_classes_from_layoutlm(
            input_ids,
            attention_mask,
            token_type_ids,
            boxes,
            self.model,
        )

        result.class_id += 1
        result.class_name = self.categories.categories[result.class_id]
        return result

    @staticmethod
    def get_wrapped_model(path_config_json: PathLikeOrStr, path_weights: PathLikeOrStr) -> Any:
        """
        Get the inner (wrapped) model.

        Args:
            path_config_json: path to `.json` config file
            path_weights: path to model artifact

        Returns:
            `nn.Module`
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        return LiltForSequenceClassification.from_pretrained(pretrained_model_name_or_path=path_weights, config=config)

    def clear_model(self) -> None:
        self.model = None


if TYPE_CHECKING:
    LayoutTokenModels: TypeAlias = Union[
        HFLayoutLmTokenClassifier,
        HFLayoutLmv2TokenClassifier,
        HFLayoutLmv3TokenClassifier,
        HFLiltTokenClassifier,
    ]

    LayoutSequenceModels: TypeAlias = Union[
        HFLayoutLmSequenceClassifier,
        HFLayoutLmv2SequenceClassifier,
        HFLayoutLmv3SequenceClassifier,
        HFLiltSequenceClassifier,
    ]
