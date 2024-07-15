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
Wrapper for the Hugging Face Language Model for sequence and token  classification
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

from lazy_imports import try_import

from ..utils.file_utils import get_pytorch_requirement, get_transformers_requirement
from ..utils.settings import TypeOrStr
from ..utils.types import JsonDict, PathLikeOrStr, Requirement
from .base import LMSequenceClassifier, ModelCategories, SequenceClassResult
from .hflayoutlm import get_tokenizer_from_model_class
from .pt.ptutils import get_torch_device

with try_import() as pt_import_guard:
    import torch
    import torch.nn.functional as F

with try_import() as tr_import_guard:
    from transformers import PretrainedConfig, XLMRobertaForSequenceClassification


def predict_sequence_classes(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    model: Union[XLMRobertaForSequenceClassification],
) -> SequenceClassResult:
    """
    :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
    :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
    :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
    :param model: layoutlm model for sequence classification
    :return: SequenceClassResult
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    score = torch.max(F.softmax(outputs.logits)).tolist()
    sequence_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()

    return SequenceClassResult(class_id=sequence_class_predictions, score=float(score))  # type: ignore


class HFLmSequenceClassifierBase(LMSequenceClassifier, ABC):
    """
    Abstract base class for wrapping Bert-type models  for sequence classification into the deepdoctection framework.
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
        """Returns the name of the model"""
        return f"Transformers_{architecture}_" + "_".join(Path(path_weights).parts[-2:])

    @staticmethod
    def get_tokenizer_class_name(model_class_name: str, use_xlm_tokenizer: bool) -> str:
        """A refinement for adding the tokenizer class name to the model configs.

        :param model_class_name: The model name, e.g. model.__class__.__name__
        :param use_xlm_tokenizer: Whether to use a XLM tokenizer.
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


class HFLmSequenceClassifier(HFLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.XLMRobertaForSequenceClassification` and similar models to use within a pipeline
    component. Check <https://huggingface.co/docs/transformers/model_doc/xlm-roberta> for documentation of the
    model itself.
    Note that this model is equipped with a head that is only useful for classifying the input sequence. For token
    classification and other things please use another model of the family.

    **Example**

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
        self.name = self.get_name(path_weights, "bert-like")
        self.model_id = self.get_model_id()
        self.model = self.get_wrapped_model(path_config_json, path_weights)
        self.model.to(self.device)
        self.model.config.tokenizer_class = self.get_tokenizer_class_name(
            self.model.__class__.__name__, use_xlm_tokenizer
        )

    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids = self._validate_encodings(**encodings)

        result = predict_sequence_classes(
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

        :param path_config_json: path to .json config file
        :param path_weights: path to model artifact
        :return: 'nn.Module'
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
        """
        return {}

    def clear_model(self) -> None:
        self.model = None
