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
HF Layoutlm model for diverse downstream tasks.
"""

from copy import copy
from typing import Dict, List, Optional, Sequence, Union, Mapping, Literal

from ..utils.detection_types import Requirement
from ..utils.file_utils import (
    get_pytorch_requirement,
    get_transformers_requirement,
    pytorch_available,
    transformers_available,
)
from ..utils.settings import names
from .base import LMTokenClassifier, PredictorBase, TokenClassResult, LMSequenceClassifier, SequenceClassResult
from .pt.ptutils import set_torch_auto_device

if pytorch_available():
    import torch
    from torch import Tensor  # pylint: disable=W0611
    import torch.nn.functional as F

if transformers_available():
    from transformers import LayoutLMForTokenClassification, PretrainedConfig, LayoutLMForSequenceClassification


def predict_token_classes(
    uuids: List[str],
    input_ids: "Tensor",
    attention_mask: "Tensor",
    token_type_ids: "Tensor",
    boxes: "Tensor",
    tokens: List[str],
    model: "LayoutLMForTokenClassification",
) -> List[TokenClassResult]:
    """
    :param uuids: A list of uuids that correspond to a word that induces the resulting token
    :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
    :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
    :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
    :param boxes: Torch tensor of bounding boxes of type 'xyxy'
    :param tokens: List of original tokens taken from LayoutLMTokenizer
    :param model: layoutlm model for token classification
    :return: A list of TokenClassResults
    """
    outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    score = torch.max(F.softmax(outputs.logits.squeeze(),dim=1),dim=1)
    token_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()
    input_ids_list = input_ids.squeeze().tolist()
    return [
        TokenClassResult(uuid=out[0], token_id=out[1], class_id=out[2], token=out[3], score= out[4].tolist())
        for out in zip(uuids, input_ids_list, token_class_predictions, tokens, score[0])
    ]


def predict_sequence_classes(
        input_ids: "Tensor",
    attention_mask: "Tensor",
    token_type_ids: "Tensor",
    boxes: "Tensor",
    model: "LayoutLMForSequenceClassification",
) -> SequenceClassResult:

    outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    score = torch.max(F.softmax(outputs.logits)).tolist()
    token_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()

    return SequenceClassResult(class_id=token_class_predictions, score= score)


class HFLayoutLmTokenClassifier(LMTokenClassifier):
    """
    A wrapper class for :class:`transformers.LayoutLMForTokenClassification` to use within a pipeline component.
    Check https://huggingface.co/docs/transformers/model_doc/layoutlm for documentation of the model itself.
    Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    **Example**

        .. code-block:: python

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizer.from_pretrained("mrm8488/layoutlm-finetuned-funsd")
            layoutlm = HFLayoutLmTokenClassifier("path/to/config.json","path/to/model.bin",
                                                  categories_explicit= ['B-ANSWER', 'B-HEAD', 'B-QUESTION', 'E-ANSWER',
                                                                        'E-HEAD', 'E-QUESTION', 'I-ANSWER', 'I-HEAD',
                                                                        'I-QUESTION', 'O', 'S-ANSWER', 'S-HEAD',
                                                                        'S-QUESTION'])

            # token classification service
            layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)

            pipe = DoctectionPipe(pipeline_component_list=[ocr_service,layoutlm_service])

            path = "path/to/some/form"
            df = pipe.analyze(path=path)

            for dp in df:
                ...
    """

    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        categories_semantics: Optional[Sequence[str]] = None,
        categories_bio: Optional[Sequence[str]] = None,
        categories: Optional[Mapping[str, str]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None
    ):
        """
        :param categories_semantics: A dict with key (indices) and values (category names) for NER semantics, i.e. the
                                     entities self. To be consistent with detectors use only values >0. Conversion will
                                     be done internally.
        :param categories_bio: A dict with key (indices) and values (category names) for NER tags (i.e. BIO). To be
                               consistent with detectors use only values>0. Conversion will be done internally.
        :param categories: If you have a pre-trained model you can pass a complete dict of NER categories
        """

        if categories is None:
            assert categories_semantics is not None
            assert categories_bio is not None

        self.path_config_json = path_config_json
        self.path_weights = path_weights
        self.categories_semantics = categories_semantics
        self.categories_bio = categories_bio
        if categories:
            self.categories = copy(categories)
        else:
            self.categories = self._categories_orig_to_categories(categories_semantics, categories_bio)  # type: ignore

        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def predict(self, **encodings: Union[List[str], "torch.Tensor"]) -> List[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

        :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
        :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
        :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
        :param boxes: Torch tensor of bounding boxes of type 'xyxy'
        :param tokens: List of original tokens taken from LayoutLMTokenizer

        :return: A list of TokenClassResults
        """

        ann_ids = encodings.get("ann_ids")
        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        boxes = encodings.get("bbox")
        tokens = encodings.get("tokens")

        assert isinstance(ann_ids, list)
        if len(ann_ids) >1:
            raise ValueError("HFLayoutLmTokenClassifier accepts for inference only batch size of 1")
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(token_type_ids, torch.Tensor)
        assert isinstance(boxes, torch.Tensor)
        assert isinstance(tokens, list)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        boxes = boxes.to(self.device)

        results = predict_token_classes(
            ann_ids[0],
            input_ids,
            attention_mask,
            token_type_ids,
            boxes,
            tokens[0],
            self.model,
        )

        return self._map_category_names(results)

    @staticmethod
    def _categories_orig_to_categories(categories_semantics: List[str], categories_bio: List[str]) -> Dict[str, str]:
        categories_list = [
            x + "-" + y for x in categories_bio if x != names.NER.O for y in categories_semantics if y != names.C.O
        ] + [names.NER.O]
        return {str(k): v for k,v in enumerate(categories_list, 1)}

    def _map_category_names(self, token_results: List[TokenClassResult]) -> List[TokenClassResult]:
        for result in token_results:
            result.class_name = self.categories[str(result.class_id + 1)]
            result.semantic_name = result.class_name.split("-")[1] if "-" in result.class_name else names.C.O
            result.bio_tag = result.class_name.split("-")[0] if "-" in result.class_name else names.NER.O
            result.class_id += 1
        return token_results

    def clone(self) -> PredictorBase:
        return self.__class__(
            self.path_config_json,
            self.path_weights,
            self.categories_semantics,
            self.categories_bio,
            self.categories,
        )

    def possible_tokens(self) -> List[str]:
        return list(self.categories.values())


class HFLayoutLmSequenceClassifier(LMSequenceClassifier):

    def __init__(self, path_config_json: str,
                 path_weights: str, categories: Dict[str, str],
                 device: Optional[Literal["cpu", "cuda"]] = None):
        self.path_config = path_config_json
        self.path_weights = path_weights
        self.categories = copy(categories)
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    def predict(self, **encodings: Union[List[str], "torch.Tensor"]) -> SequenceClassResult:

        input_ids = encodings.get("input_ids")
        attention_mask = encodings.get("attention_mask")
        token_type_ids = encodings.get("token_type_ids")
        boxes = encodings.get("bbox")

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(token_type_ids, torch.Tensor)
        assert isinstance(boxes, torch.Tensor)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        boxes = boxes.to(self.device)

        result = predict_sequence_classes(
            input_ids,
            attention_mask,
            token_type_ids,
            boxes,
            self.model,
        )

        result.class_id += 1
        result.class_name = self.categories[str(result.class_id)]
        return result

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__(
            self.path_config,
            self.path_weights,
            self.categories
        )

    def possible_categories(self) -> List[str]:
        return list(self.categories.values())
