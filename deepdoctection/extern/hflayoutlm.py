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

from abc import ABC
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..utils.detection_types import JsonDict, Requirement
from ..utils.file_utils import (
    get_pytorch_requirement,
    get_transformers_requirement,
    pytorch_available,
    transformers_available,
)
from ..utils.settings import (
    BioTag,
    ObjectTypes,
    TokenClasses,
    TypeOrStr,
    get_type,
    token_class_tag_to_token_class_with_tag,
    token_class_with_tag_to_token_class_and_tag,
)
from .base import LMSequenceClassifier, LMTokenClassifier, SequenceClassResult, TokenClassResult
from .pt.ptutils import set_torch_auto_device

if pytorch_available():
    import torch
    import torch.nn.functional as F
    from torch import Tensor  # pylint: disable=W0611

if transformers_available():
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # type: ignore
    from transformers import (
        LayoutLMForSequenceClassification,
        LayoutLMForTokenClassification,
        LayoutLMv2Config,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv3Config,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3ForTokenClassification,
        PretrainedConfig,
    )


def predict_token_classes(
    uuids: List[List[str]],
    input_ids: "Tensor",
    attention_mask: "Tensor",
    token_type_ids: "Tensor",
    boxes: "Tensor",
    tokens: List[List[str]],
    model: "LayoutLMForTokenClassification",
    images: Optional["Tensor"] = None,
) -> List[TokenClassResult]:
    """
    :param uuids: A list of uuids that correspond to a word that induces the resulting token
    :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
    :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
    :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
    :param boxes: Torch tensor of bounding boxes of type 'xyxy'
    :param tokens: List of original tokens taken from LayoutLMTokenizer
    :param model: layoutlm model for token classification
    :param images: A list of torch image tensors or None
    :return: A list of TokenClassResults
    """
    if images is None:
        outputs = model(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids)
    else:
        outputs = model(
            input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, token_type_ids=token_type_ids, image=images
        )
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


def predict_sequence_classes(
    input_ids: "Tensor",
    attention_mask: "Tensor",
    token_type_ids: "Tensor",
    boxes: "Tensor",
    model: Union[
        "LayoutLMForSequenceClassification",
        "LayoutLMv2ForSequenceClassification",
        "LayoutLMv3ForSequenceClassification",
    ],
    images: Optional["Tensor"] = None,
) -> SequenceClassResult:
    """
    :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
    :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
    :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
    :param boxes: Torch tensor of bounding boxes of type 'xyxy'
    :param model: layoutlm model for token classification
    :param images: A list of torch image tensors or None
    :return: SequenceClassResult
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
    Abstract base class for wrapping LayoutLM models for token classification into the deepdoctection framework.
    """

    model: Union["LayoutLMForTokenClassification", "LayoutLMv2ForTokenClassification"]

    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[str, TypeOrStr]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None,
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
        """

        self.name = "_".join(Path(path_weights).parts[-3:])
        if categories is None:
            if categories_semantics is None:
                raise ValueError("If categories is None then categories_semantics cannot be None")
            if categories_bio is None:
                raise ValueError("If categories is None then categories_bio cannot be None")

        self.path_config = path_config_json
        self.path_weights = path_weights
        self.categories_semantics = (
            [get_type(cat_sem) for cat_sem in categories_semantics] if categories_semantics is not None else []
        )
        self.categories_bio = [get_type(cat_bio) for cat_bio in categories_bio] if categories_bio is not None else []
        if categories:
            self.categories = copy(categories)  # type: ignore
        else:
            self.categories = self._categories_orig_to_categories(
                self.categories_semantics, self.categories_bio  # type: ignore
            )
        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    @staticmethod
    def _categories_orig_to_categories(
        categories_semantics: List[TokenClasses], categories_bio: List[BioTag]
    ) -> Dict[str, ObjectTypes]:
        categories_list = sorted(
            {
                token_class_tag_to_token_class_with_tag(token, tag)
                for token in categories_semantics
                for tag in categories_bio
            }
        )
        return {str(k): v for k, v in enumerate(categories_list, 1)}

    def _map_category_names(self, token_results: List[TokenClassResult]) -> List[TokenClassResult]:
        for result in token_results:
            result.class_name = self.categories[str(result.class_id + 1)]
            output = token_class_with_tag_to_token_class_and_tag(result.class_name)
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
    ) -> Tuple[
        List[List[str]], List[str], "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", List[List[str]]
    ]:
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

    def clone(self) -> "HFLayoutLmTokenClassifierBase":
        return self.__class__(
            self.path_config,
            self.path_weights,
            self.categories_semantics,
            self.categories_bio,
            self.categories,
            self.device,
        )


class HFLayoutLmTokenClassifier(HFLayoutLmTokenClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMForTokenClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/model_doc/layoutlm> for documentation of the model itself.
    Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    **Example**

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
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[str, TypeOrStr]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None,
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
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories_semantics, categories_bio, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[TokenClassResult]:
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

        results = predict_token_classes(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, None
        )

        return self._map_category_names(results)


class HFLayoutLmv2TokenClassifier(HFLayoutLmTokenClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv2ForTokenClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv2>  for documentation of the model
    itself. Note that this model is equipped with a head that is only useful when classifying tokens. For sequence
    classification and other things please use another model of the family.

    Note, that you must use `LayoutLMTokenizerFast` as tokenizer. `LayoutLMv2TokenizerFast` will not be accepted.

    **Example**

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
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[str, TypeOrStr]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None,
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
        """
        config = LayoutLMv2Config.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories_semantics, categories_bio, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[TokenClassResult]:
        """
        Launch inference on LayoutLm for token classification. Pass the following arguments

        `input_ids:` Token converted to ids to be taken from `LayoutLMTokenizer`

        `attention_mask:` The associated attention masks from padded sequences taken from `LayoutLMTokenizer`

        `token_type_ids:` Torch tensor of token type ids taken from `LayoutLMTokenizer`

        `boxes:` Torch tensor of bounding boxes of type `xyxy`

        `tokens:` List of original tokens taken from `LayoutLMTokenizer`

        :return: A list of TokenClassResults
        """

        ann_ids, _, input_ids, attention_mask, token_type_ids, boxes, tokens = self._validate_encodings(**encodings)

        images = encodings.get("image")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")
        results = predict_token_classes(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, images
        )

        return self._map_category_names(results)

    @staticmethod
    def default_kwargs_for_input_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {"image_width": 224, "image_height": 224}


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
            layoutlm_service = LMTokenClassifierService(tokenizer, layoutlm, image_to_layoutlm_features)

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
        categories_semantics: Optional[Sequence[TypeOrStr]] = None,
        categories_bio: Optional[Sequence[TypeOrStr]] = None,
        categories: Optional[Mapping[str, TypeOrStr]] = None,
        device: Optional[Literal["cpu", "cuda"]] = None,
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
        """
        config = LayoutLMv3Config.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories_semantics, categories_bio, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[TokenClassResult]:
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
        results = predict_token_classes(
            ann_ids, input_ids, attention_mask, token_type_ids, boxes, tokens, self.model, images
        )

        return self._map_category_names(results)

    @staticmethod
    def default_kwargs_for_input_mapping() -> JsonDict:
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


class HFLayoutLmSequenceClassifierBase(LMSequenceClassifier, ABC):
    """
    Abstract base class for wrapping LayoutLM models  for sequence classification into the deepdoctection framework.
    """

    model: Union["LayoutLMForSequenceClassification", "LayoutLMv2ForSequenceClassification"]

    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):
        self.name = "_".join(Path(path_weights).parts[-3:])
        self.path_config = path_config_json
        self.path_weights = path_weights
        self.categories = copy(categories)  # type: ignore

        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> SequenceClassResult:

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

    def clone(self) -> "HFLayoutLmSequenceClassifierBase":
        return self.__class__(self.path_config, self.path_weights, self.categories, self.device)

    def _validate_encodings(
        self, **encodings: Union[List[List[str]], "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:

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


class HFLayoutLmSequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/model_doc/layoutlm> for documentation of the model itself.
    Note that this model is equipped with a head that is only useful for classifying the input sequence. For token
    classification and other things please use another model of the family.

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
            layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)

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
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):

        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)

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


class HFLayoutLmv2SequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv2ForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv2> for documentation of the model
    itself. Note that this model is equipped with a head that is only useful for classifying the input sequence. For
    token classification and other things please use another model of the family.

    **Example**

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
            layoutlm = HFLayoutLmv2SequenceClassifier("path/to/config.json","path/to/model.bin",
                                                  categories=["handwritten", "presentation", "resume"])

            # token classification service
            layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)

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
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):

        config = LayoutLMv2Config.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMv2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)
        images = encodings.get("image")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")

        result = predict_sequence_classes(input_ids, attention_mask, token_type_ids, boxes, self.model, images)

        result.class_id += 1
        result.class_name = self.categories[str(result.class_id)]
        return result

    @staticmethod
    def default_kwargs_for_input_mapping() -> JsonDict:
        """
        Add some default arguments that might be necessary when preparing a sample. Overwrite this method
        for some custom setting.
        """
        return {"image_width": 224, "image_height": 224}


class HFLayoutLmv3SequenceClassifier(HFLayoutLmSequenceClassifierBase):
    """
    A wrapper class for `transformers.LayoutLMv3ForSequenceClassification` to use within a pipeline component.
    Check <https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/layoutlmv3> for documentation of the model
    itself. Note that this model is equipped with a head that is only useful for classifying the input sequence. For
    token classification and other things please use another model of the family.

    **Example**

            # setting up compulsory ocr service
            tesseract_config_path = ModelCatalog.get_full_path_configs("/dd/conf_tesseract.yaml")
            tess = TesseractOcrDetector(tesseract_config_path)
            ocr_service = TextExtractionService(tess)

            # hf tokenizer and token classifier
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            layoutlm = HFLayoutLmv3SequenceClassifier("path/to/config.json","path/to/model.bin",
                                                  categories=["handwritten", "presentation", "resume"])

            # token classification service
            layoutlm_service = LMSequenceClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)

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
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
    ):

        config = LayoutLMv3Config.from_pretrained(pretrained_model_name_or_path=path_config_json)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
        super().__init__(path_config_json, path_weights, categories, device)

    def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> SequenceClassResult:
        input_ids, attention_mask, token_type_ids, boxes = self._validate_encodings(**encodings)
        images = encodings.get("pixel_values")
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            raise ValueError(f"images must be list but is {type(images)}")

        result = predict_sequence_classes(input_ids, attention_mask, token_type_ids, boxes, self.model, images)

        result.class_id += 1
        result.class_name = self.categories[str(result.class_id)]
        return result

    @staticmethod
    def default_kwargs_for_input_mapping() -> JsonDict:
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
