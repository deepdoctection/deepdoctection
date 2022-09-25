# -*- coding: utf-8 -*-
# File: laylmstruct.py

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
Module for mapping annotations from image to layout lm input structure. Heavily inspired by the notebooks
https://github.com/NielsRogge/Transformers-Tutorials
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, NewType, Optional, Union

import numpy as np
from cv2 import INTER_LINEAR

from ..datapoint.annotation import ContainerAnnotation
from ..datapoint.convert import box_to_point4, point4_to_box
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.develop import deprecated
from ..utils.file_utils import pytorch_available, transformers_available
from ..utils.settings import (
    BioTag,
    DatasetType,
    LayoutType,
    ObjectTypes,
    PageType,
    WordType,
    token_class_tag_to_token_class_with_tag,
)
from ..utils.transform import ResizeTransform
from .maputils import curry

if pytorch_available():
    import torch

if transformers_available():
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # pylint: disable = W0611


__all__ = [
    "image_to_layoutlm",
    "image_to_raw_layoutlm_features",
    "raw_features_to_layoutlm_features",
    "LayoutLMDataCollator",
    "image_to_layoutlm_features",
    "DataCollator",
    "LayoutLMFeatures",
]

RawLayoutLMFeatures = NewType("RawLayoutLMFeatures", JsonDict)
LayoutLMFeatures = NewType("LayoutLMFeatures", JsonDict)
InputDataClass = NewType("InputDataClass", JsonDict)

"""
https://github.com/huggingface/transformers/src/transformers/data/data_collator.py
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""

DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])  # type: ignore

_CLS_BOX = [0.0, 0.0, 1000.0, 1000.0]
_SEP_BOX = [1000.0, 1000.0, 1000.0, 1000.0]


@deprecated("Use image_to_raw_layoutlm_features and LayoutLMDataCollator instead", "2022-07-12")
@curry
def image_to_layoutlm(
    dp: Image,
    tokenizer: "PreTrainedTokenizer",
    categories_dict_name_as_key: Optional[Dict[ObjectTypes, str]] = None,
    input_width: int = 1000,
    input_height: int = 1000,
) -> LayoutLMFeatures:
    """
    Maps an image to a dict that can be consumed by a tokenizer and ultimately be passed
    to a LayoutLM language model.

    :param dp: Image
    :param tokenizer: A tokenizer aligned with the following layout model
    :param input_width: Model image input width. Will resize the image and the bounding boxes
    :param input_height: Model image input height. Will resize the image and the bounding boxes
    :param categories_dict_name_as_key: Only necessary for training. It will convert either token category names or
                                        sequence category names according to the given dict to their corresponding id.
                                        Note, that for token classification you maybe need to pass the mapping of the
                                        token classification model.
    """

    output: JsonDict = {}

    anns = dp.get_annotation_iter(category_names=LayoutType.word)
    all_tokens = []
    all_boxes = []
    all_ann_ids = []
    words: List[str] = []
    all_input_ids = []
    for ann in anns:
        char_cat = ann.get_sub_category(WordType.characters)
        if not isinstance(char_cat, ContainerAnnotation):
            raise TypeError(f"char_cat must be of type ContainerAnnotation but is of type {type(char_cat)}")
        word = char_cat.value
        if not isinstance(word, str):
            raise ValueError(f"word must be of type str but is of type {type(word)}")
        words.append(word)
        word_tokens = tokenizer.tokenize(word)
        all_input_ids.extend(tokenizer.convert_tokens_to_ids(word_tokens))

        all_tokens.extend(word_tokens)
        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
        else:
            box = ann.bounding_box
        assert box is not None, box
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)
        box = box.to_list(mode="xyxy")

        if word_tokens:
            all_boxes.extend([box] * len(word_tokens))
            all_ann_ids.extend([ann.annotation_id] * len(word_tokens))

        if (
            WordType.token_class in ann.sub_categories
            and WordType.tag in ann.sub_categories
            and categories_dict_name_as_key is not None
        ):
            semantic_label = ann.get_sub_category(WordType.token_class).category_name
            bio_tag = ann.get_sub_category(WordType.tag).category_name
            category_name: Union[ObjectTypes, BioTag]
            if bio_tag is BioTag.outside:
                category_name = BioTag.outside
            else:
                # category_name = bio_tag + "-" + semantic_label
                category_name = token_class_tag_to_token_class_with_tag(semantic_label, bio_tag)  # type: ignore
            output["label"] = int(categories_dict_name_as_key[category_name])

        if dp.summary is not None and categories_dict_name_as_key is not None:
            summary_cat_name = dp.summary.get_sub_category(PageType.document_type).category_name
            output["label"] = int(categories_dict_name_as_key[summary_cat_name])  # type: ignore

    all_boxes = [_CLS_BOX] + all_boxes + [_SEP_BOX]
    all_ann_ids = ["CLS"] + all_ann_ids + ["SEP"]
    all_tokens = ["CLS"] + all_tokens + ["SEP"]

    max_length = tokenizer.max_model_input_sizes["microsoft/layoutlm-base-uncased"]
    encoding = tokenizer(" ".join(words), return_tensors="pt", max_length=max_length)

    if len(all_ann_ids) > max_length:
        all_ann_ids = all_ann_ids[: max_length - 1] + ["SEP"]
        all_boxes = all_boxes[: max_length - 1] + [_SEP_BOX]
        all_tokens = all_tokens[: max_length - 1] + ["SEP"]

    boxes = np.asarray(all_boxes, dtype="float32")
    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)
    if dp.image is not None:
        image = resizer.apply_image(dp.image)
        output["image"] = image

    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes)
    pt_boxes = torch.clamp(torch.round(torch.tensor([boxes.tolist()])), min=0.0, max=1000.0).int()

    output["ids"] = all_ann_ids
    output["boxes"] = pt_boxes
    output["tokens"] = all_tokens
    output["input_ids"] = encoding["input_ids"]
    output["attention_mask"] = encoding["attention_mask"]
    output["token_type_ids"] = encoding["token_type_ids"]

    return LayoutLMFeatures(output)


@curry
def image_to_raw_layoutlm_features(
    dp: Image,
    categories_dict_name_as_key: Optional[Mapping[str, str]] = None,
    dataset_type: Optional[Literal["SEQUENCE_CLASSIFICATION", "TOKEN_CLASSIFICATION"]] = None,
    input_width: int = 1000,
    input_height: int = 1000,
) -> Optional[RawLayoutLMFeatures]:
    """
    Mapping a datapoint into an intermediate format for layoutlm. Features will be provided into a dict and this mapping
    can be used for sequence or token classification as well as for inference. To generate input features for the model
    please :func:`use raw_features_to_layoutlm_features`.


    :param dp: Image
    :param categories_dict_name_as_key: categories with names and ids. In comparison with arguments of the same name in
                                        other functions the categories must be the categories of the model.
                                        For SEQUENCE_CLASSIFICATION type datasets this will be the various document
                                        classes and can be created by e.g. using
                                        sequence_dataset.dataflow.categories.get_categories(as_dict=True,
                                        name_as_key=True).
                                        For TOKEN_CLASSIFICATION you will have to generate a dict of categories of type
                                        "B-ANSWER","I-ANSWER","B-QUESTION","I-QUESTION","O" depending on what token
                                        classes the model has been trained, resp. should be trained.
                                        When using a TOKEN_CLASSIFICATION dataset note that all possible token classes
                                        are generated by concatenating SEMANTIC_ENTITY with NER_TAG, where the OTHER
                                        class is a stand-alone class with no NER_TAG.
    :param dataset_type: Either SEQUENCE_CLASSIFICATION or TOKEN_CLASSIFICATION. When using a built-in dataset use
    :param input_width: target width of the image. Under the hood, it will transform all box coordinates accordingly.
    :param input_height: target width of the image. Under the hood, it will transform all box coordinates accordingly.
    :return: dictionary with the following arguments:
            'image_id', 'width', 'height', 'ann_ids', 'words', 'bbox' and 'dataset_type'.
    """

    raw_features: RawLayoutLMFeatures = RawLayoutLMFeatures({})
    all_ann_ids = []
    all_words = []
    all_boxes = []
    all_labels: List[int] = []

    anns = dp.get_annotation_iter(category_names=LayoutType.word)

    for ann in anns:
        all_ann_ids.append(ann.annotation_id)
        char_cat = ann.get_sub_category(WordType.characters)
        if not isinstance(char_cat, ContainerAnnotation):
            raise TypeError(f"char_cat must be of type ContainerAnnotation but is of type {type(char_cat)}")
        word = char_cat.value
        if not isinstance(word, str):
            raise ValueError(f"word must be of type str but is of type {type(word)}")
        all_words.append(word)

        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
        else:
            box = ann.bounding_box
        assert box is not None, box
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)
        all_boxes.append(box.to_list(mode="xyxy"))

        if (
            WordType.token_tag in ann.sub_categories
            and categories_dict_name_as_key is not None
            and dataset_type == DatasetType.token_classification
        ):
            all_labels.append(int(ann.get_sub_category(WordType.token_tag).category_id) - 1)

    if (
        dp.summary is not None
        and categories_dict_name_as_key is not None
        and dataset_type == DatasetType.sequence_classification
    ):
        category_name = dp.summary.get_sub_category(PageType.document_type).category_name
        all_labels.append(int(categories_dict_name_as_key[category_name]) - 1)

    boxes = np.asarray(all_boxes, dtype="float32")
    if boxes.ndim == 1:
        return None

    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)

    if dp.image is not None:
        image = resizer.apply_image(dp.image)
        raw_features["image"] = image  # pylint: disable=E1137  #3162

    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes)

    # input box coordinates must be of type long. We floor the ul and ceil the lr coords
    boxes = np.concatenate((np.floor(boxes)[:, :2], np.ceil(boxes)[:, 2:]), axis=1).tolist()

    # pylint: disable=E1137  #3162
    raw_features["image_id"] = dp.image_id
    raw_features["width"] = input_width
    raw_features["height"] = input_height
    raw_features["ann_ids"] = all_ann_ids
    raw_features["words"] = all_words
    raw_features["bbox"] = boxes
    raw_features["dataset_type"] = dataset_type

    if categories_dict_name_as_key:
        raw_features["labels"] = all_labels
    # pylint: enable=E1137
    return raw_features


def features_to_pt_tensors(features: LayoutLMFeatures) -> LayoutLMFeatures:
    """
    Converting list of floats to pytorch tensors
    :param features: LayoutLMFeatures
    :return: LayoutLMFeatures
    """
    features["bbox"] = torch.tensor(features["bbox"], dtype=torch.long)
    if "labels" in features:
        features["labels"] = torch.tensor(features["labels"], dtype=torch.long)
    if "images" in features:
        features["images"] = [
            torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) for image in features["images"]
        ]
    return features


def raw_features_to_layoutlm_features(
    raw_features: Union[RawLayoutLMFeatures, List[RawLayoutLMFeatures]],
    tokenizer: "PreTrainedTokenizerFast",
    padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
    truncation: bool = True,
    return_overflowing_tokens: bool = False,
    return_tensors: Optional[Literal["pt"]] = None,
    remove_columns_for_training: bool = False,
) -> LayoutLMFeatures:
    """
    Mapping raw features to tokenized input sequences for LayoutLM models.

    :param raw_features: A dictionary with the following arguments: `image_id, width, height, ann_ids, words,
                         boxes, dataset_type`.
    :param tokenizer: A fast tokenizer for the model. Note, that the conventional python based tokenizer provided by the
                      Transformer library do not return essential word_id/token_id mappings making the feature
                      generation a lot more difficult. We therefore do not allow these tokenizer.
    :param padding: A padding strategy to be passed to the tokenizer. Must bei either `max_length, longest` or
                    `do_not_pad`.
    :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                       maximum acceptable input length for the model if that argument is not provided. This will
                       truncate token by token, removing a token from the longest sequence in the pair if a pair of
                       sequences (or a batch of pairs) is provided.
                       If `False` then no truncation (i.e., can output batch with sequence lengths greater than the
                       model maximum admissible input size).
    :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens can
                                  be returned as an additional batch element. Not that in this case, the number of input
                                  batch samples will be smaller than the output batch samples.
    :param return_tensors: If `pt` will return torch Tensors. If no argument is provided that the batches will be lists
                           of lists.
    :param remove_columns_for_training: Will remove all superfluous columns that are not required for training.
    :return: dictionary with the following arguments:  `image_ids, width, height, ann_ids, input_ids,
             token_type_ids, attention_mask, bbox, labels`.
    """

    if isinstance(raw_features, dict):
        raw_features = [raw_features]

    _has_token_labels = (
        raw_features[0]["dataset_type"] == DatasetType.token_classification
        and raw_features[0].get("labels") is not None
    )
    _has_sequence_labels = (
        raw_features[0]["dataset_type"] == DatasetType.sequence_classification
        and raw_features[0].get("labels") is not None
    )
    _has_labels = bool(_has_token_labels or _has_sequence_labels)

    tokenized_inputs = tokenizer(
        [dp["words"] for dp in raw_features],
        padding=padding,
        truncation=truncation,
        return_overflowing_tokens=return_overflowing_tokens,
        is_split_into_words=True,
        return_tensors=return_tensors,
    )

    image_ids = []
    images = []
    widths = []
    heights = []

    token_boxes = []
    token_labels = []
    sequence_labels = []
    token_ann_ids = []
    tokens = []
    for batch_index in range(len(tokenized_inputs["input_ids"])):
        batch_index_orig = batch_index
        if return_overflowing_tokens:
            # we might get more batches when we allow to get returned overflowing tokens
            batch_index_orig = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

        if "image" in raw_features[batch_index_orig]:
            images.append(raw_features[batch_index_orig]["image"])
        image_ids.append(raw_features[batch_index_orig]["image_id"])
        widths.append(raw_features[batch_index_orig]["width"])
        heights.append(raw_features[batch_index_orig]["height"])

        ann_ids = raw_features[batch_index_orig]["ann_ids"]
        boxes = raw_features[batch_index_orig]["bbox"]
        if _has_token_labels:
            labels = raw_features[batch_index_orig]["labels"]
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        token_batch = tokenized_inputs.tokens(batch_index=batch_index)

        token_batch_ann_ids = []
        token_batch_boxes = []
        token_batch_labels = []
        for idx, word_id in enumerate(word_ids):
            # Special tokens have a word id that is None. We make a lookup for the specific token and append a dummy
            # bounding box accordingly
            if word_id is None:
                if token_batch[idx] == "[CLS]":
                    token_batch_boxes.append(_CLS_BOX)
                    token_batch_ann_ids.append("[CLS]")
                elif token_batch[idx] in ("[SEP]", "[PAD]"):
                    token_batch_boxes.append(_SEP_BOX)
                    if token_batch[idx] == "[SEP]":
                        token_batch_ann_ids.append("[SEP]")
                    else:
                        token_batch_ann_ids.append("[PAD]")
                else:
                    raise ValueError(f"Special token {token_batch[idx]} not allowed")
                if _has_token_labels:
                    token_batch_labels.append(-100)
            else:
                token_batch_boxes.append(boxes[word_id])
                token_batch_ann_ids.append(ann_ids[word_id])
                if _has_token_labels:
                    token_batch_labels.append(labels[word_id])

        token_labels.append(token_batch_labels)
        token_boxes.append(token_batch_boxes)
        token_ann_ids.append(token_batch_ann_ids)
        tokens.append(token_batch)
        if _has_sequence_labels:
            sequence_labels.append(raw_features[batch_index_orig]["labels"][0])

    input_dict = {
        "image_ids": image_ids,
        "width": widths,
        "height": heights,
        "ann_ids": token_ann_ids,
        "input_ids": tokenized_inputs["input_ids"],
        "token_type_ids": tokenized_inputs["token_type_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "bbox": token_boxes,
        "tokens": tokens,
    }

    # will only add the image to features if it has been passed as raw feature
    if images:
        input_dict["images"] = images

    if _has_labels:
        input_dict["labels"] = token_labels if _has_token_labels else sequence_labels

    if remove_columns_for_training:
        input_dict.pop("image_ids")
        input_dict.pop("width")
        input_dict.pop("height")
        input_dict.pop("ann_ids")
        input_dict.pop("tokens")

    if return_tensors == "pt":
        return features_to_pt_tensors(LayoutLMFeatures(input_dict))
    return LayoutLMFeatures(input_dict)


@dataclass
class LayoutLMDataCollator:
    """
    Data collator that will dynamically tokenize, pad and truncate the inputs received.

    :param tokenizer: A fast tokenizer for the model. Note, that the conventional python based tokenizer provided by the
                      Transformer library do not return essential word_id/token_id mappings making the feature
                      generation a lot more difficult. We therefore do not allow these tokenizer.
    :param padding: A padding strategy to be passed to the tokenizer. Must bei either `max_length, longest` or
                    `do_not_pad`.
    :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                       maximum acceptable input length for the model if that argument is not provided. This will
                       truncate token by token, removing a token from the longest sequence in the pair if a pair of
                       sequences (or a batch of pairs) is provided.
                       If `False` then no truncation (i.e., can output batch with sequence lengths greater than the
                       model maximum admissible input size).
    :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens can
                                  be returned as an additional batch element. Not that in this case, the number of input
                                  batch samples will be smaller than the output batch samples.
    :param return_tensors: If `pt` will return torch Tensors. If no argument is provided that the batches will be lists
                           of lists.

    :return: dictionary with the following arguments:  `image_ids, width, height, ann_ids, input_ids,
             token_type_ids, attention_masks, boxes, labels`.
    """

    tokenizer: "PreTrainedTokenizerFast"
    padding: Literal["max_length", "do_not_pad", "longest"] = field(default="max_length")
    truncation: bool = field(default=True)
    return_overflowing_tokens: bool = field(default=False)
    return_tensors: Optional[Literal["pt"]] = field(default=None)

    def __post_init__(self) -> None:
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast), "Tokenizer must be a fast tokenizer"
        if self.return_tensors:
            assert self.padding not in ("do_not_pad",), self.padding
            assert self.truncation, self.truncation
        if self.return_overflowing_tokens:
            assert self.truncation, self.truncation

    def __call__(self, raw_features: Union[RawLayoutLMFeatures, List[RawLayoutLMFeatures]]) -> LayoutLMFeatures:
        """
        Calling the DataCollator to form model inputs for training and inference. Takes a single raw
        :param raw_features: A dictionary with the following arguments: `image_id, width, height, ann_ids, words,
                             boxes, dataset_type`.
        :return: LayoutLMFeatures
        """
        return raw_features_to_layoutlm_features(
            raw_features,
            self.tokenizer,
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
            self.return_tensors,
            True,
        )


@curry
def image_to_layoutlm_features(
    dp: Image,
    tokenizer: "PreTrainedTokenizerFast",
    return_tensors: Optional[Literal["pt"]] = "pt",
    input_width: int = 1000,
    input_height: int = 1000,
) -> Optional[LayoutLMFeatures]:
    """
    Mapping function to generate layoutlm features from `Image` to be used for inference in a pipeline component.
    :class:`LanguageModelPipelineComponent` has a positional argument `mapping_to_lm_input_func` that must be chosen
    with respect to the language model chosen. This mapper is devoted to generating features for LayoutLM.

    .. code-block:: python

            tokenizer = LayoutLMTokenizer.from_pretrained("mrm8488/layoutlm-finetuned-funsd")
            layoutlm = HFLayoutLmTokenClassifier("path/to/config.json","path/to/model.bin",
                                                  categories_explicit= ['B-ANSWER', 'B-QUESTION', 'O'])

            layoutlm_service = LMTokenClassifierService(tokenizer,layoutlm, image_to_layoutlm_features)


    :param dp: Image datapoint
    :param tokenizer: Tokenizer compatible with the language model
    :param return_tensors: Output tensor features. Either 'pt' for PyTorch models or None, if features should be
                           returned in list objects.
    :param input_width: Standard input size for image coordinates. All LayoutLM models require input features to be
                        normalized to an image width equal to 1000.
    :param input_height: Standard input size for image coordinates. All LayoutLM models require input features to be
                         normalized to an image height equal to 1000.
    :return: A dict of layoutlm features
    """
    raw_features = image_to_raw_layoutlm_features(None, None, input_width, input_height)(dp)
    if raw_features is None:
        return None
    features = raw_features_to_layoutlm_features(raw_features, tokenizer, return_tensors=return_tensors)
    return features
