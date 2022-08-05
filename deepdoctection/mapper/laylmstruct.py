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

from typing import List, Optional, Dict, Literal, Union, Mapping, NewType
from dataclasses import dataclass, field

import numpy as np
from cv2 import INTER_LINEAR
from dataflow.dataflow.imgaug.transform import ResizeTransform

from ..datapoint.annotation import ContainerAnnotation
from ..datapoint.convert import box_to_point4, point4_to_box
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.file_utils import pytorch_available, transformers_available
from ..utils.settings import names
from ..utils.develop import deprecated
from .maputils import curry

if pytorch_available():
    from torch import clamp, round, tensor, long, float  # pylint: disable = W0622

if transformers_available():
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # pylint: disable = W0611


__all__ = ["image_to_layoutlm", "image_to_raw_layoutlm_features", "raw_features_to_layoutlm_features",
           "LayoutLMDataCollator", "image_to_layoutlm_features"]


RawLayoutLMFeatures = NewType("RawLayoutLMFeatures", JsonDict)
LayoutLMFeatures = NewType("LayoutLMFeatures", JsonDict)

_CLS_BOX = [0.0, 0.0, 0.0, 0.0]
_SEP_BOX = [1000.0, 1000.0, 1000.0, 1000.0]


@deprecated("Use image_to_raw_layoutlm_features and LayoutLMDataCollator instead", "2022-07-12")
@curry
def image_to_layoutlm(
    dp: Image, tokenizer: "PreTrainedTokenizer", categories_dict_name_as_key: Optional[Dict[str,str]]=None,
        input_width: int = 1000, input_height: int = 1000,
        ) -> JsonDict:
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

    anns = dp.get_annotation_iter(category_names=names.C.WORD)
    all_tokens = []
    all_boxes = []
    all_ann_ids = []
    words: List[str] = []
    all_input_ids= []
    for ann in anns:
        char_cat = ann.get_sub_category(names.C.CHARS)
        assert isinstance(char_cat, ContainerAnnotation)
        word = char_cat.value
        assert isinstance(word, str)
        words.append(word)
        word_tokens = tokenizer.tokenize(word)
        all_input_ids.extend(tokenizer.convert_tokens_to_ids(word_tokens))

        all_tokens.extend(word_tokens)
        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
        else:
            box = ann.bounding_box
        assert box is not None
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)
        box = box.to_list(mode="xyxy")

        if word_tokens:
            all_boxes.extend([box] * len(word_tokens))
            all_ann_ids.extend([ann.annotation_id] * len(word_tokens))

        if names.C.SE in ann.sub_categories and names.NER.TAG in ann.sub_categories and categories_dict_name_as_key \
                is not None:
            semantic_label = ann.get_sub_category(names.C.SE).category_name
            bio_tag = ann.get_sub_category(names.NER.TAG).category_name
            if bio_tag == "O":
                category_name = "O"
            else:
                category_name = bio_tag + "-" + semantic_label
            output["label"] = int(categories_dict_name_as_key[category_name])

        if dp.summary is not None and categories_dict_name_as_key is not None:
            category_name = dp.summary.get_sub_category(names.C.DOC).category_name
            output["label"] = int(categories_dict_name_as_key[category_name])

    all_boxes = [_CLS_BOX] + all_boxes + [_SEP_BOX]
    all_ann_ids = ["CLS"] + all_ann_ids + ["SEP"]
    all_tokens = ["CLS"] + all_tokens + ["SEP"]

    max_length = tokenizer.max_model_input_sizes["microsoft/layoutlm-base-uncased"]
    encoding = tokenizer(" ".join(words), return_tensors="pt", max_length=max_length)

    if len(all_ann_ids)>max_length:
        all_ann_ids = all_ann_ids[:max_length-1] + ["SEP"]
        all_boxes = all_boxes[:max_length-1] + [_SEP_BOX]
        all_tokens = all_tokens[:max_length-1] + ["SEP"]

    boxes = np.asarray(all_boxes, dtype="float32")
    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)
    if dp.image is not None:
        image = resizer.apply_image(dp.image)
        output["image"] = image

    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes)
    pt_boxes = clamp(round(tensor([boxes.tolist()])), min=0.0, max=1000.0).int()

    output["ids"] = all_ann_ids
    output["boxes"] = pt_boxes
    output["tokens"] = all_tokens
    output["input_ids"] = encoding["input_ids"]
    output["attention_mask"] = encoding["attention_mask"]
    output["token_type_ids"] = encoding["token_type_ids"]

    return output


@curry
def image_to_raw_layoutlm_features(dp: Image,
                                   categories_dict_name_as_key: Optional[Mapping[str,str]]=None,
                                   dataset_type: Optional[Literal["SEQUENCE_CLASSIFICATION",
                                                                  "TOKEN_CLASSIFICATION"]] = None,
                                   input_width: int = 1000,
                                   input_height: int = 1000) -> Optional[RawLayoutLMFeatures]:
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
    all_labels = []

    anns = dp.get_annotation_iter(category_names=names.C.WORD)

    for ann in anns:
        all_ann_ids.append(ann.annotation_id)
        char_cat = ann.get_sub_category(names.C.CHARS)
        assert isinstance(char_cat, ContainerAnnotation)
        word = char_cat.value
        assert isinstance(word, str)
        all_words.append(word)

        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
        else:
            box = ann.bounding_box
        assert box is not None
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)
        all_boxes.append(box.to_list(mode="xyxy"))

        if names.C.SE in ann.sub_categories and names.NER.TAG in ann.sub_categories and categories_dict_name_as_key \
                is not None and dataset_type == names.DS.TYPE.TOK:
            semantic_label = ann.get_sub_category(names.C.SE).category_name
            bio_tag = ann.get_sub_category(names.NER.TAG).category_name
            if bio_tag == "O":
                category_name = "O"
            else:
                category_name = bio_tag + "-" + semantic_label
            all_labels.append(int(categories_dict_name_as_key[category_name]))

    if dp.summary is not None and categories_dict_name_as_key is not None and dataset_type == names.DS.TYPE.SEQ:
        category_name = dp.summary.get_sub_category(names.C.DOC).category_name
        all_labels.append(int(categories_dict_name_as_key[category_name])-1)

    boxes = np.asarray(all_boxes, dtype="float32")
    if boxes.ndim == 1:
        return None

    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)

    if dp.image is not None:
        image = resizer.apply_image(dp.image)
        raw_features["image"] = image

    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes)

    # input box coordinates must be of type long. We floor the ul and ceil the lr coords
    boxes = np.concatenate((np.floor(boxes)[:, :2], np.ceil(boxes)[:, 2:]), axis=1).tolist()

    raw_features["image_id"] = dp.image_id
    raw_features["width"] = input_width
    raw_features["height"] = input_height
    raw_features["ann_ids"] = all_ann_ids
    raw_features["words"] = all_words
    raw_features["bbox"] = boxes
    raw_features["dataset_type"] = dataset_type

    if categories_dict_name_as_key:
        raw_features["labels"] = all_labels

    return raw_features


def raw_features_to_layoutlm_features(raw_features: Union[RawLayoutLMFeatures,List[RawLayoutLMFeatures]],
                                      tokenizer: "PreTrainedTokenizerFast",
                                      padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
                                      truncation: bool = True,
                                      return_overflowing_tokens: bool = False,
                                      return_tensors: Optional[Literal["pt"]] = None,
                                      remove_columns_for_training: bool = False
                                      ) -> LayoutLMFeatures:
    """
    Mapping raw features to tokenized input sequences for LayoutLM models.

    :param raw_features: A dictionary with the following arguments: "image_id", "width", "height", "ann_ids", "words",
                         "boxes", "dataset_type".
    :param tokenizer: A fast tokenizer for the model. Note, that the conventional python based tokenizer provided by the
                      Transformer library do not return essential word_id/token_id mappings making the feature
                      generation a lot more difficult. We therefore do not allow these tokenizer.
    :param padding: A padding strategy to be passed to the tokenizer. Must bei either "max_length", "longest" or
                    "do_not_pad".
    :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                       maximum acceptable input length for the model if that argument is not provided. This will
                       truncate token by token, removing a token from the longest sequence in the pair if a pair of
                       sequences (or a batch of pairs) is provided.
                       If "False" then no truncation (i.e., can output batch with sequence lengths greater than the
                       model maximum admissible input size).
    :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens can be
                                  returned as an additional batch element. Not that in this case, the number of input
                                  batch samples will be smaller than the output batch samples.
    :param return_tensors: If "pt" will return torch Tensors. If no argument is provided that the batches will be lists
                           of lists.
    :param remove_columns_for_training: Will remove all superfluous columns that are not required for training.
    :return: dictionary with the following arguments:  "image_ids", "width", "height", "ann_ids", "input_ids",
             "token_type_ids", "attention_mask", "bbox", "labels".
    """

    if isinstance(raw_features, dict):
        raw_features = [raw_features]

    _has_token_labels = raw_features[0]["dataset_type"] == names.DS.TYPE.TOK and raw_features[0].get("labels") is not None
    _has_sequence_labels = raw_features[0]["dataset_type"] == names.DS.TYPE.SEQ and raw_features[0].get("labels") is not None

    tokenized_inputs = tokenizer(
        [dp["words"] for dp in raw_features],
        padding=padding,
        truncation=truncation,
        return_overflowing_tokens=return_overflowing_tokens,
        is_split_into_words=True,
        return_tensors = return_tensors
    )

    image_ids = []
    widths = []
    heights = []
    ann_ids = []

    token_boxes = []
    token_labels = []
    sequence_labels = []
    for batch_index in range(len(tokenized_inputs["input_ids"])):
        batch_index_orig = batch_index
        if return_overflowing_tokens:
            # we might get more batches when we allow to get returned overflowing tokens
            batch_index_orig = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

        image_ids.append(raw_features[batch_index_orig]["image_id"])
        widths.append(raw_features[batch_index_orig]["width"])
        heights.append(raw_features[batch_index_orig]["height"])
        ann_ids.append(raw_features[batch_index_orig]["ann_ids"])

        boxes = raw_features[batch_index_orig]["bbox"]
        if _has_token_labels:
            labels = raw_features[batch_index_orig]["labels"]
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        tokens = tokenized_inputs.tokens(batch_index= batch_index)

        token_batch_boxes = []
        token_batch_labels = []
        for idx, word_id in enumerate(word_ids):
            # Special tokens have a word id that is None. We make a lookup for the specific token and append a dummy
            # bounding box accordingly
            if word_id is None:
                if tokens[idx] == "[CLS]":
                    token_batch_boxes.append(_CLS_BOX)
                elif tokens[idx] in ("[SEP]","[PAD]"):
                    token_batch_boxes.append(_SEP_BOX)
                else:
                    raise ValueError(f"Special token {tokens[idx]} not allowed")
                if _has_token_labels:
                    token_batch_labels.append(-100)
            else:
                token_batch_boxes.append(boxes[word_id])
                if _has_token_labels:
                    token_batch_labels.append(labels[word_id])

        token_labels.append(token_batch_labels)
        token_boxes.append(token_batch_boxes)

        if _has_sequence_labels:
            sequence_labels.append(raw_features[batch_index_orig]["labels"][0])

    if return_tensors == "pt":
        token_boxes = tensor(token_boxes,dtype=long)
        if _has_token_labels:
            token_labels = tensor(token_labels,dtype=long)
        if _has_sequence_labels:
            sequence_labels = tensor(sequence_labels,dtype=long)

    if remove_columns_for_training:
        return LayoutLMFeatures({ "input_ids": tokenized_inputs["input_ids"],
               "token_type_ids": tokenized_inputs["token_type_ids"],
               "attention_mask": tokenized_inputs["attention_mask"],
               "bbox": token_boxes,
               "labels": token_labels if _has_token_labels else sequence_labels})

    return LayoutLMFeatures({"image_ids": image_ids,
               "width": widths,
               "height": heights,
               "ann_ids":ann_ids,
               "input_ids": tokenized_inputs["input_ids"],
               "token_type_ids": tokenized_inputs["token_type_ids"],
               "attention_mask": tokenized_inputs["attention_mask"],
               "bbox": token_boxes,
               "labels": token_labels if _has_token_labels else sequence_labels})


@dataclass
class LayoutLMDataCollator:
    """
    Data collator that will dynamically tokenize, pad and truncate the inputs received.

    :param tokenizer: A fast tokenizer for the model. Note, that the conventional python based tokenizer provided by the
                      Transformer library do not return essential word_id/token_id mappings making the feature
                      generation a lot more difficult. We therefore do not allow these tokenizer.
    :param padding: A padding strategy to be passed to the tokenizer. Must bei either "max_length", "longest" or
                    "do_not_pad".
    :param truncation: If "True" will truncate to a maximum length specified with the argument max_length or to the
                       maximum acceptable input length for the model if that argument is not provided. This will
                       truncate token by token, removing a token from the longest sequence in the pair if a pair of
                       sequences (or a batch of pairs) is provided.
                       If "False" then no truncation (i.e., can output batch with sequence lengths greater than the
                       model maximum admissible input size).
    :param return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows the overflowing tokens can be
                                  returned as an additional batch element. Not that in this case, the number of input
                                  batch samples will be smaller than the output batch samples.
    :param return_tensors: If "pt" will return torch Tensors. If no argument is provided that the batches will be lists
                           of lists.

    :return: dictionary with the following arguments:  "image_ids", "width", "height", "ann_ids", "input_ids",
             "token_type_ids", "attention_masks", "boxes", "labels".
    """

    tokenizer: PreTrainedTokenizerFast
    padding: Literal["max_length", "do_not_pad", "longest"] = field(default="max_length")
    truncation: bool = field(default=True)
    return_overflowing_tokens: bool = field(default=False)
    return_tensors: Optional[Literal["pt"]] = field(default=None)

    def __post_init__(self) -> None:
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast), "Tokenizer must be a fast tokenizer"
        if self.return_tensors:
            assert self.padding not in ("do_not_pad",)
            assert self.truncation
        if self.return_overflowing_tokens:
            assert self.truncation

    def __call__(self, raw_features: Union[RawLayoutLMFeatures,List[RawLayoutLMFeatures]]) -> LayoutLMFeatures:
        """
        Calling the DataCollator to form model inputs for training and inference. Takes a single raw
        :param raw_features: A dictionary with the following arguments: "image_id", "width", "height", "ann_ids", "words",
                             "boxes", "dataset_type".
        :return: LayoutLMFeatures
        """
        return raw_features_to_layoutlm_features(raw_features, self.tokenizer, self.padding, self.truncation,
                                                 self.return_overflowing_tokens, self.return_tensors, True)


@curry
def image_to_layoutlm_features(dp: Image,
                               tokenizer: "PreTrainedTokenizerFast",
                               return_tensors: Optional[Literal["pt"]] = None,
                               input_width: int = 1000,
                               input_height: int = 1000) -> LayoutLMFeatures:
    dp = image_to_raw_layoutlm_features(None, None, input_width, input_height)(dp)
    dp = raw_features_to_layoutlm_features(dp, tokenizer,return_tensors=return_tensors)
    return dp
