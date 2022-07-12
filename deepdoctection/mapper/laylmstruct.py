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

from typing import List, Optional, Dict, Literal

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
    from torch import clamp, round, tensor  # pylint: disable = W0622

if transformers_available():
    from transformers import PreTrainedTokenizer  # pylint: disable = W0611


__all__ = ["image_to_layoutlm"]


@deprecated("Use image_to_raw_layoutlm_features first, then tokenize and resize image and boxes", "2022-07-12")
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
    _CLS_BOX = [0.0, 0.0, 0.0, 0.0]
    _SEP_BOX = [1000.0, 1000.0, 1000.0, 1000.0]

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
                                   categories_dict_name_as_key: Optional[Dict[str,str]]=None,
                                   input_width: int = 1000,
                                   input_height: int = 1000) -> JsonDict:

    raw_features: JsonDict = {}
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
                is not None:
            semantic_label = ann.get_sub_category(names.C.SE).category_name
            bio_tag = ann.get_sub_category(names.NER.TAG).category_name
            if bio_tag == "O":
                category_name = "O"
            else:
                category_name = bio_tag + "-" + semantic_label
            all_labels.append(int(categories_dict_name_as_key[category_name]))

        if dp.summary is not None and categories_dict_name_as_key is not None:
            category_name = dp.summary.get_sub_category(names.C.DOC).category_name
            all_labels.append(int(categories_dict_name_as_key[category_name]))

    boxes = np.asarray(all_boxes, dtype="float32")
    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)

    if dp.image is not None:
        image = resizer.apply_image(dp.image)
        raw_features["image"] = image

    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes).tolist()

    raw_features["image_id"] = dp.image_id
    raw_features["width"] = dp.width
    raw_features["height"] = dp.height
    raw_features["ann_ids"] = all_ann_ids
    raw_features["words"] = all_words
    raw_features["boxes"] = boxes

    if categories_dict_name_as_key:
        raw_features["labels"] = all_labels

    return raw_features



# We assume batches when transforming raw features.
#Todo: Write collator that accepts list of dataset inputs and generates an appropriate input for this function and then
#Todo: calls this function. This collator can then be placed into HUggingface trainer.

def raw_features_to_layoutlm(dp: JsonDict, tokenizer: "PreTrainedTokenizer",
                             padding: Optional[Literal["max_length"]] = None,
                             truncation: bool = False,
                             return_overflow_tokens: bool = False,
                             is_split_into_words: bool = False
                             ):

    _label_on_token_level = len(dp["words"][0]) == len(dp["labels"])

    tokenized_inputs = tokenizer(
        dp["words"],
        padding=padding,
        truncation=truncation,
        return_overflowing_tokens=return_overflow_tokens,
        is_split_into_words=is_split_into_words,
    )

    labels = []
    bboxes = []
    images = []

    for batch_index in range(len(tokenized_inputs["input_ids"])):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]
        if _label_on_token_level:
            label = dp["labels"][org_batch_index]
        bbox = dp["bboxes"][org_batch_index]
        image = dp["image"][org_batch_index]
        previous_word_idx = None
        label_ids = []
        bbox_inputs = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
                bbox_inputs.append([0, 0, 0, 0])
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if _label_on_token_level:
                    label_ids.append(label[word_idx])
                bbox_inputs.append(bbox[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
                bbox_inputs.append(bbox[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
        bboxes.append(bbox_inputs)
        images.append(image)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["bbox"] = bboxes
    tokenized_inputs["image"] = images
    return tokenized_inputs








