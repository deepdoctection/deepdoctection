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
Module for mapping annotations from image to layout lm input structure
"""
import numpy as np
from cv2 import INTER_LINEAR
from transformers import PreTrainedTokenizer

from dataflow.dataflow.imgaug.transform import ResizeTransform  # type: ignore

from ..datapoint.annotation import ContainerAnnotation
from ..datapoint.image import Image
from ..datapoint.convert import box_to_point4, point4_to_box
from ..utils.detection_types import JsonDict
from ..utils.settings import names
from .utils import cur


@cur  # type: ignore
def image_to_layoutlm(
    dp: Image, tokenizer: PreTrainedTokenizer, input_width: int = 1000, input_height: int = 1000
) -> JsonDict:
    """
    Maps an image to a dict that can be consumed by a tokenizer and ultimately be passed
    to a LayoutLM language model.

    :param dp: Image
    :param tokenizer: A tokenizer aligned with the following layout model
    :param input_width: Model image input width. Will resize the image and the bounding boxes
    :param input_height: Model image input height. Will resize the image and the bounding boxes
    """

    output: JsonDict = {}
    anns = dp.get_annotation_iter(category_names=names.C.WORD)
    all_tokens = []
    all_boxes = []
    all_ann_ids = []

    for ann in anns:
        char_cat = ann.get_sub_category(names.C.CHARS)
        assert isinstance(char_cat, ContainerAnnotation)
        word = char_cat.value
        word_tokens = tokenizer.tokenize(word)

        all_tokens.extend(word_tokens)
        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
            box = box.to_list(mode="xyxy")
        else:
            box = ann.bounding_box
        assert box is not None
        box = box.to_list(mode="xyxy")

        if word_tokens:
            all_boxes.extend([box] * len(word_tokens))
            all_ann_ids.extend([ann.annotation_id] * len(word_tokens))

    boxes = np.asarray(all_boxes, dtype="float32")
    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)
    image = resizer.apply_image(dp.image)
    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes)

    output["image"] = image
    output["ids"] = all_ann_ids
    output["boxes"] = boxes.tolist()
    output["words"] = all_tokens

    return output
