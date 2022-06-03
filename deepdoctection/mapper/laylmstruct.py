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

from typing import List

import numpy as np
from cv2 import INTER_LINEAR
from dataflow.dataflow.imgaug.transform import ResizeTransform  # type: ignore

from ..datapoint.annotation import ContainerAnnotation
from ..datapoint.convert import box_to_point4, point4_to_box
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.file_utils import pytorch_available, transformers_available
from ..utils.settings import names
from .maputils import cur

if pytorch_available():
    from torch import clamp, round, tensor  # pylint: disable = E0611, W0611, W0622

if transformers_available():
    from transformers import PreTrainedTokenizer  # pylint: disable = W0611


__all__ = ["image_to_layoutlm"]


@cur  # type: ignore
def image_to_layoutlm(
    dp: Image, tokenizer: "PreTrainedTokenizer", input_width: int = 1000, input_height: int = 1000
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
    words: List[str] = []
    for ann in anns:
        char_cat = ann.get_sub_category(names.C.CHARS)
        assert isinstance(char_cat, ContainerAnnotation)
        word = char_cat.value
        assert isinstance(word, str)
        words.append(word)
        word_tokens = tokenizer.tokenize(word)
        all_tokens.extend(word_tokens)
        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
        else:
            box = ann.bounding_box
        assert box is not None
        box = box.to_list(mode="xyxy")

        if word_tokens:
            all_boxes.extend([box] * len(word_tokens))
            all_ann_ids.extend([ann.annotation_id] * len(word_tokens))

    all_boxes = [[0.0, 0.0, 0.0, 0.0]] + all_boxes + [[1000.0, 1000.0, 1000.0, 1000.0]]
    all_ann_ids = ["CLS"] + all_ann_ids + ["SEP"]
    all_tokens = ["CLS"] + all_tokens + ["SEP"]

    boxes = np.asarray(all_boxes, dtype="float32")
    boxes = box_to_point4(boxes)

    encoding = tokenizer(" ".join(words), return_tensors="pt")

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, INTER_LINEAR)
    image = resizer.apply_image(dp.image)
    boxes = resizer.apply_coords(boxes)
    boxes = point4_to_box(boxes)
    boxes = clamp(round(tensor([boxes.tolist()])), min=0.0, max=1000.0).int()  # type: ignore # pylint: disable = E1102

    output["image"] = image
    output["ids"] = all_ann_ids
    output["boxes"] = boxes
    output["tokens"] = all_tokens
    output["input_ids"] = encoding["input_ids"]
    output["attention_mask"] = encoding["attention_mask"]
    output["token_type_ids"] = encoding["token_type_ids"]

    return output
