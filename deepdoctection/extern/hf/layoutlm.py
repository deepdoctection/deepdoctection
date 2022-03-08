# -*- coding: utf-8 -*-
# File: layoutlm.py

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
Module for inference on layoutlm model
"""

from typing import List

from ...utils.file_utils import pytorch_available, transformers_available
from ..base import TokenClassResult

if pytorch_available():
    from torch import Tensor  # pylint: disable=W0611

if transformers_available():
    from transformers import LayoutLMForTokenClassification  # type: ignore # pylint: disable=W0611


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
    token_class_predictions = outputs.logits.argmax(-1).squeeze().tolist()
    input_ids_list = input_ids.squeeze().tolist()
    return [
        TokenClassResult(uuid=out[0], token_id=out[1], class_id=out[2], token=out[3])
        for out in zip(uuids, input_ids_list, token_class_predictions, tokens)
    ]
