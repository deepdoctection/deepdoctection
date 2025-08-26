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
<https://github.com/NielsRogge/Transformers-Tutorials>
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NewType, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from lazy_imports import try_import

from ..datapoint.annotation import ContainerAnnotation
from ..datapoint.image import Image
from ..datapoint.view import Page
from ..utils.settings import DatasetType, LayoutType, PageType, Relationships, WordType
from ..utils.transform import ResizeTransform, box_to_point4, normalize_image, point4_to_box
from ..utils.types import JsonDict
from .maputils import curry

with try_import() as import_guard:
    import torch

with try_import() as tr_import_guard:
    from transformers import BatchEncoding, PreTrainedTokenizerFast  # pylint: disable=W0611

__all__ = [
    "image_to_raw_layoutlm_features",
    "raw_features_to_layoutlm_features",
    "LayoutLMDataCollator",
    "image_to_layoutlm_features",
    "DataCollator",
    "LayoutLMFeatures",
    "image_to_raw_lm_features",
    "image_to_lm_features",
]

RawLayoutLMFeatures = NewType("RawLayoutLMFeatures", JsonDict)
RawLMFeatures = NewType("RawLMFeatures", JsonDict)
LayoutLMFeatures = NewType("LayoutLMFeatures", JsonDict)
LMFeatures = NewType("LMFeatures", JsonDict)
InputDataClass = NewType("InputDataClass", JsonDict)


"""
<https://github.com/huggingface/transformers/src/transformers/data/data_collator.py>
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""

DataCollator = NewType("DataCollator", Callable[[list[InputDataClass]], dict[str, Any]])  # type: ignore

_CLS_BOX = [0.0, 0.0, 1000.0, 1000.0]
_SEP_BOX = [1000.0, 1000.0, 1000.0, 1000.0]


@curry
def image_to_raw_layoutlm_features(
    dp: Image,
    dataset_type: Optional[Literal["sequence_classification", "token_classification"]] = None,
    input_width: int = 1000,
    input_height: int = 1000,
    image_width: int = 1000,
    image_height: int = 1000,
    color_mode: Literal["BGR", "RGB"] = "BGR",
    pixel_mean: Optional[npt.NDArray[np.float32]] = None,
    pixel_std: Optional[npt.NDArray[np.float32]] = None,
    use_token_tag: bool = True,
    segment_positions: Optional[Union[LayoutType, Sequence[LayoutType]]] = None,
) -> Optional[RawLayoutLMFeatures]:
    """
    Maps a datapoint into an intermediate format for LayoutLM. Features are provided in a dict and this mapping
    can be used for sequence or token classification as well as for inference. To generate input features for the model
    please use `raw_features_to_layoutlm_features`.

    Args:
        dp: `Image`.
        dataset_type: Either `SEQUENCE_CLASSIFICATION` or `TOKEN_CLASSIFICATION`. When using a built-in dataset use
                      this.
        input_width: Max width of box coordinates. Transforms the image and all box coordinates accordingly.
        input_height: Target height of box coordinates. Transforms the image and all box coordinates accordingly.
        image_width: Some models (e.g. `Layoutlmv2`) assume box coordinates to be normalized to `input_width`, whereas
            the image has to be resized to a different width. This input will only resize the `image` width.
        image_height: Some models (e.g. `Layoutlmv2`) assume box coordinates to be normalized to `input_height`,
            whereas the image has to be resized to a different height. This input will only resize the `image` height.
        color_mode: Either `BGR` or `RGB`. Note that LayoutLMv2 uses `BGR` because of Detectron2 backbone, whereas
            LayoutLMv3 uses `RGB`.
        pixel_mean: (3,) array for `BGR` or `RGB` mean.
        pixel_std: (3,) array for `BGR` or `RGB` std.
        use_token_tag: Used only for `dataset_type="token_classification"`. If `True`, uses labels from subcategory
            `WordType.token_tag` (with `B,I,O` suffix), otherwise `WordType.token_class`.
        segment_positions: Using bounding boxes of segment instead of words improves model accuracy significantly.
            Choose a single or a sequence of layout segments to use their bounding boxes. The layout segments need to
            have a child-relationship with words. If a word does not appear as child, it will use the word bounding box.

    Returns:
        Dictionary with the following arguments: `image_id`, `width`, `height`, `ann_ids`, `words`, `bbox`, and
        `dataset_type`.
    """

    raw_features: RawLayoutLMFeatures = RawLayoutLMFeatures({})
    all_ann_ids = []
    all_words = []
    all_boxes = []
    all_labels: list[int] = []

    anns = dp.get_annotation(category_names=LayoutType.WORD)

    word_id_to_segment_box = {}
    if segment_positions:
        if isinstance(segment_positions, LayoutType):
            segment_positions = [segment_positions]
        segment_anns = dp.get_annotation(category_names=segment_positions)
        for segm_ann in segment_anns:
            bounding_box = segm_ann.get_bounding_box(dp.image_id)
            if not bounding_box.absolute_coords:
                bounding_box = bounding_box.transform(dp.width, dp.height, absolute_coords=True)
            word_id_to_segment_box.update(
                {word_ann: bounding_box for word_ann in segm_ann.get_relationship(Relationships.CHILD)}
            )

    for ann in anns:
        all_ann_ids.append(ann.annotation_id)
        char_cat = ann.get_sub_category(WordType.CHARACTERS)
        if not isinstance(char_cat, ContainerAnnotation):
            raise TypeError(f"char_cat must be of type ContainerAnnotation but is of type {type(char_cat)}")
        word = char_cat.value
        if not isinstance(word, str):
            raise TypeError(f"word must be of type str but is of type {type(word)}")
        all_words.append(word)

        box = ann.get_bounding_box(dp.image_id)
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)
        all_boxes.append(word_id_to_segment_box.get(ann.annotation_id, box).to_list(mode="xyxy"))

        if (
            WordType.TOKEN_TAG in ann.sub_categories or WordType.TOKEN_CLASS in ann.sub_categories
        ) and dataset_type == DatasetType.TOKEN_CLASSIFICATION:
            if use_token_tag:
                all_labels.append(ann.get_sub_category(WordType.TOKEN_TAG).category_id - 1)
            else:
                all_labels.append(ann.get_sub_category(WordType.TOKEN_CLASS).category_id - 1)

    if dataset_type == DatasetType.SEQUENCE_CLASSIFICATION:
        all_labels.append(dp.summary.get_sub_category(PageType.DOCUMENT_TYPE).category_id - 1)

    boxes = np.asarray(all_boxes, dtype="float32")
    if boxes.ndim == 1:
        return None

    boxes = box_to_point4(boxes)

    resizer = ResizeTransform(dp.height, dp.width, input_height, input_width, "VIZ")

    if dp.image is not None:
        if image_width != input_width or image_height != input_height:
            image_only_resizer = ResizeTransform(dp.height, dp.width, image_height, image_width, "VIZ")
            image = image_only_resizer.apply_image(dp.image)
        else:
            image = resizer.apply_image(dp.image)
        image_key = "image"
        if color_mode == "RGB":
            image = image[..., ::-1]
            image_key = "pixel_values"
        if pixel_mean is not None and pixel_std is not None:
            image = normalize_image(image, pixel_mean, pixel_std)
        raw_features[image_key] = image  # pylint: disable=E1137  #3162

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

    if all_labels:
        raw_features["labels"] = all_labels
    # pylint: enable=E1137
    return raw_features


def layoutlm_features_to_pt_tensors(features: LayoutLMFeatures) -> LayoutLMFeatures:
    """
    Converts a list of floats to PyTorch tensors.

    Args:
        features: `LayoutLMFeatures`.

    Returns:
        `LayoutLMFeatures`.
    """

    _image_key = "pixel_values" if "pixel_values" in features else "image"
    if "bbox" in features:
        features["bbox"] = torch.tensor(features["bbox"], dtype=torch.long)
    if "labels" in features:
        features["labels"] = torch.tensor(features["labels"], dtype=torch.long)
    if _image_key in features:
        features[_image_key] = torch.tensor(
            [image.astype("float32").transpose(2, 0, 1) for image in features[_image_key]], dtype=torch.float32
        )
        # features["images"] = [
        #    torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) for image in features["images"]
        # ]
    return features


def _tokenize_with_sliding_window(
    raw_features: list[Union[RawLayoutLMFeatures, RawLMFeatures]],
    tokenizer: PreTrainedTokenizerFast,
    sliding_window_stride: int,
    max_batch_size: int,
    return_tensors: Optional[Literal["pt"]] = None,
) -> Union[JsonDict, BatchEncoding]:
    """
    Runs a tokenizer. If there are no overflowing tokens, the tokenizer output will be returned as is.
    If there are overflowing tokens, sliding windows are built. Sliding windows are prepared from raw tokenized outputs
    by running the tokenizer a second time without truncating and building the sliding windows from this output.

    Note:
        The current implementation has a bug: for higher batch sizes it will only return overflowing samples.
        If the dataset consists of many samples with lots of tokens, use a low per device batch size.

    Args:
        raw_features: List of `RawLayoutLMFeatures` or `RawLMFeatures`.
        tokenizer: `PreTrainedTokenizerFast`.
        sliding_window_stride: Stride for sliding window.
        max_batch_size: Maximum batch size.
        return_tensors: If `pt`, returns torch tensors.

    Returns:
        `JsonDict` or `BatchEncoding`.
    """
    # first try: we require return_overflowing_tokens=True. If the number of raw features is equal to
    # overflow_to_sample_mapping then there is nothing more to do because the sample has less than max_length
    # tokens
    tokenized_inputs = tokenizer(
        [dp["words"] for dp in raw_features],
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_overflowing_tokens=True,
        is_split_into_words=True,
        return_tensors=return_tensors,
    )
    if len(raw_features) == len(tokenized_inputs["overflow_to_sample_mapping"]):
        return tokenized_inputs

    # now we tokenize with neither truncation nor padding and build sliding windows.
    tokenized_inputs = tokenizer(
        [dp["words"] for dp in raw_features],
        padding="do_not_pad",
        truncation=False,
        return_token_type_ids=True,
        return_overflowing_tokens=False,
        is_split_into_words=True,
        return_tensors=None,
    )
    max_length = tokenizer.max_len_single_sentence  # here 510 (512 minus [CLS] and [SEP]
    sliding_windows_remainder = [
        divmod(len(sample.ids) - max_length, sliding_window_stride) for sample in tokenized_inputs.encodings
    ]  # list of (multiplier, remainder) per sample
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_word_ids = []
    all_tokens = []
    overflow_to_sample_mapping = []
    for idx, outputs in enumerate(
        zip(
            tokenized_inputs.encodings,
            tokenized_inputs.data["input_ids"],
            tokenized_inputs.data["token_type_ids"],
            tokenized_inputs.data["attention_mask"],
            sliding_windows_remainder,
        )
    ):
        encodings = outputs[0]
        input_ids_orig = outputs[1]
        token_type_ids_orig = outputs[2]
        attention_mask_orig = outputs[3]
        divmod_result = outputs[4]
        tokens_orig = encodings.tokens
        word_ids_orig = encodings.word_ids
        multiplier, _ = divmod_result
        total_length = len(tokens_orig)
        # suppose the sample has total_length= 525 tokens:
        #  [[CLS],1,...,523,[SEP]]. With sliding_window_stride = 8 we build windows as follows:
        # [[CLS],1,..,510,[SEP]], [[CLS],8,..,517,[SEP]], [[CLS],16,..,523,[SEP],[PAD],[PAD],[PAD]]
        # Here, we have a multiplier = 1,
        for k in range(multiplier + 2):
            start = max(k * sliding_window_stride, 1)
            end = min(max_length + start, total_length)
            pad_last = max(max_length + start - end, 0)
            overflow_to_sample_mapping.append(idx)
            if not pad_last:
                tokens = tokens_orig[start:end]
                tokens.insert(0, tokenizer.cls_token)
                tokens.append(tokenizer.sep_token)
                all_tokens.append(tokens)
                input_ids = input_ids_orig[start:end]
                input_ids.insert(0, 101)
                input_ids.append(102)
                all_input_ids.append(input_ids)
                token_type_ids = token_type_ids_orig[start:end]
                token_type_ids.insert(0, 0)
                token_type_ids.append(0)
                all_token_type_ids.append(token_type_ids)
                attention_mask = attention_mask_orig[start:end]
                attention_mask.insert(0, 1)
                attention_mask.append(1)
                all_attention_mask.append(attention_mask)
                word_ids = word_ids_orig[start:end]
                word_ids.insert(0, None)
                word_ids.append(None)
                all_word_ids.append(word_ids)
            else:
                # last sliding window. We have to pad the end in order to have equal length along all windows.
                tokens = tokens_orig[start:end]
                tokens.insert(0, tokenizer.cls_token)
                pads = [tokenizer.pad_token for _ in range(pad_last + 1)]
                tokens.extend(pads)
                all_tokens.append(tokens)
                input_ids = input_ids_orig[start:end]
                input_ids.insert(0, 101)
                pad_ids = [0 for _ in range(pad_last + 1)]
                input_ids.extend(pad_ids)
                all_input_ids.append(input_ids)
                token_type_ids = token_type_ids_orig[start:end]
                token_type_ids.insert(0, 0)
                pad_ids = [0 for _ in range(pad_last + 1)]
                token_type_ids.extend(pad_ids)
                all_token_type_ids.append(token_type_ids)
                attention_mask = attention_mask_orig[start:end]
                attention_mask.insert(0, 1)
                pad_ids = [1 for _ in range(pad_last + 1)]
                attention_mask.extend(pad_ids)
                all_attention_mask.append(attention_mask)
                word_ids = word_ids_orig[start:end]
                word_ids.insert(0, None)
                pad_none = [None for _ in range(pad_last + 1)]
                word_ids.extend(pad_none)
                all_word_ids.append(word_ids)

    if max_batch_size:
        if max_batch_size < len(overflow_to_sample_mapping):
            (
                overflow_to_sample_mapping,
                all_input_ids,
                all_token_type_ids,
                all_attention_mask,
                all_word_ids,
                all_tokens,
            ) = zip(
                *random.sample(
                    list(
                        zip(
                            overflow_to_sample_mapping,
                            all_input_ids,
                            all_token_type_ids,
                            all_attention_mask,
                            all_word_ids,
                            all_tokens,
                        )
                    ),
                    max_batch_size,
                )
            )

    slided_tokenized_inputs: dict[str, Union[list[Union[str, int]], torch.Tensor]] = {}
    if return_tensors == "pt":
        slided_tokenized_inputs["overflow_to_sample_mapping"] = torch.tensor(overflow_to_sample_mapping)
        slided_tokenized_inputs["input_ids"] = torch.tensor(all_input_ids)
        slided_tokenized_inputs["token_type_ids"] = torch.tensor(all_token_type_ids)
        slided_tokenized_inputs["attention_mask"] = torch.tensor(all_attention_mask)
    else:
        slided_tokenized_inputs["overflow_to_sample_mapping"] = overflow_to_sample_mapping  # type: ignore
        slided_tokenized_inputs["input_ids"] = all_input_ids
        slided_tokenized_inputs["token_type_ids"] = all_token_type_ids
        slided_tokenized_inputs["attention_mask"] = all_attention_mask
    slided_tokenized_inputs["word_ids"] = all_word_ids
    slided_tokenized_inputs["tokens"] = all_tokens
    return slided_tokenized_inputs


def raw_features_to_layoutlm_features(
    raw_features: Union[RawLayoutLMFeatures, RawLMFeatures, list[Union[RawLayoutLMFeatures, RawLMFeatures]]],
    tokenizer: PreTrainedTokenizerFast,
    padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
    truncation: bool = True,
    return_overflowing_tokens: bool = False,
    return_tensors: Optional[Literal["pt"]] = None,
    remove_columns_for_training: bool = False,
    sliding_window_stride: int = 0,
    max_batch_size: int = 0,
    remove_bounding_boxes: bool = False,
) -> LayoutLMFeatures:
    """
    Maps raw features to tokenized input sequences for LayoutLM models.

    Args:
        raw_features: A dictionary with the following arguments: `image_id`, `width`, `height`, `ann_ids`, `words`,
            `boxes`, `dataset_type`.
        tokenizer: A fast tokenizer for the model. The conventional Python-based tokenizer provided by the
            Transformers library does not return essential word_id/token_id mappings, making feature generation
            more difficult. Only fast tokenizers are allowed.
        padding: Padding strategy to be passed to the tokenizer. Must be either `max_length`, `longest`, or
            `do_not_pad`.
        truncation: If `True`, truncates to a maximum length specified with the argument `max_length` or to the
            maximum acceptable input length for the model if that argument is not provided. Truncates token by token,
            removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is
            provided. If `False`, no truncation (i.e., can output batch with sequence lengths greater than the model
            maximum admissible input size).
        return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows, the overflowing tokens can
            be returned as an additional batch element. In this case, the number of input batch samples will be smaller
            than the output batch samples.
        return_tensors: If `pt`, returns torch tensors. If not provided, batches will be lists of lists.
        remove_columns_for_training: Removes all superfluous columns that are not required for training.
        sliding_window_stride: If the output of the tokenizer exceeds the `max_length` sequence length, sliding windows
            will be created with each window having `max_length` sequence input. When using
            `sliding_window_stride=0`, no strides will be created; otherwise, it will create slides with windows shifted
            `sliding_window_stride` to the right.
        max_batch_size: Maximum batch size.
        remove_bounding_boxes: If `True`, removes bounding box features.

    Returns:
        Dictionary with the following arguments: `image_ids`, `width`, `height`, `ann_ids`, `input_ids`,
        `token_type_ids`, `attention_mask`, `bbox`, `labels`.
    """

    if isinstance(raw_features, dict):
        raw_features = [raw_features]

    _has_token_labels = (
        raw_features[0]["dataset_type"] == DatasetType.TOKEN_CLASSIFICATION
        and raw_features[0].get("labels") is not None
    )
    _has_sequence_labels = (
        raw_features[0]["dataset_type"] == DatasetType.SEQUENCE_CLASSIFICATION
        and raw_features[0].get("labels") is not None
    )
    _has_labels = bool(_has_token_labels or _has_sequence_labels)
    _image_key = "pixel_values" if "pixel_values" in raw_features[0] else "image"

    if sliding_window_stride:
        return_overflowing_tokens = True
        tokenized_inputs = _tokenize_with_sliding_window(
            raw_features, tokenizer, sliding_window_stride, max_batch_size, return_tensors
        )

    else:
        tokenized_inputs = tokenizer(
            [dp["words"] for dp in raw_features],
            padding=padding,
            truncation=truncation,
            return_overflowing_tokens=return_overflowing_tokens,
            return_token_type_ids=True,
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

        if _image_key in raw_features[batch_index_orig]:
            images.append(raw_features[batch_index_orig][_image_key])
        image_ids.append(raw_features[batch_index_orig]["image_id"])
        widths.append(raw_features[batch_index_orig]["width"])
        heights.append(raw_features[batch_index_orig]["height"])

        ann_ids = raw_features[batch_index_orig]["ann_ids"]
        boxes = raw_features[batch_index_orig]["bbox"]
        if _has_token_labels:
            labels = raw_features[batch_index_orig]["labels"]

        if isinstance(tokenized_inputs, dict):
            word_ids = tokenized_inputs["word_ids"][batch_index]
            token_batch = tokenized_inputs["tokens"][batch_index]
        else:
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            token_batch = tokenized_inputs.tokens(batch_index=batch_index)

        token_batch_ann_ids = []
        token_batch_boxes = []
        token_batch_labels = []
        for idx, word_id in enumerate(word_ids):
            # Special tokens have a word id that is None. We make a lookup for the specific token and append a dummy
            # bounding box accordingly
            if word_id is None:
                if token_batch[idx] == tokenizer.cls_token:
                    token_batch_boxes.append(_CLS_BOX)
                    token_batch_ann_ids.append(tokenizer.cls_token)
                elif token_batch[idx] in (tokenizer.sep_token, tokenizer.pad_token):
                    token_batch_boxes.append(_SEP_BOX)
                    if token_batch[idx] == tokenizer.sep_token:
                        token_batch_ann_ids.append(tokenizer.sep_token)
                    else:
                        token_batch_ann_ids.append(tokenizer.pad_token)
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
        input_dict[_image_key] = images

    if _has_labels:
        input_dict["labels"] = token_labels if _has_token_labels else sequence_labels

    if remove_columns_for_training:
        input_dict.pop("image_ids")
        input_dict.pop("width")
        input_dict.pop("height")
        input_dict.pop("ann_ids")
        input_dict.pop("tokens")

    if remove_bounding_boxes:
        input_dict.pop("bbox")

    if return_tensors == "pt":
        return layoutlm_features_to_pt_tensors(LayoutLMFeatures(input_dict))
    return LayoutLMFeatures(input_dict)


@dataclass
class LayoutLMDataCollator:
    """
    Data collator that will dynamically tokenize, pad, and truncate the inputs received.

    Args:
        tokenizer: A fast tokenizer for the model. The conventional Python-based tokenizer provided by the
            Transformers library does not return essential word_id/token_id mappings, making feature generation
            more difficult. Only fast tokenizers are allowed.
        padding: Padding strategy to be passed to the tokenizer. Must be either `max_length`, `longest`, or
            `do_not_pad`.
        truncation: If `True`, truncates to a maximum length specified with the argument `max_length` or to the
            maximum acceptable input length for the model if that argument is not provided. Truncates token by token,
            removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is
            provided.
            If `False`, no truncation (i.e., can output batch with sequence lengths greater than the model maximum
            admissible input size).
        return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows, the overflowing tokens can
            be returned as an additional batch element. In this case, the number of input batch samples will be smaller
            than the output batch samples.
        return_tensors: If `pt`, returns torch tensors. If not provided, batches will be lists of lists.
        sliding_window_stride: If the output of the tokenizer exceeds the `max_length` sequence length, sliding windows
            will be created with each window having `max_length` sequence input. When using
            `sliding_window_stride=0`, no strides will be created; otherwise, it will create slides with windows
             shifted `sliding_window_stride` to the right.
        max_batch_size: Maximum batch size.
        remove_bounding_box_features: If `True`, removes bounding box features.
    """

    tokenizer: PreTrainedTokenizerFast
    padding: Literal["max_length", "do_not_pad", "longest"] = field(default="max_length")
    truncation: bool = field(default=True)
    return_overflowing_tokens: bool = field(default=False)
    return_tensors: Optional[Literal["pt"]] = field(default=None)
    sliding_window_stride: int = field(default=0)
    max_batch_size: int = field(default=0)
    remove_bounding_box_features: bool = field(default=False)

    def __post_init__(self) -> None:
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast), "Tokenizer must be a fast tokenizer"
        if self.return_tensors:
            assert self.padding not in ("do_not_pad",), self.padding
            assert self.truncation, self.truncation
        if self.return_overflowing_tokens:
            assert self.truncation, self.truncation

    def __call__(self, raw_features: Union[RawLayoutLMFeatures, list[RawLayoutLMFeatures]]) -> LayoutLMFeatures:
        """
        Calls the `DataCollator` to form model inputs for training and inference.

        Args:
            raw_features: A dictionary with the following arguments: `image_id`, `width`, `height`, `ann_ids`, `words`,
                `boxes`, `dataset_type`.

        Returns:
            `LayoutLMFeatures` with arguments `image_ids`, `width`, `height`, `ann_ids`, `input_ids`,
            `token_type_ids`, `attention_masks`, `boxes`, `labels`.
        """
        return raw_features_to_layoutlm_features(
            raw_features,  # type: ignore
            self.tokenizer,
            self.padding,
            self.truncation,
            self.return_overflowing_tokens,
            self.return_tensors,
            True,
            self.sliding_window_stride,
            self.max_batch_size,
            self.remove_bounding_box_features,
        )


@curry
def image_to_layoutlm_features(
    dp: Image,
    tokenizer: PreTrainedTokenizerFast,
    padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
    truncation: bool = True,
    return_overflowing_tokens: bool = False,
    return_tensors: Optional[Literal["pt"]] = "pt",
    input_width: int = 1000,
    input_height: int = 1000,
    image_width: int = 1000,
    image_height: int = 1000,
    color_mode: Literal["BGR", "RGB"] = "BGR",
    pixel_mean: Optional[npt.NDArray[np.float32]] = None,
    pixel_std: Optional[npt.NDArray[np.float32]] = None,
    segment_positions: Optional[Union[LayoutType, Sequence[LayoutType]]] = None,
    sliding_window_stride: int = 0,
) -> Optional[LayoutLMFeatures]:
    """
    Mapping function to generate LayoutLM features from `Image` to be used for inference in a pipeline component.
    `LanguageModelPipelineComponent` has a positional argument `mapping_to_lm_input_func` that must be chosen
    with respect to the language model chosen. This mapper is devoted to generating features for LayoutLM. It will be
    used internally in `LMTokenClassifierService`.

    Example:
        ```python
        tokenizer = LayoutLMTokenizer.from_pretrained("mrm8488/layoutlm-finetuned-funsd")
        layoutlm = HFLayoutLmTokenClassifier("path/to/config.json", "path/to/model.bin",
                                             categories_explicit=['B-ANSWER', 'B-QUESTION', 'O'])
        layoutlm_service = LMTokenClassifierService(tokenizer, layoutlm)
        ```

    Args:
        dp: `Image` datapoint.
        tokenizer: Tokenizer compatible with the language model.
        padding: Padding strategy to be passed to the tokenizer. Must be either `max_length`, `longest`, or
            `do_not_pad`.
        truncation: If `True`, truncates to a maximum length specified with the argument `max_length` or to the
            maximum acceptable input length for the model if that argument is not provided. Truncates token by token,
            removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is
            provided.
            If `False`, no truncation (i.e., can output batch with sequence lengths greater than the model maximum
            admissible input size).
        return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows, the overflowing tokens
            can be returned as an additional batch element. In this case, the number of input batch samples will be
            smaller than the output batch samples.
        return_tensors: Output tensor features. Either `pt` for PyTorch models or `None` if features should be
            returned in list objects.
        input_width: Standard input size for image coordinates. All LayoutLM models require input features to be
            normalized to an image width equal to 1000.
        input_height: Standard input size for image coordinates. All LayoutLM models require input features to be
            normalized to an image height equal to 1000.
        image_width: Some models (e.g. `Layoutlmv2`) assume box coordinates to be normalized to `input_width`, whereas
            the image has to be resized to a different width. This input will only resize the `image` width.
        image_height: Some models (e.g. `Layoutlmv2`) assume box coordinates to be normalized to `input_height`,
            whereas the image has to be resized to a different height. This input will only resize the `image` height.
        color_mode: Either `BGR` or `RGB`. Note that LayoutLMv2 uses `BGR` because of Detectron2 backbone, whereas
            LayoutLMv3 uses `RGB`.
        pixel_mean: (3,) array for `BGR` or `RGB` mean.
        pixel_std: (3,) array for `BGR` or `RGB` std.
        segment_positions: Using bounding boxes of segment instead of words improves model accuracy significantly.
            Choose a single or a sequence of layout segments to use their bounding boxes. The layout segments need to
            have a child-relationship with words. If a word does not appear as child, it will use the word bounding box.
        sliding_window_stride: If the output of the tokenizer exceeds the `max_length` sequence length, sliding
            windows will be created with each window having `max_length` sequence input. When using
            `sliding_window_stride=0`, no strides will be created; otherwise, it will create slides with windows shifted
            `sliding_window_stride` to the right.

    Returns:
        A dict of LayoutLM features.
    """
    raw_features = image_to_raw_layoutlm_features(
        None,
        input_width,
        input_height,
        image_width,
        image_height,
        color_mode,
        pixel_mean,
        pixel_std,
        True,
        segment_positions,
    )(dp)
    if raw_features is None:
        return None
    features = raw_features_to_layoutlm_features(
        raw_features,
        tokenizer,
        padding,
        truncation,
        return_overflowing_tokens,
        return_tensors=return_tensors,
        sliding_window_stride=sliding_window_stride,
    )
    return features


@curry
def image_to_raw_lm_features(
    dp: Image,
    dataset_type: Optional[Literal["sequence_classification", "token_classification"]] = None,
    use_token_tag: bool = True,
    text_container: Optional[LayoutType] = LayoutType.WORD,
    floating_text_block_categories: Optional[Sequence[LayoutType]] = None,
    include_residual_text_container: bool = False,
) -> Optional[RawLMFeatures]:
    """
    Maps a datapoint into an intermediate format for BERT-like models. Features are provided in a dict and
    this mapping can be used for sequence or token classification as well as for inference. To generate input features
    for the model, please use `raw_features_to_layoutlm_features`.

    Args:
        dp: `Image`.
        dataset_type: Either `SEQUENCE_CLASSIFICATION` or `TOKEN_CLASSIFICATION`. When using a built-in dataset use
                      this.
        use_token_tag: Used only for `dataset_type="token_classification"`. If `True`, uses labels from subcategory
            `WordType.token_tag` (with `B,I,O` suffix), otherwise `WordType.token_class`.
        text_container: A `LayoutType` to get the text from. It will steer the output of `Layout.words`.
        floating_text_block_categories: A list of top-level layout objects.
        include_residual_text_container: Regards synthetic text line annotations as floating text blocks and therefore
            incorporates all image annotations of category `word` when building text strings.

    Returns:
        Dictionary with the following arguments: `image_id`, `width`, `height`, `ann_ids`, `words`, `bbox`, and
        `dataset_type`.
    """

    raw_features: RawLMFeatures = RawLMFeatures({})

    # We do not need to configure residual_text_block_categories here, because text_ does ignore these layout sections
    # anyway
    page = Page.from_image(
        image_orig=dp,
        text_container=text_container,
        floating_text_block_categories=floating_text_block_categories,
        include_residual_text_container=include_residual_text_container,
    )

    text_ = page.text_

    # pylint: disable=E1137  #3162
    raw_features["image_id"] = page.image_id
    raw_features["width"] = page.width
    raw_features["height"] = page.height
    raw_features["ann_ids"] = text_.ann_ids
    raw_features["words"] = text_.words
    # We use a dummy bounding box for all bounding boxes so that we can pass the raw features to
    # raw_features_to_layoutlm_features
    raw_features["bbox"] = [_CLS_BOX] * len(text_.words)
    raw_features["dataset_type"] = dataset_type

    if use_token_tag and text_.token_tags:
        raw_features["labels"] = text_.token_tags
    elif text_.token_classes:
        raw_features["labels"] = text_.token_classes
    elif page.document_type is not None:
        document_type_id = page.image_orig.summary.get_sub_category(PageType.DOCUMENT_TYPE).category_id - 1
        raw_features["labels"] = [document_type_id]

    raw_features["dataset_type"] = dataset_type
    # pylint: enable=E1137
    return raw_features


@curry
def image_to_lm_features(
    dp: Image,
    tokenizer: PreTrainedTokenizerFast,
    padding: Literal["max_length", "do_not_pad", "longest"] = "max_length",
    truncation: bool = True,
    return_overflowing_tokens: bool = False,
    return_tensors: Optional[Literal["pt"]] = "pt",
    sliding_window_stride: int = 0,
    text_container: Optional[LayoutType] = LayoutType.WORD,
    floating_text_block_categories: Optional[Sequence[LayoutType]] = None,
    include_residual_text_container: bool = False,
) -> Optional[LayoutLMFeatures]:
    """
    Mapping function to generate LayoutLM features from `Image` to be used for inference in a pipeline component.
    `LanguageModelPipelineComponent` has a positional argument `mapping_to_lm_input_func` that must be chosen
    with respect to the language model chosen. This mapper is devoted to generating features for LayoutLM. It will be
    used internally in `LMTokenClassifierService`.

    Example:
        ```python
        tokenizer = LayoutLMTokenizer.from_pretrained("mrm8488/layoutlm-finetuned-funsd")
        layoutlm = HFLayoutLmTokenClassifier("path/to/config.json", "path/to/model.bin",
                                             categories_explicit=['B-ANSWER', 'B-QUESTION', 'O'])
        layoutlm_service = LMTokenClassifierService(tokenizer, layoutlm)
        ```

    Args:
        dp: `Image` datapoint.
        tokenizer: Tokenizer compatible with the language model.
        padding: Padding strategy to be passed to the tokenizer. Must be either `max_length`, `longest`, or
            `do_not_pad`.
        truncation: If `True`, truncates to a maximum length specified with the argument `max_length` or to the
            maximum acceptable input length for the model if that argument is not provided. Truncates token by token,
            removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is
            provided.
            If `False`, no truncation (i.e., can output batch with sequence lengths greater than the model maximum
            admissible input size).
        return_overflowing_tokens: If a sequence (due to a truncation strategy) overflows, the overflowing tokens
            can be returned as an additional batch element. In this case, the number of input batch samples will be
            smaller than the output batch samples.
        return_tensors: Output tensor features. Either `pt` for PyTorch models or `None` if features should be
            returned in list objects.
        sliding_window_stride: If the output of the tokenizer exceeds the `max_length` sequence length, sliding
            windows will be created with each window having `max_length` sequence input. When using
            `sliding_window_stride=0`, no strides will be created; otherwise, it will create slides with windows
             shifted `sliding_window_stride` to the right.
        text_container: A `LayoutType` to get the text from. It will steer the output of `Layout.words`.
        floating_text_block_categories: A list of top-level layout objects.
        include_residual_text_container: Regards synthetic text line annotations as floating text blocks and therefore
            incorporates all image annotations of category `word` when building text strings.

    Returns:
        A dict of LM features.
    """
    raw_features = image_to_raw_lm_features(  # pylint: disable=E1102
        dataset_type=None,
        use_token_tag=True,
        text_container=text_container,
        floating_text_block_categories=floating_text_block_categories,
        include_residual_text_container=include_residual_text_container,
    )(dp)
    if raw_features is None:
        return None
    features = raw_features_to_layoutlm_features(
        raw_features,
        tokenizer,
        padding,
        truncation,
        return_overflowing_tokens,
        return_tensors=return_tensors,
        sliding_window_stride=sliding_window_stride,
    )
    return features
