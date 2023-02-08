# -*- coding: utf-8 -*-
# File: hfstruct.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Module for mapping annotations into standard Huggingface Detr input structure for training
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from transformers import BatchFeature, DetrFeatureExtractor

from ..datapoint.image import Image
from ..mapper.maputils import curry
from ..mapper.misc import get_load_image_func
from ..utils.detection_types import JsonDict
from ..utils.settings import ObjectTypes
from ..utils.transform import PadTransform


@curry
def image_to_hf_detr_training(
    dp: Image,
    add_mask: bool = False,
    category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
) -> Optional[JsonDict]:
    """
    Maps an image to a detr input datapoint dict, that, after collating can be used for training.

    :param dp: Image
    :param add_mask: True is not implemented (yet).
    :param category_names: A list of category names for training a model. Pass nothing to train with all annotations
    :return: Dict with 'image', 'width', 'height', 'image_id', 'annotations' where 'annotations' is a list of dict
             with 'boxes' and 'class_labels'.
    """

    if not os.path.isfile(dp.location) and dp.image is None:
        return None

    output: JsonDict = {"file_name": dp.location}

    if dp.image is not None:
        output["image"] = dp.image.astype("float32")
    output["width"] = dp.width
    output["height"] = dp.height
    output["image_id"] = "".join([c for c in dp.image_id if c.isdigit()])[:8]

    anns = dp.get_annotation(category_names=category_names)

    if not anns:
        return None

    annotations = []

    for ann in anns:
        if ann.image is not None:
            box = ann.image.get_embedding(dp.image_id)
        else:
            box = ann.bounding_box
        if box is None:
            raise ValueError("BoundingBox cannot be None")
        mapped_ann: Dict[str, Union[str, int, List[float]]] = {
            "id": "".join([c for c in ann.annotation_id if c.isdigit()])[:8],
            "image_id": "".join([c for c in dp.image_id if c.isdigit()])[:8],
            "bbox": box.to_list(mode="xywh"),
            "category_id": int(ann.category_id) - 1,
            "area": box.area,
        }
        annotations.append(mapped_ann)

    if add_mask:
        raise NotImplementedError

    output["annotations"] = annotations

    return output


@dataclass
class DetrDataCollator:
    """
    Data collator that will prepare a list of raw features to a BatchFeature that can be used
    to train a Detr or Tabletransformer model.

    :param feature_extractor:  DetrFeatureExtractor
    :param padder: An optional PadTransform instance
    :param return_tensors: "pt" or None
    """

    feature_extractor: DetrFeatureExtractor  # TODO: Replace deprecated DetrFeatureExtractor with DetrImageProcessor
    padder: Optional[PadTransform] = None
    return_tensors: Optional[Literal["pt"]] = field(default="pt")

    def __call__(self, raw_features: List[JsonDict]) -> BatchFeature:
        """
        Creating BatchFeature from a list of dict of raw features.

        :param raw_features: A list of dict with keys: 'image' or 'file_name', "width', "height' and 'annotations'.
                             'annotations' mus be a list of dict as well, where each dict element must contain
                             annotation information following COCO standard.
        :return: BatchFeature
        """
        images_input = []

        for feature in raw_features:
            maybe_image = feature.get("image")
            if maybe_image is None:
                load_func = get_load_image_func(feature["file_name"])
                feature["image"] = load_func(feature["file_name"])
                feature = self.maybe_pad_image_and_transform(feature)
                images_input.append(feature["image"])
            else:
                images_input.append(maybe_image)
        image_features = self.feature_extractor(  # pylint: disable=E1102
            images_input, annotations=raw_features, return_tensors=self.return_tensors
        )

        return image_features

    def maybe_pad_image_and_transform(self, feature: JsonDict) -> JsonDict:
        """
        Pads an 'image' and transforming bounding boxes from annotations.

        :param feature: A dict of raw_features
        :return: Same as input
        """
        if self.padder is None:
            return feature
        feature["image"] = self.padder.apply_image(feature["image"])
        feature["height"] = feature["image"].shape[0]
        feature["width"] = feature["image"].shape[1]
        boxes = np.array([ann["bbox"] for ann in feature["annotations"]])
        box_transform = self.padder.apply_coords(boxes)
        for idx, ann in enumerate(feature["annotations"]):
            ann["bbox"] = box_transform[:, idx].tolist()
        return feature
