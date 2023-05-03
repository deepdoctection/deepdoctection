# -*- coding: utf-8 -*-
# File: d2struct.py

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
Module for mapping annotations into standard Detectron2 dataset dict. Also providing some tools for W&B mapping and
visualising
"""


import os.path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from detectron2.layers import batched_nms
from detectron2.structures import BoxMode

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..mapper.maputils import curry
from ..utils.detection_types import JsonDict
from ..utils.file_utils import wandb_available
from ..utils.settings import ObjectTypes, TypeOrStr

if wandb_available():
    from wandb import Classes
    from wandb import Image as Wbimage


@curry
def image_to_d2_frcnn_training(
    dp: Image,
    add_mask: bool = False,
    category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
) -> Optional[JsonDict]:
    """
    Maps an image to a standard dataset dict as described in
    <https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html>. It further checks if the image is physically
    available, for otherwise the annotation will be filtered.
    Note, that the returned dict will not suffice for training as gt for RPN and anchors still need to be created.

    :param dp: Image
    :param add_mask: True is not implemented (yet).
    :param category_names: A list of category names for training a model. Pass nothing to train with all annotations
    :return: Dict with 'image', 'width', 'height', 'image_id', 'annotations' where 'annotations' is a list of dict
             with 'bbox_mode' (D2 internal bounding box description), 'bbox' and 'category_id'.
    """
    if not os.path.isfile(dp.location) and dp.image is None:
        return None

    output: JsonDict = {"file_name": str(dp.location)}

    if dp.image is not None:
        output["image"] = dp.image.astype("float32")
    output["width"] = dp.width
    output["height"] = dp.height
    output["image_id"] = dp.image_id

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
            "bbox_mode": BoxMode.XYXY_ABS,
            "bbox": box.to_list(mode="xyxy"),
            "category_id": int(ann.category_id) - 1,
        }
        annotations.append(mapped_ann)

        if add_mask:
            raise NotImplementedError

    output["annotations"] = annotations

    return output


def pt_nms_image_annotations(
    anns: Sequence[ImageAnnotation], threshold: float, image_id: Optional[str] = None, prio: str = ""
) -> Sequence[str]:
    """
    Processing given image annotations through NMS. This is useful, if you want to supress some specific image
    annotation, e.g. given by name or returned through different predictors. This is the pt version, for tf check
    `mapper.tpstruct`

    :param anns: A sequence of ImageAnnotations. All annotations will be treated as if they belong to one category
    :param threshold: NMS threshold
    :param image_id: id in order to get the embedding bounding box
    :param prio: If an annotation has prio, it will overwrite its given score to 1 so that it will never be suppressed
    :return: A list of annotation_ids that belong to the given input sequence and that survive the NMS process
    """
    if len(anns) == 1:
        return [anns[0].annotation_id]
    if not anns:
        return []
    ann_ids = np.array([ann.annotation_id for ann in anns], dtype="object")
    if image_id:
        boxes = torch.tensor(
            [ann.image.get_embedding(image_id).to_list(mode="xyxy") for ann in anns if ann.image is not None]
        )
        # if we do not have image embeddings but pass an image_id
        if not boxes.shape[0]:
            boxes = torch.tensor(
                [ann.bounding_box.to_list(mode="xyxy") for ann in anns if ann.bounding_box is not None]
            )
    else:
        boxes = torch.tensor([ann.bounding_box.to_list(mode="xyxy") for ann in anns if ann.bounding_box is not None])

    def priority_to_confidence(ann: ImageAnnotation, priority: str) -> float:
        if ann.category_name == priority:
            return 1.0
        if ann.score:
            return ann.score
        raise ValueError("score cannot be None")

    scores = torch.tensor([priority_to_confidence(ann, prio) for ann in anns])
    class_mask = torch.ones(len(boxes), dtype=torch.uint8)
    keep = batched_nms(boxes, scores, class_mask, threshold)
    ann_ids_keep = ann_ids[keep]
    if not isinstance(ann_ids_keep, str):
        return ann_ids_keep.tolist()
    return []


@curry
def to_wandb_image(dp: Image, categories: Mapping[str, TypeOrStr]) -> Tuple[str, "Wbimage"]:
    """
    Converting a deepdoctection image into a wandb image

    :param dp: deepdoctection image
    :param categories: dict of categories
    :return: a W&B image
    """
    if dp.image is None:
        raise ValueError("Cannot convert to W&B image type when Image.image is None")

    boxes = []
    anns = dp.get_annotation(category_names=list(categories.values()))
    class_labels = {int(key): val for key, val in categories.items()}

    class_set = Classes([{"name": val, "id": int(key)} for key, val in categories.items()])

    for ann in anns:
        bounding_box = ann.image.get_embedding(dp.image_id) if ann.image is not None else ann.bounding_box
        box = {
            "position": {"middle": bounding_box.center, "width": bounding_box.width, "height": bounding_box.height},
            "domain": "pixel",
            "class_id": int(ann.category_id),
            "box_caption": ann.category_name,
        }
        if ann.score:
            box["scores"] = {"acc": ann.score}
        boxes.append(box)

    predictions = {"predictions": {"box_data": boxes, "class_labels": class_labels}}

    return dp.image_id, Wbimage(dp.image[:, :, ::-1], mode="RGB", boxes=predictions, classes=class_set)
