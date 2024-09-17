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
from __future__ import annotations

import os.path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from lazy_imports import try_import

from ..datapoint.annotation import DEFAULT_CATEGORY_ID, ImageAnnotation
from ..datapoint.image import Image
from ..extern.pt.nms import batched_nms
from ..mapper.maputils import curry
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..utils.types import Detectron2Dict

with try_import() as pt_import_guard:
    import torch

with try_import() as d2_import_guard:
    from detectron2.structures import BoxMode

with try_import() as wb_import_guard:
    from wandb import Classes  # type: ignore
    from wandb import Image as Wbimage


@curry
def image_to_d2_frcnn_training(
    dp: Image,
    add_mask: bool = False,
    category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
) -> Optional[Detectron2Dict]:
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

    output: Detectron2Dict = {"file_name": str(dp.location)}

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
        box = ann.get_bounding_box(dp.image_id)
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)

        # Detectron2 does not fully support BoxMode.XYXY_REL
        mapped_ann: dict[str, Union[str, int, list[float]]] = {
            "bbox_mode": BoxMode.XYXY_ABS,
            "bbox": box.to_list(mode="xyxy"),
            "category_id": ann.category_id - 1,
        }
        annotations.append(mapped_ann)

        if add_mask:
            raise NotImplementedError("Segmentation in deepdoctection is not supported")

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

    # if all annotations are the same and prio is set, we need to return all annotation ids
    if prio and all(ann.category_name == anns[0].category_name for ann in anns):
        return [ann.annotation_id for ann in anns]

    if not anns:
        return []
    ann_ids = np.array([ann.annotation_id for ann in anns], dtype="object")
    # safety net to ensure we do not run into a ValueError
    boxes = torch.tensor(
        [ann.get_bounding_box(image_id).to_list(mode="xyxy") for ann in anns if ann.bounding_box is not None]
    )

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


def _get_category_attributes(
    ann: ImageAnnotation, cat_to_sub_cat: Optional[Mapping[ObjectTypes, ObjectTypes]] = None
) -> tuple[ObjectTypes, int, Optional[float]]:
    if cat_to_sub_cat:
        sub_cat_key = cat_to_sub_cat.get(get_type(ann.category_name))
        if sub_cat_key in ann.sub_categories:
            sub_cat = ann.get_sub_category(sub_cat_key)
            return get_type(sub_cat.category_name), sub_cat.category_id, sub_cat.score
        return DefaultType.DEFAULT_TYPE, DEFAULT_CATEGORY_ID, 0.0
    return get_type(ann.category_name), ann.category_id, ann.score


@curry
def to_wandb_image(
    dp: Image,
    categories: Mapping[int, TypeOrStr],
    sub_categories: Optional[Mapping[int, TypeOrStr]] = None,
    cat_to_sub_cat: Optional[Mapping[ObjectTypes, ObjectTypes]] = None,
) -> tuple[str, Wbimage]:
    """
    Converting a deepdoctection image into a wandb image

    :param dp: deepdoctection image
    :param categories: dict of categories. The categories refer to categories of `ImageAnnotation`s.
    :param sub_categories:  dict of sub categories. If provided, these categories will define the classes for the table
    :param cat_to_sub_cat: dict of category to sub category keys. Suppose your category `foo` has a sub category defined
                           by the key `sub_foo`. The range sub category values must then be given by `sub_categories`
                           and to extract the sub category values one must pass `{"foo": "sub_foo"}

    :return: a W&B image
    """
    if dp.image is None:
        raise ValueError("Cannot convert to W&B image type when Image.image is None")

    boxes = []
    anns = dp.get_annotation(category_names=list(categories.values()))

    if sub_categories:
        class_labels = dict(sub_categories.items())
        class_set = Classes([{"name": val, "id": key} for key, val in sub_categories.items()])
    else:
        class_set = Classes([{"name": val, "id": key} for key, val in categories.items()])
        class_labels = dict(categories.items())

    for ann in anns:
        bounding_box = ann.get_bounding_box(dp.image_id)
        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(dp.width, dp.height, True)
        category_name, category_id, score = _get_category_attributes(ann, cat_to_sub_cat)
        if category_name:
            box = {
                "position": {"middle": bounding_box.center, "width": bounding_box.width, "height": bounding_box.height},
                "domain": "pixel",
                "class_id": category_id,
                "box_caption": category_name,
            }
            if score:
                box["scores"] = {"acc": score}
            boxes.append(box)

    predictions = {"predictions": {"box_data": boxes, "class_labels": class_labels}}

    return dp.image_id, Wbimage(dp.image[:, :, ::-1], mode="RGB", boxes=predictions, classes=class_set)
