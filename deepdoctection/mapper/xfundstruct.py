# -*- coding: utf-8 -*-
# File: xfundstruct.py

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
Module for mapping annotations to and from xfund data structure
"""

import os
from collections import defaultdict
from itertools import chain
from typing import Dict, Optional

from ..datapoint import BoundingBox, CategoryAnnotation, ContainerAnnotation, Image, ImageAnnotation
from ..utils.detection_types import JsonDict
from ..utils.fs import load_image_from_file
from ..utils.settings import names
from .maputils import MappingContextManager, cur, maybe_get_fake_score


@cur  # type: ignore
def xfund_to_image(
    dp: JsonDict, load_image: bool, fake_score: bool, category_names_mapping: Dict[str, str]
) -> Optional[Image]:
    """
    Map a datapoint of annotation structure as given as from xfund or funsd dataset in to an Image structure

    :param dp: A datapoint in dict structure as returned from the xfund or funsd dataset. Each datapoint must coincide
               with exactly one image sample.
    :param load_image: If 'True' it will load image to attr:`Image.image`
    :param fake_score: If dp does not contain a score, a fake score with uniform random variables in (0,1)
                       will be added.
    :param category_names_mapping: A dictionary, mapping original label names to normalized category names
    :return: Image
    """

    img = dp.get("img")
    if img is None:
        full_path = dp.get("file_name")
    else:
        full_path = img.get("fname")

    if full_path is None:
        return None

    _, file_name = os.path.split(full_path)
    external_id = dp.get("uid")

    with MappingContextManager(file_name) as mapping_context:

        image = Image(file_name=file_name, location=full_path, external_id=external_id)

        image.image = load_image_from_file(full_path)  # type: ignore

        if not load_image:
            image.clear_image()

        entity_id_to_ann_id = defaultdict(list)
        entity_id_to_entity_link_id = defaultdict(list)
        ann_id_to_entity_id = {}

        entities = dp.get("document")

        if entities is None:
            entities = dp.get("form")
        assert isinstance(entities, list)
        for entity in entities:
            words = entity.get("words")

            for idx, word in enumerate(words):
                box = list(map(float, word["box"]))
                bbox = BoundingBox(absolute_coords=True, ulx=box[0], uly=box[1], lrx=box[2], lry=box[3])

                score = maybe_get_fake_score(fake_score)
                category_name = category_names_mapping[entity["label"]]
                ann = ImageAnnotation(category_name=names.C.WORD, bounding_box=bbox, category_id="1", score=score)
                image.dump(ann)
                sub_cat_semantic = CategoryAnnotation(category_name=category_name)
                ann.dump_sub_category(names.C.SE, sub_cat_semantic)
                sub_cat_chars = ContainerAnnotation(category_name=names.C.CHARS, value=word["text"])
                ann.dump_sub_category(names.C.CHARS, sub_cat_chars)
                if sub_cat_semantic.category_name == names.C.O:
                    sub_cat_semantic = CategoryAnnotation(category_name=names.NER.O)
                    ann.dump_sub_category(names.NER.TAG, sub_cat_semantic)
                elif not idx:
                    sub_cat_semantic = CategoryAnnotation(category_name=names.NER.B)
                    ann.dump_sub_category(names.NER.TAG, sub_cat_semantic)
                else:
                    sub_cat_semantic = CategoryAnnotation(category_name=names.NER.I)
                    ann.dump_sub_category(names.NER.TAG, sub_cat_semantic)

                entity_id_to_ann_id[entity["id"]].append(ann.annotation_id)
                ann_id_to_entity_id[ann.annotation_id] = entity["id"]

            entity_id_to_entity_link_id[entity["id"]].extend(entity["linking"])

        # now populating semantic links
        word_anns = image.get_annotation()
        for word in word_anns:
            entity_id = ann_id_to_entity_id[word.annotation_id]
            all_linked_entities = list(chain(*entity_id_to_entity_link_id[entity_id]))
            ann_ids = []
            for linked_entity in all_linked_entities:
                ann_ids.extend(entity_id_to_ann_id[linked_entity])
            for ann_id in ann_ids:
                if ann_id != word.annotation_id:
                    word.dump_relationship(names.C.SEL, ann_id)

    if mapping_context.context_error:
        return None
    return image
