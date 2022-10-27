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
from typing import Mapping, Optional

from ..datapoint import BoundingBox, CategoryAnnotation, ContainerAnnotation, Image, ImageAnnotation
from ..utils.detection_types import JsonDict
from ..utils.fs import load_image_from_file
from ..utils.settings import (
    BioTag,
    LayoutType,
    Relationships,
    TokenClasses,
    WordType,
    get_type,
    token_class_tag_to_token_class_with_tag,
)
from .maputils import MappingContextManager, curry, maybe_get_fake_score


@curry
def xfund_to_image(
    dp: JsonDict,
    load_image: bool,
    fake_score: bool,
    category_names_mapping: Mapping[str, str],
    ner_token_to_id_mapping: Mapping[str, str],
) -> Optional[Image]:
    """
    Map a datapoint of annotation structure as given as from xfund or funsd dataset in to an Image structure

    :param dp: A datapoint in dict structure as returned from the xfund or funsd dataset. Each datapoint must coincide
               with exactly one image sample.
    :param load_image: If 'True' it will load image to attr:`Image.image`
    :param fake_score: If dp does not contain a score, a fake score with uniform random variables in (0,1)
                       will be added.
    :param category_names_mapping: A dictionary, mapping original label names to normalized category names
    :param ner_token_to_id_mapping: A dictionary, mapping token classes with bio tags (i.e. token tags) into their
                                    category ids.
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

        image.image = load_image_from_file(full_path)

        if not load_image:
            image.clear_image()

        entity_id_to_ann_id = defaultdict(list)
        entity_id_to_entity_link_id = defaultdict(list)
        ann_id_to_entity_id = {}

        entities = dp.get("document", [])

        if not entities:
            entities = dp.get("form", [])

        for entity in entities:
            words = entity.get("words")

            for idx, word in enumerate(words):
                box = list(map(float, word["box"]))
                bbox = BoundingBox(absolute_coords=True, ulx=box[0], uly=box[1], lrx=box[2], lry=box[3])

                score = maybe_get_fake_score(fake_score)
                category_name = category_names_mapping[entity["label"]]
                ann = ImageAnnotation(category_name=LayoutType.word, bounding_box=bbox, category_id="1", score=score)
                image.dump(ann)
                sub_cat_semantic = CategoryAnnotation(category_name=category_name)
                ann.dump_sub_category(WordType.token_class, sub_cat_semantic)
                sub_cat_chars = ContainerAnnotation(category_name=WordType.characters, value=word["text"])
                ann.dump_sub_category(WordType.characters, sub_cat_chars)
                if sub_cat_semantic.category_name == TokenClasses.other:
                    sub_cat_tag = CategoryAnnotation(category_name=BioTag.outside)
                    ann.dump_sub_category(WordType.tag, sub_cat_tag)
                    # populating ner token to be used for training and evaluation
                    sub_cat_ner_tok = CategoryAnnotation(
                        category_name=BioTag.outside, category_id=ner_token_to_id_mapping[BioTag.outside]
                    )
                    ann.dump_sub_category(WordType.token_tag, sub_cat_ner_tok)
                elif not idx:
                    sub_cat_tag = CategoryAnnotation(category_name=BioTag.begin)
                    ann.dump_sub_category(WordType.tag, sub_cat_tag)
                    sub_cat_ner_tok = CategoryAnnotation(
                        category_name=token_class_tag_to_token_class_with_tag(
                            get_type(sub_cat_semantic.category_name), BioTag.begin
                        ),
                        category_id=ner_token_to_id_mapping[
                            token_class_tag_to_token_class_with_tag(
                                get_type(sub_cat_semantic.category_name), BioTag.begin
                            )
                        ],
                    )
                    ann.dump_sub_category(WordType.token_tag, sub_cat_ner_tok)
                else:
                    sub_cat_tag = CategoryAnnotation(category_name=BioTag.inside)
                    ann.dump_sub_category(WordType.tag, sub_cat_tag)
                    sub_cat_ner_tok = CategoryAnnotation(
                        category_name=token_class_tag_to_token_class_with_tag(
                            get_type(sub_cat_semantic.category_name), BioTag.inside
                        ),
                        category_id=ner_token_to_id_mapping[
                            token_class_tag_to_token_class_with_tag(
                                get_type(sub_cat_semantic.category_name), BioTag.inside
                            )
                        ],
                    )
                    ann.dump_sub_category(WordType.token_tag, sub_cat_ner_tok)

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
                    word.dump_relationship(Relationships.semantic_entity_link, ann_id)

    if mapping_context.context_error:
        return None
    return image
