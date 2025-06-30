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
Module for mapping annotations to and from Xfund data structure
"""

import os
from collections import defaultdict
from itertools import chain
from typing import Mapping, Optional

from ..datapoint import BoundingBox, CategoryAnnotation, ContainerAnnotation, Image, ImageAnnotation
from ..utils.fs import load_image_from_file
from ..utils.settings import (
    BioTag,
    LayoutType,
    ObjectTypes,
    Relationships,
    TokenClasses,
    WordType,
    get_type,
    token_class_tag_to_token_class_with_tag,
)
from ..utils.types import FunsdDict
from .maputils import MappingContextManager, curry, maybe_get_fake_score


@curry
def xfund_to_image(
    dp: FunsdDict,
    load_image: bool,
    fake_score: bool,
    categories_dict_name_as_key: Mapping[ObjectTypes, int],
    token_class_names_mapping: Mapping[str, str],
    ner_token_to_id_mapping: Mapping[ObjectTypes, Mapping[ObjectTypes, Mapping[ObjectTypes, int]]],
) -> Optional[Image]:
    """
    Maps a datapoint of annotation structure as given from Xfund or Funsd dataset into an `Image` structure.

    Args:
        dp: A datapoint in dict structure as returned from the Xfund or Funsd dataset. Each datapoint must coincide
            with exactly one image sample.
        load_image: If `True`, it will load image to `Image.image`.
        fake_score: If `dp` does not contain a score, a fake score with uniform random variables in `(0,1)` will be
                    added.
        categories_dict_name_as_key: A mapping from `ObjectTypes` to `int` for `category_id`s.
        token_class_names_mapping: A dictionary mapping original label names to normalized category names.
        ner_token_to_id_mapping: A dictionary mapping token classes with bio tags (i.e. token tags) into their
                                 category ids.

    Returns:
        `Image` or `None` if the image path is not found or an error occurs during mapping.

    Note:
        This function is intended for mapping xfund or funsd dataset annotation structures to the internal `Image`
        representation for further processing.
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
    tag_to_id_mapping = ner_token_to_id_mapping[LayoutType.WORD][WordType.TAG]
    token_class_to_id_mapping = ner_token_to_id_mapping[LayoutType.WORD][WordType.TOKEN_CLASS]
    token_tag_to_id_mapping = ner_token_to_id_mapping[LayoutType.WORD][WordType.TOKEN_TAG]

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
            box = list(map(float, entity["box"]))
            bbox = BoundingBox(absolute_coords=True, ulx=box[0], uly=box[1], lrx=box[2], lry=box[3])
            score = maybe_get_fake_score(fake_score)
            entity_ann = ImageAnnotation(
                category_name=LayoutType.TEXT,
                bounding_box=bbox,
                category_id=categories_dict_name_as_key[LayoutType.TEXT],
                score=score,
            )
            category_name = token_class_names_mapping[entity["label"]]
            sub_cat_semantic = CategoryAnnotation(
                category_name=category_name, category_id=token_class_to_id_mapping[get_type(category_name)]
            )
            entity_ann.dump_sub_category(WordType.TOKEN_CLASS, sub_cat_semantic)
            image.dump(entity_ann)

            words = entity.get("words")

            for idx, word in enumerate(words):
                box = list(map(float, word["box"]))
                bbox = BoundingBox(absolute_coords=True, ulx=box[0], uly=box[1], lrx=box[2], lry=box[3])

                score = maybe_get_fake_score(fake_score)

                ann = ImageAnnotation(
                    category_name=LayoutType.WORD,
                    bounding_box=bbox,
                    category_id=categories_dict_name_as_key[LayoutType.WORD],
                    score=score,
                )
                image.dump(ann)
                entity_ann.dump_relationship(Relationships.CHILD, ann.annotation_id)
                sub_cat_semantic = CategoryAnnotation(
                    category_name=category_name, category_id=token_class_to_id_mapping[get_type(category_name)]
                )
                ann.dump_sub_category(WordType.TOKEN_CLASS, sub_cat_semantic)
                sub_cat_chars = ContainerAnnotation(category_name=WordType.CHARACTERS, value=word["text"])
                ann.dump_sub_category(WordType.CHARACTERS, sub_cat_chars)
                if sub_cat_semantic.category_name == TokenClasses.OTHER:
                    sub_cat_tag = CategoryAnnotation(
                        category_name=BioTag.OUTSIDE, category_id=tag_to_id_mapping[BioTag.OUTSIDE]
                    )
                    ann.dump_sub_category(WordType.TAG, sub_cat_tag)
                    # populating ner token to be used for training and evaluation
                    sub_cat_ner_tok = CategoryAnnotation(
                        category_name=BioTag.OUTSIDE, category_id=token_tag_to_id_mapping[BioTag.OUTSIDE]
                    )
                    ann.dump_sub_category(WordType.TOKEN_TAG, sub_cat_ner_tok)
                elif not idx:
                    sub_cat_tag = CategoryAnnotation(
                        category_name=BioTag.BEGIN, category_id=tag_to_id_mapping[BioTag.BEGIN]
                    )
                    ann.dump_sub_category(WordType.TAG, sub_cat_tag)
                    sub_cat_ner_tok = CategoryAnnotation(
                        category_name=token_class_tag_to_token_class_with_tag(
                            get_type(sub_cat_semantic.category_name), BioTag.BEGIN
                        ),
                        category_id=token_tag_to_id_mapping[
                            token_class_tag_to_token_class_with_tag(
                                get_type(sub_cat_semantic.category_name), BioTag.BEGIN
                            )
                        ],
                    )
                    ann.dump_sub_category(WordType.TOKEN_TAG, sub_cat_ner_tok)
                else:
                    sub_cat_tag = CategoryAnnotation(
                        category_name=BioTag.INSIDE, category_id=tag_to_id_mapping[BioTag.INSIDE]
                    )
                    ann.dump_sub_category(WordType.TAG, sub_cat_tag)
                    sub_cat_ner_tok = CategoryAnnotation(
                        category_name=token_class_tag_to_token_class_with_tag(
                            get_type(sub_cat_semantic.category_name), BioTag.INSIDE
                        ),
                        category_id=token_tag_to_id_mapping[
                            token_class_tag_to_token_class_with_tag(
                                get_type(sub_cat_semantic.category_name), BioTag.INSIDE
                            )
                        ],
                    )
                    ann.dump_sub_category(WordType.TOKEN_TAG, sub_cat_ner_tok)

                entity_id_to_ann_id[entity["id"]].append(ann.annotation_id)
                ann_id_to_entity_id[ann.annotation_id] = entity["id"]

            entity_id_to_entity_link_id[entity["id"]].extend(entity["linking"])

        # now populating semantic links
        word_anns = image.get_annotation(category_names=LayoutType.WORD)
        for word in word_anns:
            entity_id = ann_id_to_entity_id[word.annotation_id]
            all_linked_entities = list(chain(*entity_id_to_entity_link_id[entity_id]))
            ann_ids = []
            for linked_entity in all_linked_entities:
                ann_ids.extend(entity_id_to_ann_id[linked_entity])
            for ann_id in ann_ids:
                if ann_id != word.annotation_id:
                    word.dump_relationship(Relationships.LINK, ann_id)

    if mapping_context.context_error:
        return None
    return image
