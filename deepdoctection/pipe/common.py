# -*- coding: utf-8 -*-
# File: common.py

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
Module for common pipeline components
"""

from typing import List, Optional, Union

import numpy as np

from ..dataflow import DataFlow, MapData  # type: ignore
from ..datapoint.doc import Page
from ..datapoint.image import Image
from ..mapper.maputils import MappingContextManager
from ..mapper.match import match_anns_by_intersection
from ..mapper.pagestruct import to_page
from ..utils.settings import names
from .base import PipelineComponent


class MatchingService(PipelineComponent):
    """
    Objects of two object classes can be assigned to one another by determining their pairwise average. If this is above
    a limit, a relation is created between them.
    The parent object class (based on its category) and the child object class are defined for the service. A child
    relation is created in the parent class if the conditions are met.

    Either iou (intersection-over-union) or ioa (intersection-over-area) can be selected as the matching rule.
    """

    def __init__(
        self,
        parent_categories: Union[str, List[str]],
        child_categories: Union[str, List[str]],
        matching_rule: str,
        threshold: Optional[np.float32] = None,
    ) -> None:
        """
        :param parent_categories: list of categories to be used a for parent class. Will generate a child-relationship
        :param child_categories: list of categories to be used for a child class.
        :param matching_rule: "iou" or "ioa"
        :param threshold: iou/ioa threshold. Value between [0,1]
        """
        super().__init__(None)
        self.parent_categories = parent_categories
        self.child_categories = child_categories
        assert matching_rule in ["iou", "ioa"], "segment rule must be either iou or ioa"
        self.matching_rule = matching_rule
        self.threshold = threshold

    def serve(self, dp: Image) -> None:
        """
        - generates pairwise match-score by intersection
        - generates child relationship at parent level

        :param dp: datapoint image
        """
        child_index, parent_index, child_anns, parent_anns = match_anns_by_intersection(
            dp,
            parent_ann_category_names=self.parent_categories,
            child_ann_category_names=self.child_categories,
            matching_rule=self.matching_rule,
            threshold=self.threshold,  # type: ignore
            max_parent_only=True,
        )

        with MappingContextManager(dp_name=dp.file_name):
            matched_child_anns = np.take(child_anns, child_index)  # type: ignore
            matched_parent_anns = np.take(parent_anns, parent_index)  # type: ignore
            for idx, parent in enumerate(matched_parent_anns):
                parent.dump_relationship(names.C.CHILD, matched_child_anns[idx].annotation_id)

    def clone(self) -> PipelineComponent:
        return self.__class__(self.parent_categories, self.child_categories, self.matching_rule, self.threshold)


class PageParsingService:
    """
    A "pseudo" pipeline component that can be added to a pipeline to convert Images into Page formats. It allows a
    custom parsing depending on customizing options of other pipeline components
    """

    def __init__(
        self,
        text_container: str,
        floating_text_block_names: Optional[Union[str, List[str]]] = None,
        text_block_names: Optional[Union[str, List[str]]] = None,
        text_containers_to_text_block: bool = False,
    ):
        """

        :param text_container: name of an image annotation that has a CHARS sub category. These annotations will be
                               ordered within all text blocks.
        :param floating_text_block_names: name of image annotation that belong to floating text. These annotations form
                                          the highest hierarchy of text blocks that will ordered to generate a sensible
                                          output of text
        :param text_block_names: name of image annotation that have a relation with text containers (or which might be
                                 text containers themselves).
        :param text_containers_to_text_block: Text containers are in general no text blocks and belong to a lower
                                             hierarchy. However, if a text container is not assigned to a text block
                                             you can add it to the text block ordering to ensure that the full text is
                                             part of the subsequent sub process.
        """
        if isinstance(floating_text_block_names, str):
            floating_text_block_names = [floating_text_block_names]
        elif floating_text_block_names is None:
            floating_text_block_names = []
        if isinstance(text_block_names, str):
            text_block_names = [text_block_names]
        elif text_block_names is None:
            text_block_names = []

        self._text_container = text_container
        self._floating_text_block_names = floating_text_block_names
        self._text_block_names = text_block_names
        self._text_container_to_text_block = text_containers_to_text_block
        self._init_sanity_checks()

    def pass_datapoint(self, dp: Image) -> Page:
        """
        converts Image to Page
        :param dp: Image
        :return: Page
        """
        return to_page(
            dp,
            self._text_container,
            self._floating_text_block_names,
            self._text_block_names,
            self._text_container_to_text_block,
        )

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via :meth:`pass_datapoint` within a dataflow pipeline

        :param df: An input dataflow
        :return: A output dataflow
        """
        return MapData(df, self.pass_datapoint)

    def _init_sanity_checks(self) -> None:
        assert self._text_container in [names.C.WORD, names.C.LINE], (
            f"text_container must be either {names.C.WORD} or " f"{names.C.LINE}"
        )
        assert set(self._floating_text_block_names) <= set(
            self._text_block_names
        ), "floating_text_block_names must be a subset of text_block_names"
