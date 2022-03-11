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

from ..datapoint.image import Image
from ..mapper.maputils import MappingContextManager
from ..mapper.match import match_anns_by_intersection
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
        )

        with MappingContextManager(dp_name=dp.file_name):
            for idx, ann in enumerate(child_anns):
                child_positions = child_index == idx
                parents_of_child = [parent_anns[k] for k in parent_index[child_positions]]
                # todo: to avoid duplicated and crashes thereafter, choose argmax for assigning words to items
                if len(parents_of_child) >= 1:
                    parents_of_child[0].dump_relationship(names.C.CHILD, ann.annotation_id)
