# -*- coding: utf-8 -*-
# File: test_common.py
#
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
Testing module pipe.common
"""

from deepdoctection.datapoint import Image
from deepdoctection.pipe import MatchingService
from deepdoctection.utils import names


class TestMatchingService:
    """
    Test MatchingService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._parent_categories = names.C.CELL
        self._child_categories = names.C.WORD
        self._matching_rule = "ioa"
        self._iou_threshold = None
        self._ioa_threshold = 0.499

        self.matching_service = MatchingService(
            self._parent_categories,
            self._child_categories,
            self._matching_rule,
            self._iou_threshold if self._matching_rule in ["iou"] else self._ioa_threshold,  # type: ignore
        )

    def test_integration_pipeline_component(self, dp_image_fully_segmented_unrelated_words: Image) -> None:
        """
        test matching service assigns child annotation according to matching rule correctly to parental
        annotations
        """

        dp = dp_image_fully_segmented_unrelated_words

        # Act
        dp = self.matching_service.pass_datapoint(dp)

        parent_anns = dp.get_annotation(category_names=self._parent_categories)
        child_anns = dp.get_annotation(category_names=self._child_categories)

        relationships_word_first_parent = parent_anns[0].get_relationship(names.C.CHILD)
        relationships_word_third_parent = parent_anns[2].get_relationship(names.C.CHILD)

        # Assert
        assert len(relationships_word_first_parent) == 1
        assert len(relationships_word_third_parent) == 1

        assert relationships_word_first_parent[0] == child_anns[0].annotation_id
        assert relationships_word_third_parent[0] == child_anns[1].annotation_id
