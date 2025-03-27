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
from pytest import mark

from deepdoctection.datapoint import Image
from deepdoctection.pipe import AnnotationNmsService, FamilyCompound, IntersectionMatcher, MatchingService
from deepdoctection.utils.settings import LayoutType, Relationships


class TestMatchingService:
    """
    Test MatchingService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._parent_categories = LayoutType.CELL
        self._child_categories = LayoutType.WORD
        self._matching_rule = "ioa"
        self._iou_threshold = 0.499
        self._ioa_threshold = 0.499
        self.matcher = IntersectionMatcher(
            self._matching_rule,  # type: ignore
            self._iou_threshold if self._matching_rule in ["iou"] else self._ioa_threshold,
        )
        self.family_compounds = [
            FamilyCompound(
                parent_categories=self._parent_categories,
                child_categories=self._child_categories,
                relationship_key=Relationships.CHILD,
            )
        ]

        self.matching_service = MatchingService(
            family_compounds=self.family_compounds,
            matcher=self.matcher,
        )

    @mark.basic
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

        relationships_word_first_parent = parent_anns[0].get_relationship(Relationships.CHILD)
        relationships_word_third_parent = parent_anns[2].get_relationship(Relationships.CHILD)

        # Assert
        assert len(relationships_word_first_parent) == 1
        assert len(relationships_word_third_parent) == 1

        assert relationships_word_first_parent[0] == child_anns[0].annotation_id
        assert relationships_word_third_parent[0] == child_anns[1].annotation_id


class TestAnnotationNmsService:
    """
    Test AnnotationNmsService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._nms_pairs = [["row", "column"]]
        self._thresholds = [0.01]
        self._categories = ["row", "column"]

        self.nms_service = AnnotationNmsService(self._nms_pairs, self._thresholds)

    @mark.pt_deps
    def test_integration_pipeline_component(self, dp_image_fully_segmented_unrelated_words: Image) -> None:
        """
        test annotation nms service suppresses the annotations within groups
        """

        dp = dp_image_fully_segmented_unrelated_words

        # Act
        dp = self.nms_service.pass_datapoint(dp)

        anns = dp.get_annotation(category_names=self._categories)

        # Assert
        assert len(anns) == 2
