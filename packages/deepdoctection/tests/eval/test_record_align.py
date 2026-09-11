# -*- coding: utf-8 -*-
# File: test_record_align.py

# Copyright 2026 Dr. Janis Meyer. All rights reserved.
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
Unit tests for `deepdoctection.eval.record_align`.

`bag_similarity`, `align` and `field_counts` are tested first as pure functions on small synthetic
records, mirroring the examples in their own docstrings. The `eval_doc` fixture document then supplies a
real, resolved `structured_output` (an `accountHolder`/`balances`/`bookings` bank statement extraction)
to exercise the same functions on realistic, nested data. Finally the `RecordAlignMetric` family is
tested end to end against a `CustomDataset` built from the same document, mirroring the `accmetric`
tests: a datapoint evaluated against itself returns a perfect score.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dd_core.dataflow import MapData
from dd_core.mapper.cats import remove_cats
from dd_core.utils.object_types import LayoutLabel
from dd_datasets.base import CustomDataset
from deepdoctection.eval.record_align import (
    Counts,
    RecordAlignF1Metric,
    RecordAlignF1MetricMicro,
    RecordAlignMetric,
    RecordAlignPrecisionMetric,
    RecordAlignPrecisionMetricMicro,
    RecordAlignRecallMetric,
    RecordAlignRecallMetricMicro,
    align,
    bag_similarity,
    field_counts,
)


class TestCounts:
    """
    Test `Counts.precision`, `.recall` and `.f1`
    """

    @staticmethod
    def test_counts_properties_for_mismatch() -> None:
        """
        One false positive and one false negative yields zero precision, recall and f1
        """

        # Act & Assert
        counts = Counts(tp=0, fp=1, fn=1)
        assert counts.precision == 0.0
        assert counts.recall == 0.0
        assert counts.f1 == 0.0

    @staticmethod
    def test_counts_add() -> None:
        """
        `Counts` are summed component wise
        """

        # Act & Assert
        assert Counts(1, 2, 3) + Counts(1, 1, 1) == Counts(tp=2, fp=3, fn=4)


class TestBagSimilarity:
    """
    Test `bag_similarity`
    """

    @staticmethod
    def test_bag_similarity_partial_overlap() -> None:
        """
        Two records sharing one of two values score 0.5
        """

        # Arrange
        gt = {"amount": ["450,00-"], "postings": [{"postingType": ["net"]}]}
        pred = {"amount": ["450,00-"], "postings": []}

        # Act & Assert
        assert bag_similarity(gt, pred) == 0.5

    @staticmethod
    def test_bag_similarity_identical_records() -> None:
        """
        A record compared against itself scores 1.0
        """

        # Arrange
        record = {"amount": ["1,00-"], "iban": ["DE95"]}

        # Act & Assert
        assert bag_similarity(record, record) == 1.0

    @staticmethod
    def test_bag_similarity_no_common_value() -> None:
        """
        Two records without a single common value score 0.0
        """

        # Arrange
        gt = {"amount": ["1,00-"]}
        pred = {"amount": ["2,00-"]}

        # Act & Assert
        assert bag_similarity(gt, pred) == 0.0


class TestAlign:
    """
    Test `align`
    """

    @staticmethod
    def test_align_pairs_by_content_not_position() -> None:
        """
        Records are paired by content: the best matching pair need not sit at the same index
        """

        # Arrange
        gt = [{"amount": ["1,00-"]}, {"amount": ["2,00-"]}]
        pred = [{"amount": ["2,00-"]}, {"amount": ["9,00-"]}]

        # Act
        pairs, unmatched_gt, unmatched_pred = align(gt, pred)

        # Assert
        assert pairs == [(1, 0)]
        assert unmatched_gt == [0]
        assert unmatched_pred == [1]

    @staticmethod
    def test_align_one_side_empty() -> None:
        """
        Every record on the non-empty side is reported unmatched
        """

        # Arrange
        gt = [{"amount": ["1,00-"]}, {"amount": ["2,00-"]}]

        # Act
        pairs, unmatched_gt, unmatched_pred = align(gt, [])

        # Assert
        assert pairs == []
        assert unmatched_gt == [0, 1]
        assert unmatched_pred == []

    @staticmethod
    def test_align_below_min_similarity_stays_unmatched() -> None:
        """
        A pair without any common value is reported as one missing and one spurious record instead of
        as a badly matched pair
        """

        # Arrange
        gt = [{"amount": ["1,00-"]}]
        pred = [{"amount": ["9,00-"]}]

        # Act
        pairs, unmatched_gt, unmatched_pred = align(gt, pred)

        # Assert
        assert pairs == []
        assert unmatched_gt == [0]
        assert unmatched_pred == [0]


class TestFieldCounts:
    """
    Test `field_counts`
    """

    @staticmethod
    def test_field_counts_flat_dict() -> None:
        """
        One matching and one mismatching leaf yield one `Counts` per field path
        """

        # Arrange
        gt = {"amount": "1,00-", "iban": "DE95"}
        pred = {"amount": "1,00-", "iban": "DE96"}

        # Act & Assert
        assert field_counts(gt, pred) == {
            "amount": Counts(tp=1, fp=0, fn=0),
            "iban": Counts(tp=0, fp=1, fn=1),
        }

    @staticmethod
    def test_field_counts_missing_record_only_costs_its_own_values() -> None:
        """
        A missing record in a list of records costs only its own field values, the following record is
        not shifted against the wrong partner
        """

        # Arrange
        gt = {"bookings": [{"amount": ["1,00-"]}, {"amount": ["2,00-"]}]}
        pred = {"bookings": [{"amount": ["2,00-"]}]}

        # Act
        counts = field_counts(gt, pred)

        # Assert
        assert counts == {"bookings.amount": Counts(tp=1, fp=0, fn=1)}


class TestFieldCountsOnRealStructuredOutput:
    """
    Test `field_counts`, `bag_similarity` and `align` on the resolved `structured_output` of the
    `eval_doc` fixture document
    """

    @staticmethod
    def test_field_counts_perfect_match_against_itself(eval_doc_structured_output: dict[str, Any]) -> None:
        """
        The real structured output evaluated against itself has no false positives or false negatives
        """

        # Act
        counts = field_counts(eval_doc_structured_output, eval_doc_structured_output)

        # Assert
        assert counts
        assert all(entry.fp == 0 and entry.fn == 0 for entry in counts.values())

    @staticmethod
    def test_field_counts_detects_dropped_booking_and_changed_value(
        eval_doc_structured_output: dict[str, Any],
    ) -> None:
        """
        Dropping one booking from the prediction only charges that booking's own fields, and a changed
        value is counted as exactly one false positive/false negative pair
        """

        # Arrange
        gt = eval_doc_structured_output
        pred = copy.deepcopy(gt)
        pred["bookings"].pop(2)
        pred["balances"]["closingBalance"] = [None, "999,99-EUR"]

        # Act
        counts = field_counts(gt, pred)

        # Assert
        assert counts["balances.closingBalance"] == Counts(tp=0, fp=1, fn=1)
        assert counts["bookings.amount"] == Counts(tp=4, fp=0, fn=1)
        assert counts["bookings.bookingDate"] == Counts(tp=4, fp=0, fn=1)
        assert counts["bookings.description"] == Counts(tp=5, fp=0, fn=1)
        assert counts["bookings.valueDate"] == Counts(tp=2, fp=0, fn=1)
        # every other field path is unaffected by the two changes made above
        unaffected_paths = {
            path
            for path in counts
            if path
            not in {
                "balances.closingBalance",
                "bookings.amount",
                "bookings.bookingDate",
                "bookings.description",
                "bookings.valueDate",
            }
        }
        assert all(counts[path].fp == 0 and counts[path].fn == 0 for path in unaffected_paths)

    @staticmethod
    def test_align_pairs_bookings_by_content_after_one_is_dropped(
        eval_doc_structured_output: dict[str, Any],
    ) -> None:
        """
        Removing a booking in the middle of the list does not shift every following booking against the
        wrong partner: `align` still pairs each remaining booking with its true counterpart
        """

        # Arrange
        gt_bookings = eval_doc_structured_output["bookings"]
        pred_bookings = copy.deepcopy(gt_bookings)
        removed_index = 2
        pred_bookings.pop(removed_index)

        # Act
        pairs, unmatched_gt, unmatched_pred = align(gt_bookings, pred_bookings)

        # Assert
        assert unmatched_gt == [removed_index]
        assert unmatched_pred == []
        # every pair after the removed booking is shifted by one position in pred_bookings, but each gt
        # booking is still paired with the pred_booking holding the same content
        for gt_index, pred_index in pairs:
            assert bag_similarity(gt_bookings[gt_index], pred_bookings[pred_index]) == 1.0


@pytest.mark.skipif(CustomDataset is None, reason="dd_datasets is not installed; CustomDataset unavailable")
class TestRecordAlignMetric:
    """
    Test the `RecordAlignMetric` family end to end against a `CustomDataset` built from the `eval_doc`
    fixture document
    """

    @staticmethod
    def _reset(metric_cls: type[RecordAlignMetric]) -> None:
        metric_cls._cats = None  # pylint: disable=W0212
        metric_cls._sub_cats = None  # pylint: disable=W0212
        metric_cls._summary_sub_cats = None  # pylint: disable=W0212

    @staticmethod
    def test_f1_metric_returns_perfect_score_against_itself(eval_doc_dataset: CustomDataset) -> None:
        """
        When testing a dataflow against itself, every category reaches an F1 of 1.0
        """

        # Arrange
        dataflow_gt = eval_doc_dataset.dataflow_builder.build(mode="image")
        dataflow_pred = eval_doc_dataset.dataflow_builder.build(mode="image")

        # Act
        output = RecordAlignF1Metric.get_distance(dataflow_gt, dataflow_pred, eval_doc_dataset.dataflow.categories)

        # Assert
        assert output == [
            {"key": LayoutLabel.TABLE, "val": 1.0, "num_samples": 2},
            {"key": LayoutLabel.TEXT, "val": 1.0, "num_samples": 2},
            {"key": LayoutLabel.TITLE, "val": 1.0, "num_samples": 2},
        ]

        # Clean-up
        TestRecordAlignMetric._reset(RecordAlignF1Metric)

    @staticmethod
    def test_precision_recall_f1_isolate_the_missed_category(eval_doc_dataset: CustomDataset) -> None:
        """
        A prediction that never detects `table` scores 0.0 on `table` only, `text` and `title` are
        unaffected
        """

        # Arrange
        dataflow_gt = eval_doc_dataset.dataflow_builder.build(mode="image")
        dataflow_pred = eval_doc_dataset.dataflow_builder.build(mode="image")
        dataflow_pred = MapData(dataflow_pred, remove_cats(category_names=[LayoutLabel.TABLE]))
        categories = eval_doc_dataset.dataflow.categories

        # Act
        precision = RecordAlignPrecisionMetric.get_distance(dataflow_gt, dataflow_pred, categories)
        TestRecordAlignMetric._reset(RecordAlignPrecisionMetric)
        dataflow_pred = eval_doc_dataset.dataflow_builder.build(mode="image")
        dataflow_pred = MapData(dataflow_pred, remove_cats(category_names=[LayoutLabel.TABLE]))
        recall = RecordAlignRecallMetric.get_distance(dataflow_gt, dataflow_pred, categories)
        TestRecordAlignMetric._reset(RecordAlignRecallMetric)
        dataflow_pred = eval_doc_dataset.dataflow_builder.build(mode="image")
        dataflow_pred = MapData(dataflow_pred, remove_cats(category_names=[LayoutLabel.TABLE]))
        f1 = RecordAlignF1Metric.get_distance(dataflow_gt, dataflow_pred, categories)

        # Assert
        assert precision == [
            {"key": LayoutLabel.TABLE, "val": 0.0, "num_samples": 2},
            {"key": LayoutLabel.TEXT, "val": 1.0, "num_samples": 2},
            {"key": LayoutLabel.TITLE, "val": 1.0, "num_samples": 2},
        ]
        assert recall == [
            {"key": LayoutLabel.TABLE, "val": 0.0, "num_samples": 2},
            {"key": LayoutLabel.TEXT, "val": 1.0, "num_samples": 2},
            {"key": LayoutLabel.TITLE, "val": 1.0, "num_samples": 2},
        ]
        assert f1 == [
            {"key": LayoutLabel.TABLE, "val": 0.0, "num_samples": 2},
            {"key": LayoutLabel.TEXT, "val": 1.0, "num_samples": 2},
            {"key": LayoutLabel.TITLE, "val": 1.0, "num_samples": 2},
        ]

        # Clean-up
        TestRecordAlignMetric._reset(RecordAlignF1Metric)

    @staticmethod
    def test_micro_variants_average_over_all_categories(eval_doc_dataset: CustomDataset) -> None:
        """
        The micro variants report a single row that averages counts over all three categories instead of
        one row per category
        """

        # Arrange
        categories = eval_doc_dataset.dataflow.categories

        def build_pred() -> Any:
            dataflow_pred = eval_doc_dataset.dataflow_builder.build(mode="image")
            return MapData(dataflow_pred, remove_cats(category_names=[LayoutLabel.TABLE]))

        # Act
        precision = RecordAlignPrecisionMetricMicro.get_distance(
            eval_doc_dataset.dataflow_builder.build(mode="image"), build_pred(), categories
        )
        TestRecordAlignMetric._reset(RecordAlignPrecisionMetricMicro)
        recall = RecordAlignRecallMetricMicro.get_distance(
            eval_doc_dataset.dataflow_builder.build(mode="image"), build_pred(), categories
        )
        TestRecordAlignMetric._reset(RecordAlignRecallMetricMicro)
        f1 = RecordAlignF1MetricMicro.get_distance(
            eval_doc_dataset.dataflow_builder.build(mode="image"), build_pred(), categories
        )

        # Assert
        assert precision == [{"key": "total", "val": 1.0, "num_samples": 6}]
        assert recall == [{"key": "total", "val": 2 / 3, "num_samples": 6}]
        assert f1 == [{"key": "total", "val": 0.8, "num_samples": 6}]

        # Clean-up
        TestRecordAlignMetric._reset(RecordAlignF1MetricMicro)
