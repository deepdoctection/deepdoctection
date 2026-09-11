# -*- coding: utf-8 -*-
# File: record_align.py

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
Record alignment and the metrics induced by it.

Records are nested structures of mappings, sequences and scalars, e.g. a single booking of a bank
statement together with its postings. Before two records can be aligned they have to be compared
without relying on the ordering of their sub-records. `bag_similarity` provides such an
order-independent score on top of `flatten`, and `align` uses that score to pair up two lists of
records.

On top of that, `field_counts` compares two structured outputs, e.g. the JSON returned by an
information extraction pipeline against a ground truth JSON of the same schema, and the
`RecordAlign*` metrics report precision, recall and F1 per field path with micro averaged variants.
Aligning arrays of records first is what makes a single missing record cost only its own values
instead of shifting every following record against the wrong partner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from lazy_imports import try_import
from tabulate import tabulate
from termcolor import colored

from dd_core.dataflow import DataFlow
from dd_core.mapper import image_or_docs_to_cat_id
from dd_core.utils.file_utils import Requirement, get_scipy_requirement
from dd_core.utils.logger import LoggingRecord, logger
from dd_core.utils.object_types import ObjectTypes, TypeOrStr, get_type
from dd_core.utils.types import MetricResults
from dd_core.utils.utils import as_string, flatten

from .base import MetricBase
from .registry import metric_registry

with try_import() as scipy_import_guard:
    from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from dd_datasets.info import DatasetCategories

__all__ = [
    "bag_similarity",
    "align",
    "Counts",
    "field_counts",
    "RecordAlignMetric",
    "RecordAlignPrecisionMetric",
    "RecordAlignRecallMetric",
    "RecordAlignF1Metric",
    "RecordAlignPrecisionMetricMicro",
    "RecordAlignRecallMetricMicro",
    "RecordAlignF1MetricMicro",
]

MIN_SIMILARITY = 0.1


def _value_bag(record: Any) -> Counter[tuple[str, str]]:
    """
    Reduces a record to the multiset of its `(field_path, value)` pairs.

    Args:
        record: An arbitrarily nested structure of mappings, sequences and scalars.

    Returns:
        A counter over the `(field_path, value)` pairs of the record.
    """
    return Counter((path, value) for path, values in flatten(record).items() for value in values)


def bag_similarity(record_gt: Any, record_pred: Any) -> float:
    """
    Computes the similarity of two records as the Jaccard index over their bags of values.

    Both records are flattened into `{field_path: [values]}` first. Values are then matched per field
    path as multisets, so that neither the ordering of sub-records nor the ordering of the field paths
    has any influence on the score.

    Args:
        record_gt: The ground truth record. An arbitrarily nested structure of mappings, sequences and
            scalars.
        record_pred: The predicted record, in the same format as `record_gt`.

    Returns:
        The Jaccard index in `[0.0, 1.0]`, where `1.0` means that both bags of values are identical.

    Example:
        ```python
        gt = {"amount": ["450,00-"], "postings": [{"postingType": ["net"]}]}
        pred = {"amount": ["450,00-"], "postings": []}
        bag_similarity(gt, pred)
        # Output: 0.5
        ```
    """
    bag_gt = _value_bag(record_gt)
    bag_pred = _value_bag(record_pred)

    matches = sum((bag_gt & bag_pred).values())
    total = sum(bag_gt.values()) + sum(bag_pred.values()) - matches

    # total is zero exactly when both records carry no value at all
    return matches / total if total else 1.0


def align(
    records_gt: list[Any],
    records_pred: list[Any],
    min_similarity: float = MIN_SIMILARITY,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Assigns predicted records to ground truth records.

    The cost of pairing two records is `1.0 - bag_similarity(...)`, so that identical records cost
    zero and records without any common value cost one. Solving the linear sum assignment problem on
    that cost matrix yields the pairing with the lowest total cost.

    Args:
        records_gt: Ground truth records, e.g. the elements of `bookings`.
        records_pred: Predicted records for the same field path.
        min_similarity: Lowest bag similarity that still counts as a pair. Zero accepts every pairing
            the solver produces.

    Returns:
        A tuple of three items. The first holds the accepted pairs as `(gt_index, pred_index)`, sorted
        by ground truth index.

    Note:
        Requires `scipy`.

    Example:
        ```python
        gt = [{"amount": ["1,00-"]}, {"amount": ["2,00-"]}]
        pred = [{"amount": ["2,00-"]}, {"amount": ["9,00-"]}]
        align(gt, pred)
        # Output: ([(1, 0)], [0], [1])
        ```
    """
    if not records_gt or not records_pred:
        return [], list(range(len(records_gt))), list(range(len(records_pred)))

    bags_gt = [_value_bag(record) for record in records_gt]
    bags_pred = [_value_bag(record) for record in records_pred]
    sizes_gt = [sum(bag.values()) for bag in bags_gt]
    sizes_pred = [sum(bag.values()) for bag in bags_pred]

    cost = np.empty((len(bags_gt), len(bags_pred)), dtype=np.float64)
    for row, (bag_gt, size_gt) in enumerate(zip(bags_gt, sizes_gt)):
        for column, (bag_pred, size_pred) in enumerate(zip(bags_pred, sizes_pred)):
            matches = sum(min(count, bag_pred[value]) for value, count in bag_gt.items())
            total = size_gt + size_pred - matches
            cost[row, column] = 1.0 - (matches / total if total else 1.0)

    rows, columns = linear_sum_assignment(cost)

    max_cost = 1.0 - min_similarity
    pairs: list[tuple[int, int]] = [
        (row, column) for row, column in zip(rows.tolist(), columns.tolist()) if cost[row, column] <= max_cost
    ]

    paired_gt = {row for row, _ in pairs}
    paired_pred = {column for _, column in pairs}
    unmatched_gt = [index for index in range(len(records_gt)) if index not in paired_gt]
    unmatched_pred = [index for index in range(len(records_pred)) if index not in paired_pred]

    return pairs, unmatched_gt, unmatched_pred


@dataclass
class Counts:
    """
    Confusion counts for a single field path.

    Attributes:
        tp (int): Values present in both ground truth and prediction.
        fp (int): Values predicted but not present in the ground truth.
        fn (int): Values in the ground truth but not predicted.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        """Share of predicted values that are correct, or zero if nothing was predicted"""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Share of ground truth values that were found, or zero if there are none"""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall, or zero if both are zero"""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0


def _charge_record(record: Any, prefix: str, counts: dict[str, Counts], as_false_negative: bool) -> None:
    """
    Charges every non empty value of an unmatched record as an error.

    Args:
        record: A record without a partner in the other structure.
        prefix: Field path the record sits at, e.g. `bookings`.
        counts: Accumulator, modified in place.
        as_false_negative: True for a ground truth record, False for a predicted one.
    """
    for relative_path, values in flatten(record).items():
        path = f"{prefix}.{relative_path}" if relative_path else prefix
        for _ in values:
            if as_false_negative:
                counts[path].fn += 1
            else:
                counts[path].fp += 1


def _walk(node_gt: Any, node_pred: Any, path: str, counts: dict[str, Counts]) -> None:
    """
    Recurses through both structures in parallel and fills the accumulator.

    Args:
        node_gt: Current ground truth node.
        node_pred: Current predicted node.
        path: Field path of the current node, empty at the root.
        counts: Accumulator, modified in place.
    """
    if isinstance(node_gt, dict) or isinstance(node_pred, dict):
        left = node_gt if isinstance(node_gt, dict) else {}
        right = node_pred if isinstance(node_pred, dict) else {}
        # a set union would not be order stable and the accumulator order would then differ across runs
        for key in list(left) + [key for key in right if key not in left]:
            child_path = f"{path}.{key}" if path else key
            _walk(left.get(key), right.get(key), child_path, counts)
        return

    gt_is_record_list = isinstance(node_gt, list) and any(isinstance(item, dict) for item in node_gt)
    pred_is_record_list = isinstance(node_pred, list) and any(isinstance(item, dict) for item in node_pred)

    if gt_is_record_list or pred_is_record_list:
        records_gt = node_gt if gt_is_record_list else []
        records_pred = node_pred if pred_is_record_list else []
        pairs, unmatched_gt, unmatched_pred = align(records_gt, records_pred)

        for gt_index, pred_index in pairs:
            _walk(records_gt[gt_index], records_pred[pred_index], path, counts)
        for gt_index in unmatched_gt:
            _charge_record(records_gt[gt_index], path, counts, as_false_negative=True)
        for pred_index in unmatched_pred:
            _charge_record(records_pred[pred_index], path, counts, as_false_negative=False)
        return

    left_value = as_string(node_gt)
    right_value = as_string(node_pred)
    if not left_value and not right_value:
        return
    if left_value == right_value:
        counts[path].tp += 1
        return
    if left_value:
        counts[path].fn += 1
    if right_value:
        counts[path].fp += 1


def field_counts(node_gt: Any, node_pred: Any) -> dict[str, Counts]:
    """
    Compares two structured outputs and counts matches per field path.

    Dicts are compared key by key, which is unambiguous because the schema fixes the keys. Arrays of
    records are aligned first, so that a missing record costs only its own values instead of shifting
    every following record against the wrong partner.

    Args:
        node_gt: Ground truth structure, normally the whole document.
        node_pred: Predicted structure following the same schema.

    Returns:
        Mapping from field path to its counts. Paths carry no list index, so all records of one array
        accumulate under the same path.

    Example:
        ```python
        gt = {"amount": "1,00-", "iban": "DE95"}
        pred = {"amount": "1,00-", "iban": "DE96"}
        field_counts(gt, pred)
        # Output: {'amount': Counts(tp=1, fp=0, fn=0), 'iban': Counts(tp=0, fp=1, fn=1)}
        ```
    """
    counts: dict[str, Counts] = defaultdict(Counts)
    _walk(node_gt, node_pred, "", counts)
    return dict(counts)


class RecordAlignMetric(MetricBase):
    """
    Base metric class for comparing structured outputs of two dataflows.

    Attributes:
        metric: The function that turns two structured outputs into counts per field path.
        mapper: Function to map images to `category_id`
        _cats: Optional sequence of `ObjectTypes`
        _sub_cats: Optional mapping of object types to object types or sequences of `ObjectTypes`
        _summary_sub_cats: Optional sequence of `ObjectTypes` for summary
        _id_name_or_value: Which of `id`, `name` or `value` `mapper` extracts for a sub category or a
                           summary sub category. Use `value` to compare a `ContainerAnnotation`'s value,
                           e.g. a `structured_output` summary sub category.
    """

    # a plain function assigned as a class attribute is returned unbound via cls.metric, which is why
    # field_counts takes no cls. mypy assumes a method here and binds the first argument away.
    metric = field_counts  # type: ignore[assignment]
    mapper = image_or_docs_to_cat_id
    _cats: Optional[Sequence[ObjectTypes]] = None
    _sub_cats: Optional[Union[Mapping[ObjectTypes, ObjectTypes], Mapping[ObjectTypes, Sequence[ObjectTypes]]]] = None
    _summary_sub_cats: Optional[Sequence[ObjectTypes]] = None
    _id_name_or_value: Literal["id", "name", "value"] = "id"

    @classmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dataflow_gt.reset_state()
        dataflow_predictions.reset_state()

        cls._category_sanity_checks(categories)
        if cls._cats is None and cls._sub_cats is None:
            cls._cats = categories.get_categories(as_dict=False, filtered=True)
        mapper_with_setting = cls.mapper(cls._cats, cls._sub_cats, cls._summary_sub_cats, cls._id_name_or_value)

        # returned images of gt and predictions are likely not in the same order. We therefore first
        # stream all data into a dict and pair them by image_id thereafter.
        structured_per_image_gt: dict[str, Any] = {}
        structured_per_image_predictions: dict[str, Any] = {}
        for dp_gt, dp_pd in zip(dataflow_gt, dataflow_predictions):
            node_gt, image_id_gt = mapper_with_setting(dp_gt)  # pylint: disable=E1102
            structured_per_image_gt[image_id_gt] = node_gt
            node_pd, image_id_pd = mapper_with_setting(dp_pd)  # pylint: disable=E1102
            structured_per_image_predictions[image_id_pd] = node_pd

        structured_gt: dict[str, Any] = {}
        structured_predictions: dict[str, Any] = {}
        for image_id, node_gt in structured_per_image_gt.items():
            structured_gt[image_id] = node_gt
            structured_predictions[image_id] = structured_per_image_predictions.get(image_id, {})

        return structured_gt, structured_predictions

    @classmethod
    def _get_counts_per_path(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> dict[str, Counts]:
        structured_gt, structured_predictions = cls.dump(dataflow_gt, dataflow_predictions, categories)

        counts_per_path: dict[str, Counts] = defaultdict(Counts)
        for image_id, node_gt in structured_gt.items():
            for path, entry in cls.metric(node_gt, structured_predictions[image_id]).items():
                counts_per_path[path] = counts_per_path[path] + entry
        return counts_per_path

    @classmethod
    def set_categories(
        cls,
        category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
        sub_category_names: Optional[
            Union[Mapping[TypeOrStr, TypeOrStr], Mapping[TypeOrStr, Sequence[TypeOrStr]]]
        ] = None,
        summary_sub_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
        id_name_or_value: Optional[Literal["id", "name", "value"]] = None,
    ) -> None:
        """
        Set categories that are supposed to be evaluated.

        If `sub_categories` have to be considered, they need to be passed explicitly.

        Example:
            ```python
            # Evaluate sub_cat1, sub_cat2 of cat1 and sub_cat3 of cat2
            set_categories(sub_category_names={cat1: [sub_cat1, sub_cat2], cat2: sub_cat3})
            ```

        Args:
            category_names: List of category names
            sub_category_names: Dict of categories and their sub categories to be evaluated
            summary_sub_category_names: String or list of summary sub categories
            id_name_or_value: Which of `id`, `name` or `value` `mapper` extracts for a sub category or a
                              summary sub category. Use `value` to compare a `ContainerAnnotation`'s value,
                              e.g. a `structured_output` summary sub category.
        """

        if category_names is not None:
            cls._cats = (
                [get_type(category_names)]
                if isinstance(category_names, str)
                else [get_type(category) for category in category_names]
            )
        if sub_category_names is not None:
            _sub_cats = {}
            if isinstance(list(sub_category_names.values())[0], list):
                for key, _ in sub_category_names.items():
                    _sub_cats[get_type(key)] = [get_type(item) for item in sub_category_names[key]]
            else:
                for key, _ in sub_category_names.items():
                    _sub_cats[get_type(key)] = get_type(sub_category_names[key])  # type: ignore
            cls._sub_cats = _sub_cats
        if summary_sub_category_names is not None:
            cls._summary_sub_cats = (
                [get_type(summary_sub_category_names)]
                if isinstance(summary_sub_category_names, str)
                else [get_type(category) for category in summary_sub_category_names]
            )
        if id_name_or_value is not None:
            cls._id_name_or_value = id_name_or_value

    @classmethod
    def _category_sanity_checks(cls, categories: DatasetCategories) -> None:
        cats = categories.get_categories(as_dict=False, filtered=True)
        if cats:
            sub_cats = categories.get_sub_categories(cats)
        else:
            sub_cats = categories.get_sub_categories()

        if cls._cats:
            for cat in cls._cats:
                if cat not in cats:
                    raise ValueError(f"{cat} must be in {cats}")
                assert cat in cats

        if cls._sub_cats:
            for key, val in cls._sub_cats.items():
                if set(val) > set(sub_cats[key]):
                    raise ValueError(f"set(val) = {set(val)} must be a sub set of sub_cats[{key}]={sub_cats[key]}")

        if cls._cats is None and cls._sub_cats is None and cls._summary_sub_cats is None:
            logger.warning(
                LoggingRecord(
                    "RecordAlign metric has not correctly been set up: No category, sub category or summary has "
                    "been defined, therefore it is undefined what to evaluate."
                )
            )

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_scipy_requirement()]

    @classmethod
    def print_result(cls) -> None:
        table = tabulate(
            [x.values() for x in cls._results],
            list(cls._results[0].keys()),
            tablefmt="pipe",
            stralign="center",
            numalign="left",
        )
        logger.info(LoggingRecord(f"{cls.name} results:\n {colored(table, 'cyan')}"))


@metric_registry.register("record_align_precision")
class RecordAlignPrecisionMetric(RecordAlignMetric):
    """
    Metric induced by `field_counts`. Will calculate the precision per field path
    """

    name = "Record Align Precision"

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        counts_per_path = cls._get_counts_per_path(dataflow_gt, dataflow_predictions, categories)

        results = []
        for path in sorted(counts_per_path):
            entry = counts_per_path[path]
            results.append(
                {
                    "key": path,
                    "val": float(entry.precision),
                    "num_samples": entry.tp + entry.fn,
                }
            )
        cls._results = results
        return results


@metric_registry.register("record_align_recall")
class RecordAlignRecallMetric(RecordAlignMetric):
    """
    Metric induced by `field_counts`. Will calculate the recall per field path
    """

    name = "Record Align Recall"

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        counts_per_path = cls._get_counts_per_path(dataflow_gt, dataflow_predictions, categories)

        results = []
        for path in sorted(counts_per_path):
            entry = counts_per_path[path]
            results.append(
                {
                    "key": path,
                    "val": float(entry.recall),
                    "num_samples": entry.tp + entry.fn,
                }
            )
        cls._results = results
        return results


@metric_registry.register("record_align_f1")
class RecordAlignF1Metric(RecordAlignMetric):
    """
    Metric induced by `field_counts`. Will calculate the F1 per field path
    """

    name = "Record Align F1"

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        counts_per_path = cls._get_counts_per_path(dataflow_gt, dataflow_predictions, categories)

        results = []
        for path in sorted(counts_per_path):
            entry = counts_per_path[path]
            results.append(
                {
                    "key": path,
                    "val": float(entry.f1),
                    "num_samples": entry.tp + entry.fn,
                }
            )
        cls._results = results
        return results


@metric_registry.register("record_align_precision_micro")
class RecordAlignPrecisionMetricMicro(RecordAlignMetric):
    """
    Metric induced by `field_counts`. Will calculate the micro average precision
    """

    name = "Record Align Micro Precision"

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        counts_per_path = cls._get_counts_per_path(dataflow_gt, dataflow_predictions, categories)

        summed = sum(counts_per_path.values(), Counts())
        results = [
            {
                "key": "total",
                "val": float(summed.precision),
                "num_samples": summed.tp + summed.fn,
            }
        ]
        cls._results = results
        return results


@metric_registry.register("record_align_recall_micro")
class RecordAlignRecallMetricMicro(RecordAlignMetric):
    """
    Metric induced by `field_counts`. Will calculate the micro average recall
    """

    name = "Record Align Micro Recall"

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        counts_per_path = cls._get_counts_per_path(dataflow_gt, dataflow_predictions, categories)

        summed = sum(counts_per_path.values(), Counts())
        results = [
            {
                "key": "total",
                "val": float(summed.recall),
                "num_samples": summed.tp + summed.fn,
            }
        ]
        cls._results = results
        return results


@metric_registry.register("record_align_f1_micro")
class RecordAlignF1MetricMicro(RecordAlignMetric):
    """
    Metric induced by `field_counts`. Will calculate the micro average F1
    """

    name = "Record Align Micro F1"

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        counts_per_path = cls._get_counts_per_path(dataflow_gt, dataflow_predictions, categories)

        summed = sum(counts_per_path.values(), Counts())
        results = [
            {
                "key": "total",
                "val": float(summed.f1),
                "num_samples": summed.tp + summed.fn,
            }
        ]
        cls._results = results
        return results
