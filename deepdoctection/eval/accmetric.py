# -*- coding: utf-8 -*-
# File: accmetric.py

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
Module for Accuracy metric
"""
from collections import Counter
from typing import Any
from typing import Counter as TypeCounter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import float32, int32
from numpy.typing import NDArray
from tabulate import tabulate
from termcolor import colored

from ..dataflow import DataFlow
from ..datasets.info import DatasetCategories
from ..mapper.cats import image_to_cat_id
from ..utils.detection_types import JsonDict
from ..utils.file_utils import Requirement
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import ObjectTypes, TypeOrStr, get_type
from .base import MetricBase
from .registry import metric_registry

__all__ = [
    "ClassificationMetric",
    "AccuracyMetric",
    "ConfusionMetric",
    "PrecisionMetric",
    "RecallMetric",
    "F1Metric",
    "PrecisionMetricMicro",
    "RecallMetricMicro",
    "F1MetricMicro",
]


def _mask_some_gt_and_pr_labels(
    np_label_gt: NDArray[int32], np_label_pr: NDArray[int32], masks: Sequence[int]
) -> Tuple[NDArray[int32], NDArray[int32]]:
    if len(np_label_gt) != len(masks):
        raise ValueError(f"length of label_gt ({len(np_label_gt)}) and masks ({len(masks)}) must be equal")
    np_masks = np.asarray(masks)
    np_masks.astype(bool)
    np_label_gt = np_label_gt[np_masks]
    np_label_pr = np_label_pr[np_masks]
    return np_label_gt, np_label_pr


def _confusion(np_label_gt: NDArray[int32], np_label_pr: NDArray[int32]) -> NDArray[int32]:
    number_classes = max(len(np.unique(np_label_gt)), np.amax(np_label_gt))  # type: ignore
    confusion_matrix = np.zeros((number_classes, number_classes), dtype=np.int32)  # type: ignore
    for i, gt_val in enumerate(np_label_gt):
        confusion_matrix[gt_val - 1][np_label_pr[i] - 1] += 1
    return confusion_matrix


def accuracy(label_gt: Sequence[int], label_predictions: Sequence[int], masks: Optional[Sequence[int]] = None) -> float:
    """
    Calculates the accuracy given predictions and labels. Ignores masked indices. Uses
    `sklearn.metrics.accuracy_score`

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: An optional list with masks to ignore some samples.

    :return: Accuracy score with only unmasked values to be considered
    """

    np_label_gt, np_label_pr = np.asarray(label_gt), np.asarray(label_predictions)
    if len(np_label_gt) != len(np_label_pr):
        raise ValueError(
            f"length label_gt: {len(np_label_gt)}, length label_predictions: ({len(np_label_pr)}) but must be equal"
        )
    if masks is not None:
        np_label_gt, np_label_pr = _mask_some_gt_and_pr_labels(np_label_gt, np_label_pr, masks)

    number_samples = np_label_gt.shape[0]
    return float((np_label_gt == np_label_pr).sum() / number_samples)


def confusion(
    label_gt: Sequence[int], label_predictions: Sequence[int], masks: Optional[Sequence[int]] = None
) -> NDArray[int32]:
    """
    Calculates the accuracy matrix given the predictions and labels. Ignores masked indices.

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: List with masks of same length as label_gt.

    :return: numpy array
    """

    np_label_gt, np_label_pr = np.asarray(label_gt), np.asarray(label_predictions)

    if masks is not None:
        np_label_gt, np_label_pr = _mask_some_gt_and_pr_labels(np_label_gt, np_label_pr, masks)

    return _confusion(np_label_gt, np_label_pr)


def precision(
    label_gt: Sequence[int],
    label_predictions: Sequence[int],
    masks: Optional[Sequence[int]] = None,
    micro: bool = False,
) -> NDArray[float32]:
    """
    Calculates the precision for a multi classification problem using a confusion matrix. The output will
    be the precision by category.

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: List with masks of same length as label_gt.
    :param micro: If True, it will calculate the micro average precision
    :return:  numpy array
    """
    np_label_gt, np_label_pr = np.asarray(label_gt), np.asarray(label_predictions)

    if masks is not None:
        np_label_gt, np_label_pr = _mask_some_gt_and_pr_labels(np_label_gt, np_label_pr, masks)
    confusion_matrix = _confusion(np_label_gt, np_label_pr)
    true_positive = np.diagonal(confusion_matrix)

    if micro:
        all_outputs = confusion_matrix.sum()
        return np.nan_to_num(true_positive.sum() / all_outputs)
    output_per_label = confusion_matrix.sum(axis=0)
    return np.nan_to_num(true_positive / output_per_label, nan=1.0)


def recall(
    label_gt: Sequence[int],
    label_predictions: Sequence[int],
    masks: Optional[Sequence[int]] = None,
    micro: bool = False,
) -> NDArray[float32]:
    """
    Calculates the recall for a multi classification problem using a confusion matrix. The output will
    be the recall by category.

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: List with masks of same length as label_gt.
    :param micro: If True, it will calculate the micro average recall
    :return:  numpy array
    """
    np_label_gt, np_label_pr = np.asarray(label_gt), np.asarray(label_predictions)

    if masks is not None:
        np_label_gt, np_label_pr = _mask_some_gt_and_pr_labels(np_label_gt, np_label_pr, masks)
    confusion_matrix = _confusion(np_label_gt, np_label_pr)
    true_positive = np.diagonal(confusion_matrix)

    if micro:
        all_outputs = confusion_matrix.sum()
        return np.nan_to_num(true_positive.sum() / all_outputs)
    number_gt_per_label = confusion_matrix.sum(axis=1)
    return np.nan_to_num(true_positive / number_gt_per_label, nan=1.0)


def f1_score(
    label_gt: Sequence[int],
    label_predictions: Sequence[int],
    masks: Optional[Sequence[int]] = None,
    micro: bool = False,
    per_label: bool = True,
) -> NDArray[float32]:
    """
    Calculates the recall for a multi classification problem using a confusion matrix. The output will
    be the recall by category.

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: List with masks of same length as label_gt.
    :param micro: If True, it will calculate the micro average f1 score
    :param per_label: If True, it will return the f1 score per label, otherwise will return the mean of all f1's
    :return:  numpy array
    """

    np_precision = precision(label_gt, label_predictions, masks, micro)
    np_recall = recall(label_gt, label_predictions, masks, micro)
    f_1 = 2 * np_precision * np_recall / (np_precision + np_recall)
    if per_label:
        return f_1
    return np.average(f_1)  # type: ignore


class ClassificationMetric(MetricBase):
    """
    Metric induced by `accuracy`
    """

    mapper = image_to_cat_id
    _cats: Optional[Sequence[ObjectTypes]] = None
    _sub_cats: Optional[Union[Mapping[ObjectTypes, ObjectTypes], Mapping[ObjectTypes, Sequence[ObjectTypes]]]] = None
    _summary_sub_cats: Optional[Sequence[ObjectTypes]] = None

    @classmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> Tuple[Any, Any]:
        dataflow_gt.reset_state()
        dataflow_predictions.reset_state()

        cls._category_sanity_checks(categories)
        if cls._cats is None and cls._sub_cats is None:
            cls._cats = categories.get_categories(as_dict=False, filtered=True)
        mapper_with_setting = cls.mapper(cls._cats, cls._sub_cats, cls._summary_sub_cats)
        labels_gt: Dict[str, List[int]] = {}
        labels_predictions: Dict[str, List[int]] = {}

        # returned images of gt and predictions are likely not in the same order. We therefore first stream all data
        # into a dict and generate our result vectors thereafter.
        labels_per_image_gt = {}
        labels_per_image_predictions = {}
        for dp_gt, dp_pd in zip(dataflow_gt, dataflow_predictions):
            dp_labels_gt, image_id_gt = mapper_with_setting(dp_gt)  # pylint: disable=E1102
            labels_per_image_gt[image_id_gt] = dp_labels_gt
            dp_labels_predictions, image_id_pr = mapper_with_setting(dp_pd)  # pylint: disable=E1102
            labels_per_image_predictions[image_id_pr] = dp_labels_predictions

        for image_id, dp_labels_gt in labels_per_image_gt.items():
            dp_labels_predictions = labels_per_image_predictions[image_id]
            for key in dp_labels_gt.keys():
                if key not in labels_gt:
                    labels_gt[key] = dp_labels_gt[key]
                    labels_predictions[key] = dp_labels_predictions[key]
                else:
                    labels_gt[key].extend(dp_labels_gt[key])
                    labels_predictions[key].extend(dp_labels_predictions[key])

        return labels_gt, labels_predictions

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> List[JsonDict]:
        labels_gt, labels_pr = cls.dump(dataflow_gt, dataflow_predictions, categories)

        results = []
        for key in labels_gt:  # pylint: disable=C0206
            res = cls.metric(labels_gt[key], labels_pr[key])
            results.append(
                {
                    "key": key.value if isinstance(key, ObjectTypes) else key,
                    "val": res,
                    "num_samples": len(labels_gt[key]),
                }
            )

        cls._results = results
        return results

    @classmethod
    def set_categories(
        cls,
        category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
        sub_category_names: Optional[
            Union[Mapping[TypeOrStr, TypeOrStr], Mapping[TypeOrStr, Sequence[TypeOrStr]]]
        ] = None,
        summary_sub_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    ) -> None:
        """
        Set categories that are supposed to be evaluated. If sub_categories have to be considered then they need to be
        passed explicitly.

        **Example:**

            You want to evaluate sub_cat1, sub_cat2 of cat1 and sub_cat3 of cat2. Set

                 sub_category_names = {cat1: [sub_cat1, sub_cat2], cat2: sub_cat3}


        :param category_names: List of category names
        :param sub_category_names: Dict of categories and their sub categories that are supposed to be evaluated,
                                   e.g. {"FOO": ["bak","baz"]} will evaluate "bak" and "baz"
        :param summary_sub_category_names: string or list of summary sub categories
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
                    "Accuracy metric has not correctly been set up: No category, sub category or summary has been "
                    "defined, therefore it is undefined what to evaluate."
                )
            )

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return []

    @property
    def sub_cats(
        self,
    ) -> Optional[Union[Mapping[ObjectTypes, ObjectTypes], Mapping[ObjectTypes, Sequence[ObjectTypes]]]]:
        """sub cats"""
        return self._sub_cats

    @property
    def summary_sub_cats(self) -> Optional[Sequence[ObjectTypes]]:
        """summary sub categories"""
        return self._summary_sub_cats

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


@metric_registry.register("accuracy")
class AccuracyMetric(ClassificationMetric):
    """
    Metric induced by `accuracy`
    """

    name = "Accuracy"
    metric = accuracy


@metric_registry.register("confusion")
class ConfusionMetric(ClassificationMetric):
    """
    Metric induced by `confusion`
    """

    name = "Confusion"
    metric = confusion

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> List[JsonDict]:
        labels_gt, labels_pr = cls.dump(dataflow_gt, dataflow_predictions, categories)

        results = []
        for key in labels_gt:  # pylint: disable=C0206
            confusion_matrix = cls.metric(labels_gt[key], labels_pr[key])
            number_labels: TypeCounter[int] = Counter(labels_gt[key])
            for row_number, row in enumerate(confusion_matrix, 1):
                for col_number, val in enumerate(row, 1):
                    results.append(
                        {
                            "key": key.value if isinstance(key, ObjectTypes) else key,
                            "category_id_gt": row_number,
                            "category_id_pr": col_number,
                            "val": float(val),
                            "num_samples_gt": number_labels[row_number],
                        }
                    )
        cls._results = results
        return results

    @classmethod
    def print_result(cls) -> None:
        data = {}
        for entry in cls._results:
            if entry["category_id_gt"] not in data:
                data[entry["category_id_gt"]] = [entry["category_id_gt"], entry["val"]]
            else:
                data[entry["category_id_gt"]].append(entry["val"])

        header = ["predictions -> \n  ground truth |\n              v"] + list(data.keys())
        table = tabulate([data[k] for k, _ in enumerate(data, 1)], headers=header, tablefmt="pipe")
        logger.info("Confusion matrix: \n %s", colored(table, "cyan"))


@metric_registry.register("precision")
class PrecisionMetric(ClassificationMetric):
    """
    Metric induced by `precision`. Will calculate the precision per category
    """

    name = "Precision"
    metric = precision

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> List[JsonDict]:
        labels_gt, labels_pr = cls.dump(dataflow_gt, dataflow_predictions, categories)

        results = []
        for key in labels_gt:  # pylint: disable=C0206
            score = cls.metric(labels_gt[key], labels_pr[key])
            number_labels: TypeCounter[int] = Counter(labels_gt[key])
            for label_id, val in enumerate(score, 1):
                results.append(
                    {
                        "key": key.value if isinstance(key, ObjectTypes) else key,
                        "category_id": label_id,
                        "val": float(val),
                        "num_samples": number_labels[label_id],
                    }
                )
        cls._results = results
        return results


@metric_registry.register("recall")
class RecallMetric(PrecisionMetric):
    """
    Metric induced by `recall`. Will calculate the recall per category
    """

    name = "Recall"
    metric = recall


@metric_registry.register("f1")
class F1Metric(PrecisionMetric):
    """
    Metric induced by `f1_score`. Will calculate the f1 per category
    """

    name = "F1"
    metric = f1_score


@metric_registry.register("precision_micro")
class PrecisionMetricMicro(ClassificationMetric):
    """
    Metric induced by `precision`. Will calculate the micro average precision
    """

    name = "Micro Precision "
    metric = precision

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> List[JsonDict]:
        labels_gt, labels_pr = cls.dump(dataflow_gt, dataflow_predictions, categories)

        results = []
        for key in labels_gt:  # pylint: disable=C0206
            score = cls.metric(labels_gt[key], labels_pr[key], micro=True)
            results.append(
                {
                    "key": key.value if isinstance(key, ObjectTypes) else key,
                    "val": float(score),
                    "num_samples": len(labels_gt[key]),
                }
            )
        cls._results = results
        return results


@metric_registry.register("recall_micro")
class RecallMetricMicro(PrecisionMetricMicro):
    """
    Metric induced by `recall`. Will calculate the micro average recall
    """

    name = "Micro Recall"
    metric = recall


@metric_registry.register("f1_micro")
class F1MetricMicro(PrecisionMetricMicro):
    """
    Metric induced by `f1_score`. Will calculate the micro average f1
    """

    name = "Micro F1"
    metric = f1_score
