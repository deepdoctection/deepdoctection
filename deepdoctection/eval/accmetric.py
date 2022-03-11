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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore
from tabulate import tabulate
from termcolor import colored

from ..dataflow import DataFlow
from ..datasets.info import DatasetCategories
from ..mapper.cats import image_to_cat_id
from ..utils.detection_types import JsonDict
from .base import MetricBase

__all__ = ["AccuracyMetric", "ConfusionMetric"]


def accuracy(label_gt: List[np.int32], label_predictions: List[np.int32], masks: Optional[List[int]] = None) -> float:
    """
    Calculates the accuracy given predictions and labels. Ignores masked indices. Uses
    :func:`sklearn.metrics.accuracy_score`

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: An optional list with masks to ignore some samples.

    :return: Accuracy score with only unmasked values to be considered
    """

    np_label_gt, np_label_predictions = np.asarray(label_gt), np.asarray(label_predictions)
    assert len(np_label_gt) == len(
        np_label_predictions
    ), f"length of label_gt ({len(np_label_gt)}) and label_predictions ({len(np_label_gt)}) must be equal"

    if masks is not None:
        assert len(np_label_gt) == len(
            masks
        ), f"length of label_gt ({len(np_label_gt)}) and label_predictions ({len(masks)}) must be equal"
        masks = np.asarray(masks)  # type: ignore
        masks.astype(bool)  # type: ignore
        np_label_gt = np_label_gt[masks]
        np_label_predictions = np_label_predictions[masks]
    return accuracy_score(np_label_gt, np_label_predictions)


class AccuracyMetric(MetricBase):
    """
    Metric induced by :func:`accuracy`
    """

    metric = accuracy  # type: ignore
    mapper = image_to_cat_id
    _cats = None
    _sub_cats = None

    @classmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> Tuple[Any, Any]:

        dataflow_gt.reset_state(), dataflow_predictions.reset_state()  # pylint: disable=W0106

        cls._category_sanity_checks(categories)
        if cls._cats is None and cls._sub_cats is None:
            cls._cats = categories.get_categories(as_dict=False, filtered=True)
        mapper_with_setting = cls.mapper(cls._cats, cls._sub_cats)  # type: ignore
        labels_gt: Dict[str, List[int]] = {}
        labels_predictions: Dict[str, List[int]] = {}
        for dp_gt, dp_pd in zip(dataflow_gt, dataflow_predictions):
            dp_labels_gt = mapper_with_setting(dp_gt)  # pylint: disable=E1102
            dp_labels_predictions = mapper_with_setting(dp_pd)  # pylint: disable=E1102
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
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories, as_dict: bool = False
    ) -> Union[List[JsonDict], JsonDict]:

        labels_gt, labels_predictions = cls.dump(dataflow_gt, dataflow_predictions, categories)

        results = []
        for key in labels_gt:  # pylint: disable=C0206
            res = cls.metric(labels_gt[key], labels_predictions[key])  # type: ignore
            results.append({"key": key, "val": res, "num_samples": len(labels_gt[key])})
        if as_dict:
            return cls.result_list_to_dict(results)
        return results

    @classmethod
    def set_categories(
        cls,
        category_names: Optional[Union[str, List[str]]] = None,
        sub_category_names: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
    ) -> None:
        """
        Set categories that are supposed to be evaluated. If sub_categories have to be considered then they need to be
        passed explicitly.

        **Example:**

            You want to evaluate sub_cat1, sub_cat2 of cat1 and sub_cat3 of cat2. Set

            .. code-block:: python

                 sub_category_names = {cat1: [sub_cat1, sub_cat2], cat2: sub_cat3}


        :param category_names: List of category names
        :param sub_category_names: Dict of categories and their sub categories that are supposed to be evaluated,
                                   e.g. {"FOO": ["bak","baz"]} will evaluate "bak" and "baz"
        """

        if category_names is not None:
            cls._cats = category_names  # type: ignore
        if sub_category_names is not None:
            cls._sub_cats = sub_category_names

    @classmethod
    def _category_sanity_checks(cls, categories: DatasetCategories) -> None:
        cats = categories.get_categories(as_dict=False, filtered=True)
        if cats:
            sub_cats = categories.get_sub_categories(cats)  # type: ignore
        else:
            sub_cats = categories.get_sub_categories()

        if cls._cats:
            for cat in cls._cats:
                assert cat in cats

        if cls._sub_cats:
            for key, val in cls._sub_cats.items():
                assert set(val) <= set(sub_cats[key])


def confusion(
    label_gt: List[int], label_predictions: List[int], masks: Optional[List[bool]] = None
) -> NDArray[np.float32]:
    """
    Calculates the accuracy matrix given the predictions and labels. Ignores masked indices. Uses
    :func:`sklearn.metrics.confusion_matrix`

    :param label_gt: List of ground truth labels
    :param label_predictions: List of predictions. Must have the same length as label_gt
    :param masks: List with masks of same length as label_gt.

    :return: numpy array
    """

    np_label_gt, np_label_predictions = np.asarray(label_gt), np.asarray(label_predictions)

    if masks is not None:
        masks = np.asarray(masks)  # type: ignore
        masks.astype(bool)  # type: ignore
        np_label_gt = np_label_gt[masks]
        np_label_predictions = np_label_predictions[masks]

    return confusion_matrix(np_label_gt, np_label_predictions)


class ConfusionMetric(AccuracyMetric):
    """
    Metric induced by :func:`confusion`
    """

    metric = confusion  # type: ignore

    @classmethod
    def print_distance(cls, results: List[JsonDict]) -> None:
        """
        print distance results
        """
        key_list = [list(k.keys())[0] for k in results]
        for key, result in zip(key_list, results):
            data = []
            header: List[Union[int, str]] = ["predictions -> \n ground truth |\n              v"]
            conf = result[key]
            for idx, row in enumerate(conf):
                row = row.tolist()
                row.insert(0, idx)
                data.append(row)
                header.append(idx)
            table = tabulate(data, headers=header, tablefmt="pipe")  # type: ignore
            print(f"Confusion matrix for {key}: \n" + colored(table, "cyan"))
