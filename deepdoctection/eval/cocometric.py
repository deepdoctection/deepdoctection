# -*- coding: utf-8 -*-
# File: test_cocometric.py

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
Metrics that require the `COCOeval` class.
"""
from __future__ import annotations

from copy import copy
from typing import Optional, Union

import numpy as np
from lazy_imports import try_import

from ..dataflow import DataFlow
from ..datasets.info import DatasetCategories
from ..mapper.cats import re_assign_cat_ids
from ..mapper.cocostruct import image_to_coco
from ..utils.file_utils import Requirement, cocotools_available, get_cocotools_requirement
from ..utils.types import JsonDict, MetricResults
from .base import MetricBase
from .registry import metric_registry

with try_import() as cc_import_guard:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

__all__ = ["CocoMetric"]


_COCOEVAL_DEFAULTS = [
    {"ap": 1, "iouThr": "", "areaRng": "all", "maxDets": 100},
    {"ap": 1, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
    {"ap": 1, "iouThr": 0.75, "areaRng": "all", "maxDets": 100},
    {"ap": 1, "iouThr": "", "areaRng": "small", "maxDets": 100},
    {"ap": 1, "iouThr": "", "areaRng": "medium", "maxDets": 100},
    {"ap": 1, "iouThr": "", "areaRng": "large", "maxDets": 100},
    {"ap": 0, "iouThr": "", "areaRng": "all", "maxDets": 1},
    {"ap": 0, "iouThr": "", "areaRng": "all", "maxDets": 10},
    {"ap": 0, "iouThr": "", "areaRng": "all", "maxDets": 100},
    {"ap": 0, "iouThr": "", "areaRng": "small", "maxDets": 100},
    {"ap": 0, "iouThr": "", "areaRng": "medium", "maxDets": 100},
    {"ap": 0, "iouThr": "", "areaRng": "large", "maxDets": 100},
]

_F1_DEFAULTS = [
    {"ap": 1, "iouThr": 0.9, "areaRng": "all", "maxDets": 100},
    {"ap": 0, "iouThr": 0.9, "areaRng": "all", "maxDets": 100},
]

_MAX_DET_INDEX = [2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2]

"""
Taken from <https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py>
"""


def _summarize(  # type: ignore
    self, ap: int = 1, iouThr: float = 0.9, areaRng: str = "all", maxDets: int = 100, per_category: bool = False
) -> Union[float, list[float]]:
    # pylint: disable=C0103
    p = self.params
    iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
    titleStr = "Average Precision" if ap == 1 else "Average Recall"
    typeStr = "(AP)" if ap == 1 else "(AR)"
    iouStr = (
        "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])  # pylint: disable=C0209
        if iouThr is None
        else "{:0.2f}".format(iouThr)  # pylint: disable=C0209
    )

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if per_category:
        if ap == 1:
            s = self.eval["precision"]
            num_classes = s.shape[2]
            results_per_class = []
            for idx in range(num_classes):
                if iouThr is not None:
                    s = self.eval["precision"]
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                precision = s[:, :, idx, aind, mind]
                precision = precision[precision > -1]
                res = np.mean(precision) if precision.size else float("nan")
                results_per_class.append(float(res))
                print(f"Precision for class {idx+1}: @[ IoU={iouStr} | area={areaRng} | maxDets={maxDets} ] = {res}")
        else:
            s = self.eval["recall"]
            num_classes = s.shape[1]
            results_per_class = []
            for idx in range(num_classes):
                if iouThr is not None:
                    s = self.eval["recall"]
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                recall = s[:, idx, aind, mind]
                recall = recall[recall > -1]
                res = np.mean(recall) if recall.size else float("nan")
                results_per_class.append(float(res))
                print(f"Recall for class {idx+1}: @[ IoU={iouStr} | area={areaRng} | maxDets={maxDets} ] = {res}")
        return results_per_class
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = self.eval["precision"]
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = self.eval["recall"]
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


if cocotools_available():
    COCOeval.summarize_f1 = _summarize


@metric_registry.register("coco")
class CocoMetric(MetricBase):
    """
    Metric induced by `pycocotools.cocoeval.COCOeval`.
    """

    name = "mAP and mAR"
    metric = COCOeval if cocotools_available() else None
    mapper = image_to_coco
    _f1_score = None
    _f1_iou = None
    _per_category = False
    _params: dict[str, Union[list[int], list[list[int]]]] = {}

    @classmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> tuple[COCO, COCO]:
        cats = [{"id": int(k), "name": v} for k, v in categories.get_categories(as_dict=True, filtered=True).items()]
        imgs_gt, imgs_pr = [], []
        anns_gt, anns_pr = [], []

        dataflow_gt.reset_state()
        dataflow_predictions.reset_state()

        for dp_gt, dp_pred in zip(dataflow_gt, dataflow_predictions):
            img_gt, ann_gt = cls.mapper(dp_gt)
            dp_pred = re_assign_cat_ids(categories.get_categories(as_dict=True, filtered=True, name_as_key=True))(
                dp_pred
            )
            img_pr, ann_pr = cls.mapper(dp_pred)
            imgs_gt.append(img_gt)
            imgs_pr.append(img_pr)
            anns_gt.extend(ann_gt)
            anns_pr.extend(ann_pr)

        dataset_gt = {"images": imgs_gt, "annotations": anns_gt, "categories": cats}
        dataset_pr = {"images": imgs_pr, "annotations": anns_pr, "categories": cats}

        coco_gt = COCO()
        coco_gt.dataset = dataset_gt
        coco_gt.createIndex()
        coco_predictions = COCO()
        coco_predictions.dataset = dataset_pr
        coco_predictions.createIndex()

        return coco_gt, coco_predictions

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        coco_gt, coco_predictions = cls.dump(dataflow_gt, dataflow_predictions, categories)

        metric = cls.metric(coco_gt, coco_predictions, iouType="bbox")
        if cls._params:
            for key, value in cls._params.items():
                setattr(metric.params, key, value)

        metric.evaluate()
        metric.accumulate()

        if cls._f1_score:
            summary_bbox = [
                metric.summarize_f1(1, cls._f1_iou, maxDets=metric.params.maxDets[2], per_category=cls._per_category),
                metric.summarize_f1(0, cls._f1_iou, maxDets=metric.params.maxDets[2], per_category=cls._per_category),
            ]
        else:
            metric.summarize()
            summary_bbox = metric.stats

        results = []

        default_parameters = cls.get_summary_default_parameters()
        if cls._per_category:
            default_parameters = default_parameters * len(summary_bbox[0])
            summary_bbox = [item for pair in zip(*summary_bbox) for item in pair]
        val = 0
        for idx, (params, value) in enumerate(zip(default_parameters, summary_bbox)):
            params = copy(params)
            params["mode"] = "bbox"
            params["val"] = value
            if cls._per_category:
                if idx % 2 == 0:
                    val += 1
                params["category_id"] = val
            results.append(params)

        return results

    @classmethod
    def get_summary_default_parameters(cls) -> list[JsonDict]:
        """
        Get default parameters of evaluation results. May differ from other `CocoMetric` classes.

        Returns:
            List of dict with default configuration, e.g. setting of average precision, iou threshold,
            area range and maximum detections.
        """
        if cls._f1_score:
            for el, idx in zip(_F1_DEFAULTS, [2, 2]):
                if cls._params:
                    if cls._params.get("maxDets") is not None:
                        el["maxDets"] = cls._params["maxDets"][idx]
                el["iouThr"] = cls._f1_iou
            return _F1_DEFAULTS

        for el, idx in zip(_COCOEVAL_DEFAULTS, _MAX_DET_INDEX):
            if cls._params:
                if cls._params.get("maxDets") is not None:
                    el["maxDets"] = cls._params["maxDets"][idx]
        return _COCOEVAL_DEFAULTS

    @classmethod
    def set_params(
        cls,
        max_detections: Optional[list[int]] = None,
        area_range: Optional[list[list[int]]] = None,
        f1_score: bool = False,
        f1_iou: float = 0.9,
        per_category: bool = False,
    ) -> None:
        """
        Setting params for different coco metric modes.

        Args:
            max_detections: The maximum number of detections to consider
            area_range: The area range to classify objects as `all`, `small`, `medium` and `large`
            f1_score: Will use F1-score setting with default `iouThr=0.9`. To be more precise it does not calculate
                      the F1-score but the precision and recall for a given `iou` threshold. Use the harmonic mean to
                      get the ultimate F1-score.
            f1_iou: Use with `f1_score=True` and reset the f1 iou threshold
                    per_category: Whether to calculate metrics per category
        """
        if max_detections is not None:
            assert len(max_detections) == 3, max_detections
            cls._params["maxDets"] = max_detections
        if area_range is not None:
            assert len(area_range) == 4, area_range
            cls._params["areaRng"] = area_range

        cls._f1_score = f1_score
        cls._f1_iou = f1_iou
        cls._per_category = per_category

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_cocotools_requirement()]
