# -*- coding: utf-8 -*-
# File: tedsmetric.py

# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

"""
Tree distance similarity metric taken from https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py
"""

import statistics
from collections import defaultdict, deque
from typing import List, Optional, Tuple

from ..dataflow import DataFlow, DataFromList, MapData, MultiThreadMapData
from ..datasets.base import DatasetCategories
from ..mapper.pagestruct import to_page
from ..utils.detection_types import JsonDict
from ..utils.file_utils import (
    Requirement,
    apted_available,
    distance_available,
    get_apted_requirement,
    get_distance_requirement,
    get_lxml_requirement,
    lxml_available,
)
from ..utils.logger import logger
from ..utils.settings import names
from .base import MetricBase
from .registry import metric_registry

if distance_available() and lxml_available() and apted_available():
    import distance
    from apted import APTED, Config
    from apted.helpers import Tree
    from lxml import etree


class TableTree(Tree):
    """
    TableTree is derived class from :class:`APTED.helpers.Tree`.
    """

    def __init__(
        self,
        *children: str,
        tag: str,
        colspan: Optional[int] = None,
        rowspan: Optional[int] = None,
        content: List[str] = None,
    ) -> None:
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)
        super().__init__("", *children)

    def bracket(self) -> str:
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = f'"tag": {self.tag}, "colspan": {self.colspan}, "rowspan": {self.rowspan}, "text": {self.content}'
        else:
            result = f'"tag": {self.tag}'
        for child in self.children:
            result += child.bracket()
        return f"{{{result}}}"


class CustomConfig(Config):
    """
    CustomConfig for calculating APTED tree edit distance. Check APTED docs for more information
    """

    @staticmethod
    def maximum(*sequences) -> int:
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences) -> float:
        """Get distance from 0 to 1"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2) -> float:
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance basead Similarity"""

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append(f"<{node.tag}>")
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        global __tokens__   # pylint: disable = W0602
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(*deque(),
                node.tag, int(node.attrib.get("colspan", "1")), int(node.attrib.get("rowspan", "1")), cell
            )
        else:
            new_node = TableTree(*deque(), node.tag, None, None, None)
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, inputs):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        ground_truth, pred, file_name = inputs[0], inputs[1], inputs[2]
        if (not pred) or (not ground_truth):
            return 0.0
        parser = etree.XMLParser()
        try:
            ground_truth = etree.XML(ground_truth, parser)
            pred = etree.XML(pred, parser)
        except etree.XMLSyntaxError:
            logger.info("Error while xml parsing for %s. Will be removed", file_name)
            return -1.0

        etree.strip_tags(pred)
        etree.strip_tags(ground_truth)
        n_nodes_pred = len(pred.xpath(".//*"))
        n_nodes_true = len(ground_truth.xpath(".//*"))
        n_nodes = max(n_nodes_pred, n_nodes_true)
        tree_pred = self.load_html_tree(pred)
        tree_true = self.load_html_tree(ground_truth)
        dist = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        if n_nodes:
            return 1.0 - (float(dist) / n_nodes)
        return 0.0


def teds_metric(gt_list: List[str], predict_list: List[str], file_name_list: List[str], structure_only: bool):
    """
    Computes tree edit distance score (TEDS) between the prediction and the ground truth of a batch of samples. The
    approach to measure similarity of tables by means of their html representation has been adovacated in
    https://arxiv.org/abs/1911.10683 .

    """
    teds = TEDS(structure_only=structure_only)

    input_list = list(zip(gt_list, predict_list, file_name_list))
    df = DataFromList(input_list)
    if len(input_list)>=2:
        df = MultiThreadMapData(df, 2, teds.evaluate, strict=True)
    else:
        df = MapData(df, teds.evaluate)
    scores = []
    df.reset_state()

    for dp in df:
        if dp != -1.0:
            scores.append(dp)

    return statistics.fmean(scores), len(scores)


@metric_registry.register("teds")
class TedsMetric(MetricBase):
    """
    Metric induced by :func:`teds`
    """

    metric = teds_metric
    mapper = to_page
    structure_only = False

    @classmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> Tuple[List[str], List[str], List[str]]:

        dataflow_gt.reset_state()
        dataflow_predictions.reset_state()

        # gt and predictions are not necessarily in same order. Will need to reorder
        gt_dict = defaultdict(list)
        pred_dict = defaultdict(list)
        for dp_gt, dp_pred in zip(dataflow_gt, dataflow_predictions):
            page_gt = cls.mapper(dp_gt, names.C.WORD, None, [names.C.TAB])
            for table in page_gt.tables:
                gt_dict[page_gt.uuid].append(table.html)

            page_pred = cls.mapper(dp_pred, names.C.WORD, None, [names.C.TAB])
            for table in page_pred.tables:
                pred_dict[page_pred.uuid].append(table.html)

        gt_list = []
        pred_list = []
        file_name_list = []
        for sample in gt_dict:
            gt_list.extend(gt_dict[sample])
            pred_list.extend(pred_dict[sample])
            file_name_list.append(sample)

        return gt_list, pred_list, file_name_list

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> List[JsonDict]:
        html_gt_list, html_pr_list, file_name_list = cls.dump(dataflow_gt, dataflow_predictions, categories)

        score, num_samples = cls.metric(html_gt_list, html_pr_list, file_name_list, cls.structure_only)
        return [{"teds_score": score, "num_samples": num_samples}]

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_apted_requirement(), get_distance_requirement(), get_lxml_requirement()]
