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
Tree distance similarity (TEDS) metric

Taken from <https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py>
"""

import statistics
from collections import defaultdict, deque
from typing import Any, Callable, Optional

from lazy_imports import try_import

from ..dataflow import DataFlow, DataFromList, MapData, MultiThreadMapData
from ..datapoint.image import Image
from ..datapoint.view import Page
from ..datasets.base import DatasetCategories
from ..utils.file_utils import Requirement, get_apted_requirement, get_distance_requirement, get_lxml_requirement
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import LayoutType
from ..utils.types import MetricResults
from .base import MetricBase
from .registry import metric_registry

with try_import() as ap_import_guard:
    from apted import APTED, Config  # type: ignore
    from apted.helpers import Tree  # type: ignore


if not ap_import_guard.is_successful():
    from ..utils.mocks import Config, Tree


with try_import() as ds_import_guard:
    import distance  # type: ignore

with try_import() as lx_import_guard:
    from lxml import etree


class TableTree(Tree):
    """
    TableTree is derived class from `APTED.helpers.Tree`.
    """

    def __init__(  # pylint: disable=W0231
        self,
        *children: Any,
        tag: str,
        colspan: Optional[int] = None,
        rowspan: Optional[int] = None,
        content: Optional[list[str]] = None,
    ) -> None:
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

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
    `CustomConfig` for calculating `APTED` tree edit distance.
    Check APTED docs for more information
    """

    @staticmethod
    def maximum(*sequences: Any) -> int:
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences: Any) -> float:
        """Get distance from `0` to `1`"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1: Any, node2: Any) -> float:
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance similarity"""

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only
        self.__tokens__: list[str] = []

    def tokenize(self, node: TableTree) -> None:
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

    def load_html_tree(self, node: TableTree, parent: Optional[TableTree] = None) -> Optional[TableTree]:
        """Converts `HTML` tree to the format required by APTED"""
        global __tokens__  # pylint: disable = W0602
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                *deque(),
                tag=node.tag,
                colspan=int(node.attrib.get("colspan", "1")),
                rowspan=int(node.attrib.get("rowspan", "1")),
                content=cell,
            )
        else:
            new_node = TableTree(*deque(), tag=node.tag, rowspan=None, colspan=None, content=None)
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node
        return None

    def evaluate(self, inputs: tuple[str, str]) -> float:
        """
        Computes TEDS score between the prediction and the ground truth of a
        given sample

        Args:
            inputs: A tuple of ground truth and prediction in xml format

        Returns:
            A float value between 0.0 and 1.0, where 1.0 means perfect match
        """

        ground_truth, pred = inputs[0], inputs[1]
        if (not pred) or (not ground_truth):
            return 0.0
        parser = etree.XMLParser()
        try:
            ground_truth_tr = etree.XML(ground_truth, parser)
        except etree.XMLSyntaxError:
            logger.info(
                LoggingRecord(
                    "SyntaxError while xml parsing ground truth. Sample will be removed", {"xml_gt": ground_truth}
                )
            )
            return -1.0
        try:
            pred_tr = etree.XML(pred, parser)
        except etree.XMLSyntaxError:
            logger.info(
                LoggingRecord("SyntaxError while xml parsing prediction. Sample will be removed", {"xml_pr": pred})
            )
            return -1.0

        etree.strip_tags(pred_tr)
        etree.strip_tags(ground_truth_tr)
        n_nodes_pred = len(pred_tr.xpath(".//*"))  # type: ignore
        n_nodes_true = len(ground_truth_tr.xpath(".//*"))  # type: ignore
        n_nodes = max(n_nodes_pred, n_nodes_true)
        tree_pred = self.load_html_tree(pred_tr)  # type: ignore
        tree_true = self.load_html_tree(ground_truth_tr)  # type: ignore
        dist = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        if n_nodes:
            return 1.0 - (float(dist) / n_nodes)
        return 0.0


def teds_metric(gt_list: list[str], predict_list: list[str], structure_only: bool) -> tuple[float, int]:
    """
    Computes tree edit distance score (TEDS) between the prediction and the ground truth of a batch of samples. The
    approach to measure similarity of tables by means of their html representation has been advocated in
    <https://arxiv.org/abs/1911.10683>

    Args:
        gt_list: A list of ground truth samples in `xml` format
        predict_list: A list of predictions in `xml` format
        structure_only: If `True`, only the structure of the table is considered, but no text

    """
    teds = TEDS(structure_only=structure_only)

    input_list = list(zip(gt_list, predict_list))
    df: DataFlow
    df = DataFromList(input_list)
    if len(input_list) >= 2:
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
    Metric induced by `TEDS`
    """

    metric = teds_metric  # type: ignore
    mapper: Callable[[Image, LayoutType, list[LayoutType]], Page] = Page.from_image
    text_container: LayoutType = LayoutType.WORD
    floating_text_block_categories = [LayoutType.TABLE]

    structure_only = False

    @classmethod
    def dump(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> tuple[list[str], list[str]]:
        dataflow_gt.reset_state()
        dataflow_predictions.reset_state()

        # gt and predictions are not necessarily in same order. Will need to reorder
        gt_dict = defaultdict(list)
        pred_dict = defaultdict(list)
        for dp_gt, dp_pred in zip(dataflow_gt, dataflow_predictions):
            page_gt = cls.mapper(dp_gt, cls.text_container, cls.floating_text_block_categories)
            for table in page_gt.tables:
                gt_dict[page_gt.image_id].append(table.html)

            page_pred = cls.mapper(dp_pred, cls.text_container, cls.floating_text_block_categories)
            for table in page_pred.tables:
                pred_dict[page_pred.image_id].append(table.html)

        gt_list = []
        pred_list = []
        for sample in gt_dict:
            gt_list.extend(gt_dict[sample])
            pred_list.extend(pred_dict[sample])

        return gt_list, pred_list  # type: ignore

    @classmethod
    def get_distance(
        cls, dataflow_gt: DataFlow, dataflow_predictions: DataFlow, categories: DatasetCategories
    ) -> list[MetricResults]:
        html_gt_list, html_pr_list = cls.dump(dataflow_gt, dataflow_predictions, categories)

        score, num_samples = cls.metric(html_gt_list, html_pr_list, cls.structure_only)  # type: ignore
        return [{"teds_score": score, "num_samples": num_samples}]

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_apted_requirement(), get_distance_requirement(), get_lxml_requirement()]
