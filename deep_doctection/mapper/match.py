# -*- coding: utf-8 -*-
# File: match.py

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
Module for matching detections according to various matching rules
"""

from typing import List, Optional, Union, Tuple, Any
import numpy as np

from ..datapoint.image import Image
from ..datapoint.annotation import ImageAnnotation
from ..extern.tp.tpfrcnn.common import np_iou
from ..extern.tp.tpfrcnn.utils.np_box_ops import ioa as np_ioa


def match_anns_by_intersection(
    dp: Image,
    parent_ann_category_names: Union[str, List[str]],
    child_ann_category_names: Union[str, List[str]],
    matching_rule: str,
    iou_threshold: Optional[np.float32] = None,
    ioa_threshold: Optional[np.float32] = None,
    parent_ann_ids: Optional[Union[List[str], str]] = None,
    child_ann_ids: Optional[Union[str, List[str]]] = None,
) -> Tuple[Any, Any, List[ImageAnnotation], List[ImageAnnotation]]:
    """
    Generates an iou/ioa-matrix for parent_ann_categories and child_ann_categories.

    **Example:**

    Let p_i, c_j be annotations ids of parent and children according to some category names.

    +-------+-------+-------+
    |**ioa**|**c_1**|**c_2**|
    +-------+-------+-------+
    |**p_1**|  0.3  |  0.8  |
    +-------+-------+-------+
    |**p_2**|  0.4  |  0.1  |
    +-------+-------+-------+
    |**p_3**|  1.   |  0.4  |
    +-------+-------+-------+

    With ioa_threshold = 0.5 it will return [[2],[0]], [[1],[],[1]], [c_1,c_2], [p_1,p_2,p_3]

    :param dp: image datapoint
    :param parent_ann_category_names: single str or list of category names
    :param child_ann_category_names: single str or list of category names
    :param matching_rule: intersection measure type, either "iou" or "ioa"
    :param iou_threshold: Threshold, if iou chosen. When choosing the other rule, will do nothing.
    :param ioa_threshold: Threshold, if ioa chosen. When choosing the other rule, will do nothing.
    :param parent_ann_ids: Additional filter condition. If some ids are selected, it will ignore all other parent candi-
                           dates which are not in the list.
    :param child_ann_ids: Additional filter condition. If some ids are selected, it will ignore all other children
                          candidates which are not in the list.
    :return: child indices, parent indices (see Example), list of parent ids and list of children ids.
    """

    assert matching_rule in ["iou", "ioa"], "matching rule must be either iou or ioa"

    if matching_rule in ["iou"]:
        assert iou_threshold is not None, "matching rule iou requires iou_threshold to be passed"
    else:
        assert ioa_threshold is not None, "matching rule ioa requires iou_threshold to be passed"

    child_anns = dp.get_annotation(annotation_ids=child_ann_ids, category_names=child_ann_category_names)
    child_ann_boxes = np.array(
        [ann.image.get_embedding(dp.image_id).to_list(mode="xyxy") for ann in child_anns]  # type: ignore
    )

    parent_anns = dp.get_annotation(annotation_ids=parent_ann_ids, category_names=parent_ann_category_names)
    parent_ann_boxes = np.array(
        [ann.image.get_embedding(dp.image_id).to_list(mode="xyxy") for ann in parent_anns]  # type: ignore
    )

    if matching_rule in ["iou"] and parent_anns and child_anns:
        iou_matrix = np_iou(child_ann_boxes, parent_ann_boxes)  # type: ignore
        output = iou_matrix > iou_threshold
        child_index, parent_index = output.nonzero()
    elif matching_rule in ["ioa"] and parent_anns and child_anns:
        ioa_matrix = np.transpose(np_ioa(parent_ann_boxes, child_ann_boxes))  # type: ignore
        output = ioa_matrix > ioa_threshold
        child_index, parent_index = output.nonzero()
    else:
        return [], [], [], []

    return child_index, parent_index, child_anns, parent_anns
