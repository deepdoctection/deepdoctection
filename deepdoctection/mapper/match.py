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

from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import iou
from ..datapoint.image import Image
from ..extern.tp.tpfrcnn.utils.np_box_ops import ioa as np_ioa
from ..utils.settings import TypeOrStr


def match_anns_by_intersection(
    dp: Image,
    matching_rule: Literal["iou", "ioa"],
    threshold: float,
    use_weighted_intersections: bool = False,
    parent_ann_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    child_ann_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    parent_ann_ids: Optional[Union[Sequence[str], str]] = None,
    child_ann_ids: Optional[Union[str, Sequence[str]]] = None,
    parent_ann_service_ids: Optional[Union[str, Sequence[str]]] = None,
    child_ann_service_ids: Optional[Union[str, Sequence[str]]] = None,
    max_parent_only: bool = False,
) -> tuple[Any, Any, Sequence[ImageAnnotation], Sequence[ImageAnnotation]]:
    """
    Generates an iou/ioa-matrix for `parent_ann_categories` and `child_ann_categories` and returns pairs of child/parent
    indices that are above some intersection threshold. It will also return a list of all pre-selected parent and child
    annotations.

    Example:
        ```python
        match_anns_by_intersection()
        ```

        Let `p_i, c_j` be annotation ids of parent and children according to some category names.

        | ioa   | c_1 | c_2 |
        |-------|-----|-----|
        | p_1   | 0.3 | 0.8 |
        | p_2   | 0.4 | 0.1 |
        | p_3   | 1.0 | 0.4 |

        With `ioa_threshold = 0.5` it will return:

        `[[2],[0]], [[1],[],[1]], [c_1,c_2], [p_1,p_2,p_3]`.

        For each child, the sum of all ioas with all parents sum up to 1. Hence, the ioa with one parent will in general
        decrease if one child intersects with more parents. Take two children, one matching two parents with an ioa of
        0.5 each, while the second matches four parents with an ioa of 0.25 each. In this situation, it is difficult to
        assign children according to a given threshold and one also has to take into account the number of parental
        intersections for each child.

        Note:
            Setting `use_weighted_intersections` to True will multiply each ioa with the number of intersections,
            making it easier to work with an absolute threshold.

        Note:
            In some situations, you want to assign to each child at most one parent. Setting `max_parent_only` to `True`
            will select the parent with the highest ioa. There is currently no implementation for iou.

    Args:
        dp: Image datapoint.
        matching_rule: Intersection measure type, either `iou` or `ioa`.
        threshold: Threshold for the given matching rule. Will assign every child annotation with iou/ioa above the
            threshold to the parental annotation.
        use_weighted_intersections: This is currently only implemented for matching_rule `ioa`. Instead of using
            the ioa_matrix, it will use a weighted ioa in order to take into account that intersections with more cells
            will likely decrease the ioa value. By multiplying the ioa with the number of all intersections for each
            child, this value calibrates the ioa.
        parent_ann_category_names: Single str or list of category names.
        child_ann_category_names: Single str or list of category names.
        parent_ann_ids: Additional filter condition. If some ids are selected, it will ignore all other parent
            candidates which are not in the list.
        child_ann_ids: Additional filter condition. If some ids are selected, it will ignore all other children
            candidates which are not in the list.
        parent_ann_service_ids: Additional filter condition. If some ids are selected, it will ignore all other parent
            candidates which are not in the list.
        child_ann_service_ids: Additional filter condition. If some ids are selected, it will ignore all other children
            candidates which are not in the list.
        max_parent_only: Will assign to each child at most one parent with maximum ioa.

    Returns:
        child indices, parent indices (see Example), list of parent ids and list of children ids.
    """

    assert matching_rule in ["iou", "ioa"], "matching rule must be either iou or ioa"

    child_anns = dp.get_annotation(
        annotation_ids=child_ann_ids, category_names=child_ann_category_names, service_ids=child_ann_service_ids
    )
    child_ann_boxes = np.array(
        [
            ann.get_bounding_box(dp.image_id).transform(dp.width, dp.height, absolute_coords=True).to_list(mode="xyxy")
            for ann in child_anns
        ]
    )

    parent_anns = dp.get_annotation(
        annotation_ids=parent_ann_ids, category_names=parent_ann_category_names, service_ids=parent_ann_service_ids
    )
    parent_ann_boxes = np.array(
        [
            ann.get_bounding_box(dp.image_id).transform(dp.width, dp.height, absolute_coords=True).to_list(mode="xyxy")
            for ann in parent_anns
        ]
    )

    if matching_rule in ["iou"] and parent_anns and child_anns:
        iou_matrix = iou(child_ann_boxes, parent_ann_boxes)
        output = iou_matrix > threshold
        child_index, parent_index = output.nonzero()
    elif matching_rule in ["ioa"] and parent_anns and child_anns:
        ioa_matrix = np.transpose(np_ioa(parent_ann_boxes, child_ann_boxes))

        if max_parent_only:
            # set all matrix values below threshold to 0
            ioa_matrix[ioa_matrix < threshold] = 0
            # add a dummy column to the left. argmax will choose this column if all ioa values of one child are 0.
            # This index will be ignored in output
            ioa_matrix = np.hstack([np.zeros((ioa_matrix.shape[0], 1)), ioa_matrix])
            child_index_arg_max = ioa_matrix.argmax(1)
            child_index = child_index_arg_max.nonzero()[0]
            child_index_nonzero = child_index_arg_max[child_index]
            # reduce parent index by one, as all indices have been increased by one because of the dummy column
            parent_index = child_index_nonzero - np.ones(child_index_nonzero.shape[0], dtype=np.intc)
        else:

            def _weighted_ioa_matrix(mat: NDArray[np.float32]) -> NDArray[np.float32]:
                sum_of_rows = (mat != 0).sum(1)
                multiplier = np.transpose(sum_of_rows * np.ones((mat.shape[1], mat.shape[0])))
                return multiplier * mat

            if use_weighted_intersections:
                ioa_matrix = _weighted_ioa_matrix(ioa_matrix)
            output = ioa_matrix > threshold
            child_index, parent_index = output.nonzero()

    else:
        return [], [], [], []

    return child_index, parent_index, child_anns, parent_anns


def match_anns_by_distance(
    dp: Image,
    parent_ann_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    child_ann_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    parent_ann_ids: Optional[Union[Sequence[str], str]] = None,
    child_ann_ids: Optional[Union[str, Sequence[str]]] = None,
    parent_ann_service_ids: Optional[Union[str, Sequence[str]]] = None,
    child_ann_service_ids: Optional[Union[str, Sequence[str]]] = None,
) -> list[tuple[ImageAnnotation, ImageAnnotation]]:
    """
    Generates pairs of parent and child annotations by calculating the euclidean distance between the centers of the
    parent and child bounding boxes. It will return the closest child for each parent.

    Note:
        A child can be assigned multiple times to different parents.

    Args:
        dp: Image datapoint.
        parent_ann_category_names: Single str or list of category names.
        child_ann_category_names: Single str or list of category names.
        parent_ann_ids: Additional filter condition. If some ids are selected, it will ignore all other parent
            candidates which are not in the list.
        child_ann_ids: Additional filter condition. If some ids are selected, it will ignore all other children
            candidates which are not in the list.
        parent_ann_service_ids: Additional filter condition. If some ids are selected, it will ignore all other parent
            candidates which are not in the list.
        child_ann_service_ids: Additional filter condition. If some ids are selected, it will ignore all other children
            candidates which are not in the list.

    Returns:
        List of tuples of parent and child annotations.
    """

    parent_anns = dp.get_annotation(
        annotation_ids=parent_ann_ids, category_names=parent_ann_category_names, service_ids=parent_ann_service_ids
    )
    child_anns = dp.get_annotation(
        annotation_ids=child_ann_ids, category_names=child_ann_category_names, service_ids=child_ann_service_ids
    )
    child_centers = [block.get_bounding_box(dp.image_id).center for block in child_anns]
    parent_centers = [block.get_bounding_box(dp.image_id).center for block in parent_anns]
    if child_centers and parent_centers:
        child_indices = distance.cdist(parent_centers, child_centers).argmin(axis=1)
        return [(parent_anns[i], child_anns[j]) for i, j in enumerate(child_indices)]
    return []
