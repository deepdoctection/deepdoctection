# -*- coding: utf-8 -*-
# File: anngen.py

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
Module for datapoint populating helpers
"""

from typing import Dict, Optional, Union

from ..datapoint.annotation import CategoryAnnotation, ContainerAnnotation, ImageAnnotation, SummaryAnnotation
from ..datapoint.box import BoundingBox, local_to_global_coords, rescale_coords
from ..datapoint.image import Image
from ..extern.base import DetectionResult
from ..mapper.maputils import MappingContextManager
from ..utils.settings import names


class DatapointManager:
    """
    Class whose methods provide an API for manipulating image datapoints. This includes the creation and storage of
    annotations in the cache of the image but also in the annotations themselves.

    When the image is transferred, the annotations are stored in a cache dictionary so that access via the annotation ID
    can be performed efficiently.

    The manager is part of each PipelineComponent.
    """

    def __init__(self, category_id_mapping: Optional[Dict[int, int]]) -> None:
        """
        :param category_id_mapping: Reassignment of category ids. Handover via dict
        """
        self._datapoint: Optional[Image] = None
        self._cache_anns: Dict[str, ImageAnnotation] = {}
        self.datapoint_is_passed: bool = False
        self.category_id_mapping: Optional[Dict[int, int]] = category_id_mapping

    @property
    def datapoint(self) -> Image:
        """
        datapoint
        """
        if self._datapoint is not None:
            return self._datapoint
        raise AssertionError("no datapoint passed")

    @datapoint.setter
    def datapoint(self, dp: Image) -> None:
        """
        datapoint
        """
        self._datapoint = dp
        self._cache_anns = {ann.annotation_id: ann for ann in dp.get_annotation()}
        self.datapoint_is_passed = True

    def assert_datapoint_passed(self) -> None:
        """
        assert that datapoint is passed
        """
        assert self.datapoint_is_passed, "Pass datapoint to  DatapointManager before creating anns"

    def maybe_map_category_id(self, category_id: Union[str, int]) -> int:
        """
        Maps categories if a category id mapping is provided in :meth:`__init__`.

        :param category_id: category id via integer or string.
        :return: mapped category id
        """
        if self.category_id_mapping is None:
            return int(category_id)
        return self.category_id_mapping[int(category_id)]

    def set_image_annotation(
        self,
        detect_result: DetectionResult,
        to_annotation_id: Optional[str] = None,
        to_image: bool = False,
        crop_image: bool = False,
        detect_result_max_width: Optional[float] = None,
        detect_result_max_height: Optional[float] = None,
    ) -> Optional[str]:
        """
        Creating an image annotation from a raw DetectionResult dataclass. Beside dumping the annotation to the Image
        annotation cache you can also dump the annotation to the :attr:`image` of an annotation with given
        annotation_id. This is handy if, you know, you want to send the sub image to a subsequent pipeline component.

        Moreover, it is possible to generate an Image of the given raw annotation and store it in its :attr:`image`. The
        resulting image is given as a sub image of :attr:`self` defined by it bounding box coordinates. Use crop_image
        to explicitly store the sub image as numpy array.

        :param detect_result: A :class:`DetectionResult` in general coming from ObjectDetector
        :param to_annotation_id: Will dump the created image annotation to :attr:`image` of the given annotation_id.
                                 Requires the to_annotation to have a not None image.
        :param to_image: If True will populate :attr:`image`.
        :param crop_image: Makes only sense if to_image=True and if a numpy array is stored in the original image.
                           Will generate :attr:`Image.image`.
        :param detect_result_max_width: If detect result has a different scaling scheme from the image it refers to,
                                        pass the max width possible so coords can be rescaled.
        :param detect_result_max_height: If detect result has a different scaling scheme from the image it refers to,
                                        pass the max height possible so coords can be rescaled.
        :return: the annotation_id of the generated image annotation
        """
        self.assert_datapoint_passed()
        detect_result.class_id = self.maybe_map_category_id(detect_result.class_id)
        with MappingContextManager(dp_name=str(detect_result)) as annotation_context:
            box = BoundingBox(
                ulx=detect_result.box[0],
                uly=detect_result.box[1],
                lrx=detect_result.box[2],
                lry=detect_result.box[3],
                absolute_coords=True,
            )
            if detect_result_max_width and detect_result_max_height:
                box = rescale_coords(
                    box,
                    detect_result_max_width,
                    detect_result_max_height,
                    self.datapoint.width,
                    self.datapoint.height,
                )
            ann = ImageAnnotation(
                category_name=detect_result.class_name,
                bounding_box=box,
                category_id=str(detect_result.class_id),
                score=detect_result.score,
            )
            if to_annotation_id is not None:
                parent_ann = self._cache_anns[to_annotation_id]
                parent_ann.image.dump(ann)  # type: ignore
                parent_ann.image.image_ann_to_image(ann.annotation_id)  # type: ignore
                ann_global_box = local_to_global_coords(
                    ann.bounding_box, parent_ann.image.get_embedding(self._datapoint.image_id)  # type: ignore
                )
                ann.image.set_embedding(parent_ann.annotation_id, ann.bounding_box)  # type: ignore
                ann.image.set_embedding(self._datapoint.image_id, ann_global_box)  # type: ignore
                parent_ann.dump_relationship(names.C.CHILD, ann.annotation_id)

            self._datapoint.dump(ann)  # type: ignore
            self._cache_anns[ann.annotation_id] = ann

            if to_image and to_annotation_id is None:
                self._datapoint.image_ann_to_image(  # type: ignore
                    annotation_id=ann.annotation_id, crop_image=crop_image
                )

        if annotation_context.context_error:
            return None
        return ann.annotation_id

    def set_category_annotation(
        self,
        category_name: str,
        category_id: Optional[Union[str, int]],
        sub_cat_key: str,
        annotation_id: str,
        score: Optional[float] = None,
    ) -> str:
        """
        Create a category annotation and dump it as sub category to an already created annotation.

        :param category_name: category name
        :param category_id: category id
        :param sub_cat_key: the key to dump the created annotation to.
        :param annotation_id: id, of the parent annotation. Currently, this can only be an image annotation.
        :param score: Add a score.
        :return: the annotation_id of the generated category annotation
        """
        self.assert_datapoint_passed()
        cat_ann = CategoryAnnotation(category_name=category_name, category_id=category_id, score=score)  # type: ignore
        self._cache_anns[annotation_id].dump_sub_category(sub_cat_key, cat_ann)
        return cat_ann.annotation_id

    def set_container_annotation(
        self,
        category_name: str,
        category_id: Optional[Union[str, int]],
        sub_cat_key: str,
        annotation_id: str,
        value: str,
        score: Optional[float] = None,
    ) -> str:
        """
        Create a container annotation and dump it as sub category to an already created annotation.

        :param category_name: category name
        :param category_id: category id
        :param sub_cat_key: the key to dump the created annotation to.
        :param annotation_id: id, of the parent annotation. Currently, this can only be an image annotation.
        :param value: A value to store
        :param score: Add a score.
        :return: annotation_id of the generated container annotation
        """
        self.assert_datapoint_passed()
        cont_ann = ContainerAnnotation(
            category_name=category_name, category_id=category_id, value=value, score=score  # type: ignore
        )
        self._cache_anns[annotation_id].dump_sub_category(sub_cat_key, cont_ann)
        return cont_ann.annotation_id

    def set_summary_annotation(
        self, summary_name: str, summary_number: int, annotation_id: Optional[str] = None
    ) -> str:
        """
        Creates a sub category of a summary annotation. If a summary of the given annotation_id does not exist, it will
        create a new one.
        :param summary_name: will create the summary name as category name and store the generated annotation under the
        same key name
        :param summary_number: will store the value in category_id.
        :param annotation_id: id of the parent annotation. Note, that the parent annotation must have :attr:`image` to
        be not None.
        :return: annotation_id of the generated category annotation
        """
        self.assert_datapoint_passed()
        if annotation_id is not None:
            image = self._cache_anns[annotation_id].image
        else:
            image = self._datapoint
        assert image is not None
        if image.summary is None:
            image.summary = SummaryAnnotation()
        cat_ann = CategoryAnnotation(category_name=summary_name, category_id=summary_number)
        image.summary.dump_sub_category(summary_name, cat_ann, image.image_id)
        return cat_ann.annotation_id
