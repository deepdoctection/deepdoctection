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
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Union

import numpy as np

from ..datapoint.annotation import CategoryAnnotation, ContainerAnnotation, ImageAnnotation, SummaryAnnotation
from ..datapoint.box import BoundingBox, local_to_global_coords, rescale_coords
from ..datapoint.image import Image
from ..extern.base import DetectionResult
from ..mapper.maputils import MappingContextManager
from ..utils.settings import ObjectTypes, Relationships


class DatapointManager:
    """
    Class whose methods provide an API for manipulating image datapoints. This includes the creation and storage of
    annotations in the cache of the image but also in the annotations themselves.

    When the image is transferred, the annotations are stored in a cache dictionary so that access via the annotation ID
    can be performed efficiently.

    The manager is part of each `PipelineComponent`.
    """

    def __init__(self) -> None:
        self._datapoint: Optional[Image] = None
        self._cache_anns: Dict[str, ImageAnnotation] = {}
        self.datapoint_is_passed: bool = False
        self.category_id_mapping: Optional[Mapping[int, int]] = None

    @property
    def datapoint(self) -> Image:
        """
        datapoint
        """
        if self._datapoint is not None:
            return self._datapoint
        raise ValueError("no datapoint passed")

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
        assert self.datapoint_is_passed, "Pass datapoint to DatapointManager before creating anns"

    def maybe_map_category_id(self, category_id: Union[str, int]) -> int:
        """
        Maps categories if a category id mapping is provided in `__init__`.

        :param category_id: category id via integer or string.
        :return: mapped category id
        """
        if self.category_id_mapping is None:
            return int(category_id)
        return self.category_id_mapping[int(category_id)]

    def set_category_id_mapping(self, category_id_mapping: Mapping[int, int]) -> None:
        """
        In many cases the category ids sent back from a model have to be modified. Pass a mapping from model
        category ids to target annotation category ids.

        :param category_id_mapping: A mapping of model category ids (sent from DetectionResult) to category ids (saved
                                    in annotations)
        """
        self.category_id_mapping = category_id_mapping

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
        Creating an image annotation from a raw `DetectionResult` dataclass. Beside dumping the annotation to the
        `ImageAnnotation` cache you can also dump the annotation to the `image` of an annotation with given
        `annotation_id`. This is handy if, you know, you want to send the sub image to a subsequent pipeline component.

        Moreover, it is possible to generate an Image of the given raw annotation and store it in its `image`. The
        resulting image is given as a sub image of `self` defined by it bounding box coordinates. Use `crop_image`
        to explicitly store the sub image as numpy array.

        :param detect_result: A `DetectionResult` in general coming from ObjectDetector
        :param to_annotation_id: Will dump the created image annotation to `image` of the given annotation_id.
                                 Requires the to_annotation to have a not `None` image.
        :param to_image: If True will populate `image`.
        :param crop_image: Makes only sense if to_image=True and if a numpy array is stored in the original image.
                           Will generate `Image.image`.
        :param detect_result_max_width: If detect result has a different scaling scheme from the image it refers to,
                                        pass the max width possible so coords can be rescaled.
        :param detect_result_max_height: If detect result has a different scaling scheme from the image it refers to,
                                        pass the max height possible so coords can be rescaled.
        :return: the annotation_id of the generated image annotation
        """
        self.assert_datapoint_passed()
        if detect_result.class_id is None:
            raise ValueError("class_id of detect_result cannot be None")
        if not isinstance(detect_result.box, (list, np.ndarray)):
            raise TypeError(
                f"detect_result.box must be of type list or np.ndarray, but is of type {(type(detect_result.box))}"
            )
        detect_result.class_id = self.maybe_map_category_id(detect_result.class_id)
        with MappingContextManager(
            dp_name=self.datapoint.file_name, filter_level="annotation", detect_result=asdict(detect_result)
        ) as annotation_context:
            box = BoundingBox(
                ulx=detect_result.box[0],
                uly=detect_result.box[1],
                lrx=detect_result.box[2],
                lry=detect_result.box[3],
                absolute_coords=detect_result.absolute_coords,
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
                if parent_ann.image is None:
                    raise ValueError("image cannot be None")
                parent_ann.image.dump(ann)
                parent_ann.image.image_ann_to_image(ann.annotation_id)
                ann_global_box = local_to_global_coords(
                    ann.bounding_box, parent_ann.get_bounding_box(self.datapoint.image_id)  # type: ignore
                )
                if ann.image is None:
                    raise ValueError("image cannot be None")
                ann.image.set_embedding(parent_ann.annotation_id, ann.bounding_box)
                ann.image.set_embedding(self.datapoint.image_id, ann_global_box)
                parent_ann.dump_relationship(Relationships.child, ann.annotation_id)

            self.datapoint.dump(ann)
            self._cache_anns[ann.annotation_id] = ann

            if to_image and to_annotation_id is None:
                self.datapoint.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=crop_image)

        if annotation_context.context_error:
            return None
        return ann.annotation_id

    def set_category_annotation(
        self,
        category_name: ObjectTypes,
        category_id: Optional[Union[str, int]],
        sub_cat_key: ObjectTypes,
        annotation_id: str,
        score: Optional[float] = None,
    ) -> Optional[str]:
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
        with MappingContextManager(
            dp_name=self.datapoint.file_name,
            filter_level="annotation",
            category_annotation={
                "category_name": category_name.value,
                "sub_cat_key": sub_cat_key.value,
                "annotation_id": annotation_id,
            },
        ) as annotation_context:
            cat_ann = CategoryAnnotation(category_name=category_name, category_id=str(category_id), score=score)
            self._cache_anns[annotation_id].dump_sub_category(sub_cat_key, cat_ann)
        if annotation_context.context_error:
            return None
        return cat_ann.annotation_id

    def set_container_annotation(
        self,
        category_name: ObjectTypes,
        category_id: Optional[Union[str, int]],
        sub_cat_key: ObjectTypes,
        annotation_id: str,
        value: Union[str, List[str]],
        score: Optional[float] = None,
    ) -> Optional[str]:
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
        with MappingContextManager(
            dp_name=self.datapoint.file_name,
            filter_level="annotation",
            container_annotation={
                "category_name": category_name.value,
                "sub_cat_key": sub_cat_key.value,
                "annotation_id": annotation_id,
                "value": str(value),
            },
        ) as annotation_context:
            cont_ann = ContainerAnnotation(
                category_name=category_name, category_id=str(category_id), value=value, score=score
            )
            self._cache_anns[annotation_id].dump_sub_category(sub_cat_key, cont_ann)
        if annotation_context.context_error:
            return None
        return cont_ann.annotation_id

    def set_summary_annotation(
        self,
        summary_key: ObjectTypes,
        summary_name: ObjectTypes,
        summary_number: int,
        summary_value: Optional[str] = None,
        summary_score: Optional[float] = None,
        annotation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Creates a sub category of a summary annotation. If a summary of the given `annotation_id` does not exist, it
        will create a new one.
        :param summary_key: will store the category annotation as sub category
        :param summary_name: will create the summary name as category name
        :param summary_number: will store the value in category_id.
        :param summary_value: will create a ContainerAnnotation and store the corresponding value
        :param summary_score: will store the score
        :param annotation_id: id of the parent annotation. Note, that the parent annotation must have `image` to
        be not None.
        :return: `annotation_id` of the generated category annotation
        """
        self.assert_datapoint_passed()
        if annotation_id is not None:
            image = self._cache_anns[annotation_id].image
        else:
            image = self.datapoint
        assert image is not None, image
        if image.summary is None:
            image.summary = SummaryAnnotation()

        ann: Union[CategoryAnnotation, ContainerAnnotation]
        with MappingContextManager(
            dp_name=annotation_id,
            filter_level="annotation",
            summary_annotation={
                "summary_key": summary_key.value,
                "summary_name": summary_name.value,
                "summary_value": summary_value,
                "annotation_id": annotation_id,
            },
        ) as annotation_context:
            if summary_value:
                ann = ContainerAnnotation(
                    category_name=summary_name,
                    category_id=str(summary_number),
                    value=summary_value,
                    score=summary_score,
                )
            else:
                ann = CategoryAnnotation(
                    category_name=summary_name, category_id=str(summary_number), score=summary_score
                )
            image.summary.dump_sub_category(summary_key, ann, image.image_id)

        if annotation_context.context_error:
            return None
        return ann.annotation_id

    def deactivate_annotation(self, annotation_id: str) -> None:
        """
        Deactivate annotation by given annotation_id

        :param annotation_id: annotation_id
        """
        ann = self._cache_anns[annotation_id]
        ann.deactivate()

    def get_annotation(self, annotation_id: str) -> ImageAnnotation:
        """get single `ImageAnnotation`"""
        return self._cache_anns[annotation_id]
    
    # TODO: Check with Janis, do we need to apply the @setter decorator?
    # TODO: Make this more generic for other types of annotations
    def update_annotation(self, annotation_id: str, new_value: Union[str, List[str], None] = None, new_score: Optional[float] = None, sub_category_key: Optional[ObjectTypes] = None):
        """
        Updates the specified annotation or its sub-category with a new value and/or score.

        Parameters:
        - annotation_id (str): The ID of the annotation to update.
        - new_value (Union[str, List[str], None]): The new value to set for the annotation, if applicable.
        - new_score (Optional[float]): The new score to set for the annotation.
        - sub_category_key (Optional[ObjectTypes]): The key of the sub-category to update.

        Raises:
        - ValueError: If the annotation_id does not exist within the current datapoint.
        """
        print(f"DEBUG: Entering update_annotation. Annotation ID: {annotation_id}, New Value: {new_value}, New Score: {new_score}, Sub-category Key: {sub_category_key}")

        self.assert_datapoint_passed()

        annotation = self._cache_anns.get(annotation_id)
        if not annotation:
            raise ValueError(f"Annotation with ID {annotation_id} not found.")

        # Handle updating sub-categories if a key is provided
        if sub_category_key:
            sub_category = annotation.sub_categories.get(sub_category_key)
            if not sub_category:
                raise ValueError(f"Sub-category with key {sub_category_key} not found in annotation {annotation_id}.")
            if new_value is not None and hasattr(sub_category, 'value'):
                sub_category.value = new_value
            if new_score is not None and hasattr(sub_category, 'score'):
                sub_category.score = new_score
        else:
            # Update the main annotation if no sub-category key is provided
            if new_value is not None and hasattr(annotation, 'value'):
                annotation.value = new_value
            if new_score is not None and hasattr(annotation, 'score'):
                annotation.score = new_score

        # Reflect changes in the cache and datapoint's annotation list
        self._cache_anns[annotation_id] = annotation
        self.datapoint.annotations = [ann if ann.annotation_id != annotation_id else annotation for ann in self.datapoint.annotations]