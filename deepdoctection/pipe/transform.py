# -*- coding: utf-8 -*-
# File: transform.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Module for transform style pipeline components. These pipeline components are used for various transforming operations
on images (e.g. deskew, de-noising or more general GAN like operations.
"""

from __future__ import annotations

from .. import DetectionResult
from ..datapoint.image import Image
from ..extern.base import ImageTransformer
from .base import MetaAnnotation, PipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("SimpleTransformService")
class SimpleTransformService(PipelineComponent):
    """
    Pipeline component for transforming an image. The service is designed for applying transform predictors that
    take an image as numpy array as input and return the same. The service itself will change the underlying metadata
    like height and width of the returned transform.

    This component is meant to be used at the very first stage of a pipeline. If components have already returned image
    annotations then this component will currently not re-calculate bounding boxes in terms of the transformed image.
    It will raise a warning (at runtime) if image annotations have already been appended.
    """

    def __init__(self, transform_predictor: ImageTransformer):
        """

        :param transform_predictor: image transformer
        """
        self.transform_predictor = transform_predictor
        super().__init__(self._get_name(transform_predictor.name), self.transform_predictor.model_id)

    def serve(self, dp: Image) -> None:
        if dp.image is not None:
            detection_result = self.transform_predictor.predict(dp.image)
            transformed_image = self.transform_predictor.transform_image(dp.image, detection_result)
            self.dp_manager.datapoint.clear_image(True)
            self.dp_manager.datapoint.image = transformed_image
            self.dp_manager.set_summary_annotation(
                summary_key=self.transform_predictor.get_category_names()[0],
                summary_name=self.transform_predictor.get_category_names()[0],
                summary_number=None,
                summary_value=getattr(detection_result, self.transform_predictor.get_category_names()[0].value, None),
                summary_score=detection_result.score,
            )
            detect_results = []
            for ann in dp.get_annotation():
                detect_results.append(DetectionResult(box=ann.get_bounding_box().to_list(mode="xyxy"),
                                                      class_name=ann.category_name,
                                                      score=ann.score,
                                                      class_id=ann.category_id,
                                                      uuid= ann.annotation_id))
            output_detect_results = self.transform_predictor.transform_coords(detect_results)

            for detect_result in output_detect_results:
                ann = dp.get_annotation(annotation_ids=detect_result.uuid)[0]
                transformed_ann_id = self.dp_manager.set_image_annotation(detect_result)
                ann.deactivate()
                transformed_ann = self.dp_manager.get_image_annotation(annotation_ids=transformed_ann_id)[0]
                for key, sub_ann in ann.sub_categories.items():
                    transformed_ann.dump_sub_category(key, sub_ann)

                #dp.add_annotation(detect_result.box, detect_result.class_name, detect_result.score, detect_result.class_id, detect_result.uuid)

    def clone(self) -> SimpleTransformService:
        return self.__class__(self.transform_predictor)

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(
            image_annotations=(),
            sub_categories={},
            relationships={},
            summaries=self.transform_predictor.get_category_names(),
        )

    @staticmethod
    def _get_name(transform_name: str) -> str:
        return f"simple_transform_{transform_name}"

    def clear_predictor(self) -> None:
        pass
