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

from ..datapoint.image import Image
from ..extern.base import ImageTransformer
from ..utils.detection_types import JsonDict
from ..utils.logger import logger
from .base import ImageTransformPipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("SimpleTransformService")
class SimpleTransformService(ImageTransformPipelineComponent):
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
        super().__init__(self._get_name(transform_predictor.name), transform_predictor)

    def serve(self, dp: Image) -> None:
        if dp.annotations:
            logger.warning(
                "%s has already received image with image annotations. These annotations will not "
                "be transformed and might cause unexpected output in your pipeline.",
                self.name,
            )
        if dp.image is not None:
            np_image_transform = self.transform_predictor.transform(dp.image)
            self.dp_manager.datapoint.clear_image(True)
            self.dp_manager.datapoint.image = np_image_transform

    def clone(self) -> "SimpleTransformService":
        return self.__class__(self.transform_predictor)

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )

    @staticmethod
    def _get_name(transform_name: str) -> str:
        return f"simple_transform_{transform_name}"
