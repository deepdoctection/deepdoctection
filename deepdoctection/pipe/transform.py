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
from ..utils.detection_types import JsonDict
from ..extern.base import ImageTransformer
from .base import ImageTransformPipelineComponent


class SimpleTransformService(ImageTransformPipelineComponent):

    def __init__(self, transform_predictor: ImageTransformer):
        super().__init__(self._get_name(transform_predictor.name), transform_predictor)

    def serve(self, dp: Image) -> None:
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
