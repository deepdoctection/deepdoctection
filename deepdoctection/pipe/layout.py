# -*- coding: utf-8 -*-
# File: layout.py

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
Module for layout pipeline component
"""
from typing import Dict, Optional

from ..datapoint.image import Image
from ..extern.base import ObjectDetector, PdfMiner
from ..utils.detection_types import JsonDict
from .base import PredictorPipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("ImageLayoutService")
class ImageLayoutService(PredictorPipelineComponent):
    """
    Pipeline component for determining the layout. Which layout blocks are determined depends on the Detector and thus
    usually on the data set on which the Detector was pre-trained. If the Detector has been trained on Publaynet, these
    are layouts such as text, title, table, list and figure. If the Detector has been trained on DocBank, these are
    rather Abstract, Author, Caption, Equation, Figure, Footer, List, Paragraph, Reference, Section, Table, Title.

    The component is usually at the beginning of the pipeline. Cropping of the layout blocks can be selected to simplify
    further processing.

    **Example**

        .. code-block:: python

            d_items = TPFrcnnDetector(item_config_path, item_weights_path, {"1": "ROW", "2": "COLUMNS"})
            item_component = ImageLayoutService(d_items)
    """

    def __init__(
        self,
        layout_detector: ObjectDetector,
        category_id_mapping: Optional[Dict[int, int]] = None,
        to_image: bool = False,
        crop_image: bool = False,
    ):
        """
        :param layout_detector: object detector
        :param category_id_mapping: Mapping of category IDs. Usually, the category ids start with 1.
        :param to_image: Generate an image for each detected block, e.g. populate :attr:`ImageAnnotation.image`. Useful,
                         if you want to process only some blocks in a subsequent pipeline component.
        :param crop_image: Do not only populate :attr:`ImageAnnotation.image` but also crop the detected block according
                           to its bounding box and populate the resulting sub image to
                           :attr:`ImageAnnotation.image.image`.
        """
        self.to_image = to_image
        self.crop_image = crop_image
        super().__init__(layout_detector, category_id_mapping)

    def serve(self, dp: Image) -> None:
        assert dp.image is not None
        detect_result_list = self.predictor.predict(dp.image)  # type: ignore
        for detect_result in detect_result_list:
            self.dp_manager.set_image_annotation(detect_result, to_image=self.to_image, crop_image=self.crop_image)

    def get_meta_annotation(self) -> JsonDict:
        assert isinstance(self.predictor, (ObjectDetector, PdfMiner))
        return dict(
            [
                ("image_annotations", self.predictor.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )
