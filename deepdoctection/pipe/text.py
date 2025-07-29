# -*- coding: utf-8 -*-
# File: text.py

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
Text extraction pipeline component
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence, Union

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image, MetaAnnotation
from ..extern.base import ObjectDetector, PdfMiner, TextRecognizer
from ..extern.tessocr import TesseractOcrDetector
from ..utils.error import ImageError
from ..utils.settings import ObjectTypes, PageType, TypeOrStr, WordType, get_type
from ..utils.types import PixelValues
from .base import PipelineComponent
from .registry import pipeline_component_registry

__all__ = ["TextExtractionService"]


@pipeline_component_registry.register("TextExtractionService")
class TextExtractionService(PipelineComponent):
    """
    Text extraction pipeline component.

    This component is responsible for extracting text from images or selected regions of interest (ROIs) using a
    specified detector. The detector must be able to evaluate a numpy array as an image.

    Text extraction can be performed on the entire image or on selected ROIs, which are layout components determined by
    a previously run pipeline component. ROI extraction is particularly useful when using an OCR component as the
    detector and the document has a complex structure. By transferring only the ROIs to the detector, OCR results can
    be significantly improved due to the simpler structure of the ROI compared to the entire document page.

    Text components (currently only words) are attached to the image as image annotations. A relation is assigned
    between text and ROI or between text and the entire image. When selecting ROIs, only the selected categories are
    processed. ROIs that are not selected are not presented to the detector.

    Example:
    ```python
    textract_predictor = TextractOcrDetector()
    text_extract = TextExtractionService(textract_predictor)

    pipe = DoctectionPipe([text_extract])
    df = pipe.analyze(path="path/to/document.pdf")

    for dp in df:
        ...
    """

    def __init__(
        self,
        text_extract_detector: Union[ObjectDetector, PdfMiner, TextRecognizer],
        extract_from_roi: Optional[Union[Sequence[TypeOrStr], TypeOrStr]] = None,
        run_time_ocr_language_selection: bool = False,
    ):
        """
        Args:
            text_extract_detector: The detector used for text extraction.
            extract_from_roi: One or more category names for ROI selection.
            run_time_ocr_language_selection: If True, enables runtime OCR language selection. Only available for
                                             `TesseractOcrDetector` as this framework supports multiple languages.
                                              Requires a language detection pipeline component to have run before.
                                              Selects the expert language OCR model based on the determined language.

        Raises:
            TypeError: If predicting from a cropped image and the detector is not an `ObjectDetector` or
                       `TextRecognizer`.
            TypeError: If `run_time_ocr_language_selection` is True and the detector is not a `TesseractOcrDetector`.
        """

        if extract_from_roi is None:
            extract_from_roi = []
        self.extract_from_category = (
            (get_type(extract_from_roi),)
            if isinstance(extract_from_roi, str)
            else tuple((get_type(roi_category) for roi_category in extract_from_roi))
        )

        self.predictor = text_extract_detector
        super().__init__(self._get_name(text_extract_detector.name), self.predictor.model_id)
        if self.extract_from_category:
            if not isinstance(self.predictor, (ObjectDetector, TextRecognizer)):
                raise TypeError(
                    f"Predicting from a cropped image requires to pass an ObjectDetector or "
                    f"TextRecognizer. Got {type(self.predictor)}"
                )
        if run_time_ocr_language_selection:
            if not isinstance(self.predictor, TesseractOcrDetector):
                raise TypeError("Only TesseractOcrDetector supports multiple languages")

        self.run_time_ocr_language_selection = run_time_ocr_language_selection

    def serve(self, dp: Image) -> None:
        maybe_batched_text_rois = self.get_text_rois(dp)
        for text_roi in maybe_batched_text_rois:
            ann_id = None
            if isinstance(text_roi, ImageAnnotation):
                ann_id = text_roi.annotation_id
            predictor_input = self.get_predictor_input(text_roi)
            if predictor_input is None:
                raise ValueError("predictor_input cannot be None")
            if isinstance(predictor_input, int):
                pass
            else:
                width, height = None, None
                if self.run_time_ocr_language_selection:
                    self.predictor.set_language(dp.summary.get_sub_category(PageType.LANGUAGE).value)  # type: ignore
                detect_result_list = self.predictor.predict(predictor_input)  # type: ignore
                if isinstance(self.predictor, PdfMiner):
                    width, height = self.predictor.get_width_height(predictor_input)  # type: ignore

                for detect_result in detect_result_list:
                    if isinstance(self.predictor, TextRecognizer):
                        detect_ann_id = detect_result.uuid
                    else:
                        detect_ann_id = self.dp_manager.set_image_annotation(
                            detect_result, ann_id, True, detect_result_max_width=width, detect_result_max_height=height
                        )
                    if detect_ann_id is not None:
                        self.dp_manager.set_container_annotation(
                            WordType.CHARACTERS,
                            None,
                            WordType.CHARACTERS,
                            detect_ann_id,
                            detect_result.text if detect_result.text is not None else "",
                            detect_result.score,
                        )

    def get_text_rois(self, dp: Image) -> Sequence[Union[Image, ImageAnnotation, list[ImageAnnotation]]]:
        """
        Returns image ROIs based on selected categories.

        This selection is only meaningful for specific text extractors (e.g., those performing OCR and not mining text
        from native PDFs). Performs sanity checks. If a preceding text extractor has already dumped text, and the
        predictor should not extract text as well, returns an empty list.

        Args:
            dp: The `Image` to process.

        Returns:
            A list of `ImageAnnotation` or `Image`.
        """

        if self.extract_from_category:
            if self.predictor.accepts_batch:
                return [dp.get_annotation(category_names=self.extract_from_category)]
            return dp.get_annotation(category_names=self.extract_from_category)
        return [dp]

    def get_predictor_input(
        self, text_roi: Union[Image, ImageAnnotation, list[ImageAnnotation]]
    ) -> Optional[Union[bytes, PixelValues, list[tuple[str, PixelValues]], int]]:
        """
        Returns raw input for a given `text_roi`. The input can be a numpy array or PDF bytes, depending on the chosen
        predictor.

        Args:
            text_roi: The `Image`, `ImageAnnotation`, or list of `ImageAnnotation` to process.

        Returns:
            PDF bytes, numpy array, or other predictor-specific input.

        Raises:
            ImageError: If required image data is missing or if `text_roi` is not an `Image` when required.
        """

        if isinstance(text_roi, ImageAnnotation):
            if text_roi.image is None:
                raise ImageError("text_roi.image cannot be None")
            if text_roi.image.image is None:
                raise ImageError("text_roi.image.image cannot be None")
            return text_roi.image.image
        if isinstance(self.predictor, ObjectDetector):
            if not isinstance(text_roi, Image):
                raise ImageError("text_roi must be an image")
            return text_roi.image
        if isinstance(text_roi, list):
            assert all(roi.image is not None for roi in text_roi)
            assert all(roi.image.image is not None for roi in text_roi)  # type: ignore
            return [(roi.annotation_id, roi.image.image) for roi in text_roi]  # type: ignore
        if isinstance(self.predictor, PdfMiner) and text_roi.pdf_bytes is not None:
            return text_roi.pdf_bytes
        return 1

    def get_meta_annotation(self) -> MetaAnnotation:
        sub_cat_dict: dict[ObjectTypes, dict[ObjectTypes, set[ObjectTypes]]]
        if self.extract_from_category:
            sub_cat_dict = {
                category: {WordType.CHARACTERS: {WordType.CHARACTERS}} for category in self.extract_from_category
            }
        else:
            if not isinstance(self.predictor, (ObjectDetector, PdfMiner)):
                raise TypeError(
                    f"self.predictor must be of type ObjectDetector or PdfMiner but is of type "
                    f"{type(self.predictor)}"
                )
            sub_cat_dict = {
                category: {WordType.CHARACTERS: {WordType.CHARACTERS}}
                for category in self.predictor.get_category_names()
            }
        return MetaAnnotation(
            image_annotations=self.predictor.get_category_names()
            if isinstance(self.predictor, (ObjectDetector, PdfMiner))
            else (),
            sub_categories=sub_cat_dict,
            relationships={},
            summaries=(),
        )

    @staticmethod
    def _get_name(text_detector_name: str) -> str:
        return f"text_extract_{text_detector_name}"

    def clone(self) -> TextExtractionService:
        predictor = self.predictor.clone()
        if not isinstance(predictor, (ObjectDetector, PdfMiner, TextRecognizer)):
            raise ImageError(f"predictor must be of type ObjectDetector or PdfMiner, but is of type {type(predictor)}")
        return self.__class__(
            text_extract_detector=predictor,
            extract_from_roi=deepcopy(self.extract_from_category),
            run_time_ocr_language_selection=self.run_time_ocr_language_selection,
        )

    def clear_predictor(self) -> None:
        self.predictor.clear_model()
