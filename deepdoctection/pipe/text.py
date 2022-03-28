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
Module for text extraction pipeline component
"""
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..extern.base import ObjectDetector, PdfMiner, TextRecognizer
from ..utils.detection_types import ImageType
from ..utils.logger import logger
from ..utils.settings import names
from .base import PipelineComponent, PredictorPipelineComponent

__all__ = ["TextExtractionService", "TextOrderService"]


class TextExtractionService(PredictorPipelineComponent):
    """
    Pipeline component for extracting text. Any detector can be selected, provided that it can evaluate a
    numpy array as an image.

    Text extraction can either be carried out over the entire image or over selected regions of interests (ROIs).
    ROIs are layout components that have been determined by means of a pipeline component that has been run through
    beforehand. ROI extraction is particularly suitable when an OCR component is selected as the detector and the
    document has a complex structure. Instead of transferring the entire image, only the ROIs are transferred to
    the detector. Since the ROI has a simpler structure than the entire document page, it can significantly improve
    the OCR results.

    Text components (currently only words) are attached to the image as image annotations. A relation is assigned in
    relation to text and ROI or in relation to text and the entire image. When selecting ROIs, only the selected
    categories are taken into account during processing. ROIs that are not selected are not presented to the
    detector.
    """

    def __init__(
        self,
        text_extract_detector: Union[ObjectDetector, PdfMiner, TextRecognizer],
        extract_from_roi: Optional[Union[List[str], str]] = None,
        category_id_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        :param text_extract_detector: ObjectDetector
        :param extract_from_roi: one or more category names for roi selection
        :param category_id_mapping: Mapping of category IDs. The word category is set to 1 without mapping.

              Example: {1: 9} sets the ID of the image annotation WORD to 9.
        """

        super().__init__(text_extract_detector, category_id_mapping)
        self.extract_from_category = extract_from_roi
        if self.extract_from_category:
            assert isinstance(
                self.predictor, (ObjectDetector, TextRecognizer)
            ), "Predicting from a cropped image requires to pass an ObjectDetector or TextRecognizer."

    def serve(self, dp: Image) -> None:
        maybe_batched_text_rois = self.get_text_rois(dp)
        for text_roi in maybe_batched_text_rois:
            ann_id = None
            if isinstance(text_roi, ImageAnnotation):
                ann_id = text_roi.annotation_id
            predictor_input = self.get_predictor_input(text_roi)
            assert predictor_input is not None
            width, height = None, None
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
                        names.C.CHARS,
                        None,
                        names.C.CHARS,
                        detect_ann_id,
                        detect_result.text if detect_result.text is not None else "",
                        detect_result.score,
                    )
                    if detect_result.block:
                        self.dp_manager.set_category_annotation(
                            names.C.BLOCK, detect_result.block, names.C.BLOCK, detect_ann_id
                        )
                    if detect_result.line:
                        self.dp_manager.set_category_annotation(
                            names.C.TLINE, detect_result.line, names.C.TLINE, detect_ann_id
                        )

    def get_text_rois(self, dp: Image) -> List[Union[Image, ImageAnnotation, List[ImageAnnotation]]]:
        """
        Return image rois based on selected categories. As this selection makes only sense for specific text extractors
        (e.g. those who do proper OCR and do not mine from text from native pdfs) it will do some sanity checks.
        :return: list of ImageAnnotation or Image
        """

        if self.extract_from_category:
            if self.predictor.accepts_batch:
                return [dp.get_annotation(category_names=self.extract_from_category)]
            return dp.get_annotation(category_names=self.extract_from_category)  # type: ignore
        return [dp]

    def get_predictor_input(
        self, text_roi: Union[Image, ImageAnnotation, List[ImageAnnotation]]
    ) -> Optional[Union[bytes, ImageType, List[Tuple[str, ImageType]]]]:
        """
        Return raw input for a given text_roi. This can be a numpy array or pdf bytes and depends on the chosen
        predictor.

        :param text_roi: Image or ImageAnnotation
        :return: pdf bytes or numpy array
        """

        if isinstance(text_roi, ImageAnnotation):
            assert text_roi.image is not None
            assert text_roi.image.image is not None
            return text_roi.image.image
        if isinstance(self.predictor, ObjectDetector):
            assert isinstance(text_roi, Image)
            return text_roi.image
        if isinstance(text_roi, list):
            assert all(roi.image is not None for roi in text_roi)
            assert all(roi.image.image is not None for roi in text_roi)  # type: ignore
            return [(roi.annotation_id, roi.image.image) for roi in text_roi]  # type: ignore
        return text_roi.pdf_bytes


def _reading_lines(image_id: str, word_anns: List[ImageAnnotation]) -> List[Tuple[int, str]]:
    reading_lines = []
    rows: List[Dict[str, float]] = []
    for word in word_anns:
        if word.image is not None:
            bounding_box = word.image.get_embedding(image_id)
        else:
            bounding_box = word.bounding_box
        row_found = False
        for idx, row in enumerate(rows):
            row_cy = (row["upper"] + row["lower"]) / 2

            if (row["upper"] < bounding_box.cy < row["lower"]) or (bounding_box.uly < row_cy < bounding_box.lry):
                reading_lines.append((idx, word.annotation_id))
                row_found = True
                break

        if not row_found:
            rows.append({"upper": bounding_box.uly, "lower": bounding_box.lry})
            reading_lines.append((len(rows) - 1, word.annotation_id))

    rows_dict = {k: rows[k] for k in range(len(rows))}
    rows_dict = {
        idx: key[0] for idx, key in enumerate(sorted(rows_dict.items(), key=lambda it: it[1]["upper"]))  # type:ignore
    }
    reading_lines.sort(key=lambda x: rows_dict[x[0]])  # type:ignore
    reading_lines = [(idx + 1, word[1]) for idx, word in enumerate(reading_lines)]
    return reading_lines


def _reading_columns(
    dp: Image,
    anns: List[ImageAnnotation],
    starting_point_tolerance: float = 0.01,
    height_tolerance: float = 3.0,
    same_line_top_tolerance: float = 0.001,
    same_line_spacing_tolerance: float = 5.0,
) -> List[Tuple[int, str]]:
    reading_blocks = []
    columns: List[Dict[str, float]] = []
    anns.sort(key=lambda x: (x.bounding_box.cy, x.bounding_box.cx))  # type: ignore
    for ann in anns:
        if ann.image is not None:
            bounding_box = ann.image.get_embedding(dp.image_id)
        else:
            bounding_box = ann.bounding_box

        if bounding_box.absolute_coords:
            rel_coords_box = bounding_box.transform(dp.width, dp.height, output="box")
        else:
            rel_coords_box = bounding_box

        column_found = False
        for idx, col in enumerate(columns):
            # if the starting point is within starting_point_tolerance (first_condition) and
            # the top location is within height_tolerance * bbox_height (second_condition), or
            # the new line appeared to be broken by Textract mistake and should be of the same line
            # by looking at the top (third_condition) and
            # the left of the new line appears right next to the right of the last line (fourth_condition)
            # then consider the new line as part of said column
            first_condition = abs(rel_coords_box.ulx - col["left"]) < starting_point_tolerance
            second_condition = abs(rel_coords_box.uly - col["top"]) < height_tolerance * bounding_box.height
            third_condition = (
                abs(rel_coords_box.uly - col["top"]) < same_line_top_tolerance
            )  # appeared to be in the same line
            fourth_condition = (
                abs(rel_coords_box.ulx - col["right"]) < same_line_spacing_tolerance * starting_point_tolerance
            )
            if (first_condition and second_condition) or (third_condition and fourth_condition):
                reading_blocks.append((idx, ann.annotation_id))
                # update the top and right with the new line added.
                col["top"] = rel_coords_box.uly
                col["right"] = rel_coords_box.lry
                column_found = True
                break

        if not column_found:
            columns.append({"left": rel_coords_box.ulx, "right": rel_coords_box.lrx, "top": rel_coords_box.uly})
            # update the top and right with the new reading block added.
            reading_blocks.append((len(columns) - 1, ann.annotation_id))

    columns_dict = {k: columns[k] for k in range(len(columns))}
    columns_dict = {
        idx: key[0] for idx, key in enumerate(sorted(columns_dict.items(), key=lambda it: it[1]["left"]))  # type:ignore
    }
    reading_blocks = [(columns_dict[x[0]], x[1]) for x in reading_blocks]  # type:ignore
    reading_blocks.sort(key=lambda x: x[0])
    reading_blocks = [(idx + 1, block[1]) for idx, block in enumerate(reading_blocks)]
    return reading_blocks


class TextOrderService(PipelineComponent):
    """
    Reading order of words within floating text blocks as well as reading order of blocks within simple text blocks.
    To understand the difference between floating text blocks and simple text blocks consider a page containing an
    article and a table. Table Cells are text blocks that contain words which must be sorted.
    However, they do not belong to floating text that encircle a table. They are rather an element that is supposed to
    be read independently.

    A heuristic argument for its ordering is used where the underlying assumption is the reading order from left to
    right.

        - For the reading order within a text block, text containers (i.e. image annotations that contain character
          sub annotations) are sorted based on their bounding box center and then lines are formed: Each word induces a
          new line, provided that its center is not in a line that has already
          been created by an already processed word. The entire block width is defined as the line width
          and the upper or lower line limit of the word bounding box as the upper or lower line limit. The reading order
          of the words is from left to right within a line. The reading order of the lines is from top to bottom.

        - For the reading order of text blocks within a page, the blocks are sorted using a similar procedure, with the
          difference that columns are formed instead of lines. Column lengths are defined as the length of the entire
          page and the left and right text block boundaries as the left and right column boundaries.

    A category annotation per word is generated, which fixes the order per word in the block, as well as a category
    annotation per block, which saves the reading order of the block per page.

    The blocks are defined in :attr:`_floating_text_block_names` and text blocks in :attr:`_floating_text_block_names`.
    """

    def __init__(
        self,
        text_container: str,
        floating_text_block_names: Optional[Union[str, List[str]]] = None,
        text_block_names: Optional[Union[str, List[str]]] = None,
        text_containers_to_text_block: bool = False,
    ) -> None:
        """
        :param text_container: name of an image annotation that has a CHARS sub category. These annotations will be
                               ordered within all text blocks.
        :param floating_text_block_names: name of image annotation that belong to floating text. These annotations form
                                          the highest hierarchy of text blocks that will ordered to generate a sensible
                                          output of text
        :param text_block_names: name of image annotation that have a relation with text containers (or which might be
                                 text containers themselves).
        :param text_containers_to_text_block: Text containers are in general no text blocks and belong to a lower
                                              hierarchy. However, if a text container is not assigned to a text block
                                              you can add it to the text block ordering to ensure that the full text is
                                              part of the subsequent sub process.
        """
        if isinstance(floating_text_block_names, str):
            floating_text_block_names = [floating_text_block_names]
        elif floating_text_block_names is None:
            floating_text_block_names = []
        if isinstance(text_block_names, str):
            text_block_names = [text_block_names]
        elif text_block_names is None:
            text_block_names = []

        super().__init__(None)
        self._text_container = text_container
        self._floating_text_block_names = floating_text_block_names
        self._text_block_names = text_block_names
        self._text_containers_to_text_block = text_containers_to_text_block
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        # select all text blocks that are considered to be relevant for page text. This does not include some layout
        # items that have to be considered independently (e.g. tables). Order the blocks by column wise reading order
        text_block_anns = dp.get_annotation(category_names=self._floating_text_block_names)

        # maybe add all text containers that are not mapped to a text block
        if self._text_containers_to_text_block:
            text_ann_ids = list(chain(*[text_block.get_relationship(names.C.CHILD) for text_block in text_block_anns]))
            text_container_anns = dp.get_annotation(category_names=self._text_container)
            text_container_anns = [ann for ann in text_container_anns if ann.annotation_id not in text_ann_ids]
            text_block_anns.extend(text_container_anns)

        raw_reading_order_list = _reading_columns(dp, text_block_anns)
        for raw_reading_order in raw_reading_order_list:
            self.dp_manager.set_category_annotation(names.C.RO, raw_reading_order[0], names.C.RO, raw_reading_order[1])
        # next we select all blocks that might contain text. We sort all text within these blocks
        block_anns = dp.get_annotation(category_names=self._text_block_names)
        for text_block in block_anns:
            text_container_ann_ids = text_block.get_relationship(names.C.CHILD)
            text_container_anns = dp.get_annotation(
                annotation_ids=text_container_ann_ids if text_container_ann_ids is not None else [],
                category_names=self._text_container,
            )
            raw_reading_order_list = _reading_lines(dp.image_id, text_container_anns)
            for raw_reading_order in raw_reading_order_list:
                self.dp_manager.set_category_annotation(
                    names.C.RO, raw_reading_order[0], names.C.RO, raw_reading_order[1]
                )

    def clone(self) -> PipelineComponent:
        return self.__class__(
            self._text_container,
            self._floating_text_block_names,
            self._text_block_names,
            self._text_containers_to_text_block,
        )

    def _init_sanity_checks(self) -> None:
        assert self._text_container in [names.C.WORD, names.C.LINE], (
            f"text_container must be either {names.C.WORD} or " f"{names.C.LINE}"
        )
        assert set(self._floating_text_block_names) <= set(
            self._text_block_names
        ), "floating_text_block_names must be a subset of text_block_names"
        if (
            not self._floating_text_block_names
            and not self._text_block_names
            and not self._text_containers_to_text_block
        ):
            logger.info(
                "floating_text_block_names and text_block_names are set to None and "
                "text_containers_to_text_block is set to False. This setting will return no reading order!"
            )
        if (
            self._text_container == names.C.WORD
            and self._text_containers_to_text_block
            and not self._floating_text_block_names
        ):
            logger.info(
                "Choosing %s text_container while choosing no text_blocks will give no sensible "
                "results. Choose %s text_container if you do not have text_blocks available.",
                names.C.WORD,
                names.C.LINE,
            )
