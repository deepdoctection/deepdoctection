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
from copy import copy, deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..extern.base import ObjectDetector, PdfMiner, TextRecognizer
from ..extern.tessocr import TesseractOcrDetector
from ..utils.detection_types import ImageType, JsonDict
from ..utils.logger import logger
from ..utils.settings import LayoutType, PageType, Relationships, TypeOrStr, WordType, get_type
from .base import PipelineComponent, PredictorPipelineComponent
from .registry import pipeline_component_registry

__all__ = ["TextExtractionService", "TextOrderService"]


@pipeline_component_registry.register("TextExtractionService")
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

    .. code-block:: python

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
        :param text_extract_detector: ObjectDetector
        :param extract_from_roi: one or more category names for roi selection
        """

        if extract_from_roi is None:
            extract_from_roi = []
        self.extract_from_category = (
            [get_type(extract_from_roi)]
            if isinstance(extract_from_roi, str)
            else [get_type(roi_category) for roi_category in extract_from_roi]
        )
        super().__init__(self._get_name(text_extract_detector.name), text_extract_detector)
        if self.extract_from_category:
            if not isinstance(self.predictor, (ObjectDetector, TextRecognizer)):
                raise TypeError("Predicting from a cropped image requires to pass an ObjectDetector or TextRecognizer.")
        if run_time_ocr_language_selection:
            assert isinstance(self.predictor, TesseractOcrDetector), (
                "Only TesseractOcrDetector supports multiple " "languages"
            )

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
            width, height = None, None
            if self.run_time_ocr_language_selection:
                self.predictor.set_language(dp.summary.get_sub_category(PageType.language).value)  # type: ignore
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
                        WordType.characters,
                        None,
                        WordType.characters,
                        detect_ann_id,
                        detect_result.text if detect_result.text is not None else "",
                        detect_result.score,
                    )
                    if detect_result.block:
                        self.dp_manager.set_category_annotation(
                            WordType.block, detect_result.block, WordType.block, detect_ann_id
                        )
                    if detect_result.line:
                        self.dp_manager.set_category_annotation(
                            WordType.text_line, detect_result.line, WordType.text_line, detect_ann_id
                        )

    def get_text_rois(self, dp: Image) -> Sequence[Union[Image, ImageAnnotation, List[ImageAnnotation]]]:
        """
        Return image rois based on selected categories. As this selection makes only sense for specific text extractors
        (e.g. those who do proper OCR and do not mine from text from native pdfs) it will do some sanity checks.
        :return: list of ImageAnnotation or Image
        """

        if self.extract_from_category:
            if self.predictor.accepts_batch:
                return [dp.get_annotation(category_names=self.extract_from_category)]
            return dp.get_annotation(category_names=self.extract_from_category)
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
            if text_roi.image is None:
                raise ValueError("text_roi.image cannot be None")
            if text_roi.image.image is None:
                raise ValueError("text_roi.image.image cannot be None")
            return text_roi.image.image
        if isinstance(self.predictor, ObjectDetector):
            if not isinstance(text_roi, Image):
                raise ValueError("text_roi must be an image")
            return text_roi.image
        if isinstance(text_roi, list):
            assert all(roi.image is not None for roi in text_roi)
            assert all(roi.image.image is not None for roi in text_roi)  # type: ignore
            return [(roi.annotation_id, roi.image.image) for roi in text_roi]  # type: ignore
        return text_roi.pdf_bytes

    def get_meta_annotation(self) -> JsonDict:
        if self.extract_from_category:
            sub_cat_dict = {category: {WordType.characters} for category in self.extract_from_category}
        else:
            if not isinstance(self.predictor, (ObjectDetector, PdfMiner)):
                raise TypeError(
                    f"self.predictor must be of type ObjectDetector or PdfMiner but is of type "
                    f"{type(self.predictor)}"
                )
            sub_cat_dict = {category: {WordType.characters} for category in self.predictor.possible_categories()}
        return dict(
            [
                (
                    "image_annotations",
                    self.predictor.possible_categories()
                    if isinstance(self.predictor, (ObjectDetector, PdfMiner))
                    else [],
                ),
                ("sub_categories", sub_cat_dict),
                ("relationships", {}),
                ("summaries", []),
            ]
        )

    @staticmethod
    def _get_name(text_detector_name: str) -> str:
        return f"text_extract_{text_detector_name}"

    def clone(self) -> "PredictorPipelineComponent":
        predictor = self.predictor.clone()
        if not isinstance(predictor, (ObjectDetector, PdfMiner, TextRecognizer)):
            raise ValueError(f"predictor must be of type ObjectDetector or PdfMiner, but is of type {type(predictor)}")
        return self.__class__(predictor, deepcopy(self.extract_from_category), self.run_time_ocr_language_selection)


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
                reading_lines.append((idx, word.annotation_id, bounding_box.cx))
                row_found = True
                break

        if not row_found:
            rows.append({"upper": bounding_box.uly, "lower": bounding_box.lry})
            reading_lines.append((len(rows) - 1, word.annotation_id, bounding_box.cx))

    rows_dict = {k: rows[k] for k in range(len(rows))}
    rows_dict = {
        idx: key[0] for idx, key in enumerate(sorted(rows_dict.items(), key=lambda it: it[1]["upper"]))  # type:ignore
    }
    reading_lines.sort(key=lambda x: (rows_dict[x[0]], x[2]))
    return [(idx + 1, word[1]) for idx, word in enumerate(reading_lines)]


def _reading_columns(
    dp: Image,
    anns: List[ImageAnnotation],
    starting_point_tolerance: float = 0.01,
    height_tolerance: float = 3.0,
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
            rel_coords_box = bounding_box.transform(dp.width, dp.height)
        else:
            rel_coords_box = bounding_box

        column_found = False
        for idx, col in enumerate(columns):
            # if the x-coordinate left and right is within starting_point_tolerance (first_condition and
            # second_condition) or if x-coordinate left and right is within the left or right border of the column
            # then the annotation will belong to this column and column left/right will be re-adjusted

            first_condition = all(
                (
                    col["left"] - starting_point_tolerance < rel_coords_box.ulx,
                    rel_coords_box.lrx < col["right"] + starting_point_tolerance,
                )
            )
            second_condition = all(
                (
                    rel_coords_box.ulx - starting_point_tolerance < col["left"],
                    col["right"] < rel_coords_box.lrx + starting_point_tolerance,
                )
            )

            third_condition = abs(rel_coords_box.uly - col["bottom"]) < height_tolerance * rel_coords_box.height
            fourth_condition = abs(rel_coords_box.lry - col["top"]) < height_tolerance * rel_coords_box.height

            if (first_condition and (third_condition or fourth_condition)) or (  # pylint: disable=R0916
                second_condition and (third_condition or fourth_condition)
            ):
                reading_blocks.append((idx, ann.annotation_id))
                # update the top and right with the new line added.
                col["left"] = min(rel_coords_box.ulx, col["left"])
                col["top"] = min(rel_coords_box.uly, col["top"])
                col["right"] = max(rel_coords_box.lrx, col["right"])
                col["bottom"] = max(rel_coords_box.lry, col["bottom"])
                column_found = True
                break

        if not column_found:
            columns.append(
                {
                    "left": rel_coords_box.ulx,
                    "right": rel_coords_box.lrx,
                    "top": rel_coords_box.uly,
                    "bottom": rel_coords_box.lry,
                }
            )
            # update the top and right with the new reading block added.
            reading_blocks.append((len(columns) - 1, ann.annotation_id))

    # building connected components of columns
    connected_components: List[Dict[str, Any]] = []
    for idx, col in enumerate(columns):
        col["id"] = idx
        component_found = False
        for comp in connected_components:
            if comp["top"] < col["top"] < comp["bottom"] or comp["top"] < col["bottom"] < comp["bottom"]:
                comp["top"] = min(comp["top"], col["top"])
                comp["bottom"] = max(comp["bottom"], col["bottom"])
                comp["left"] = col["left"]
                comp["column"].append(col)
                component_found = True
                break
        if not component_found:
            connected_components.append(
                {"top": col["top"], "bottom": col["bottom"], "left": col["left"], "column": [col]}
            )

    # next, sorting columns in connected components by increasing x-value
    for comp in connected_components:
        comp["column"].sort(key=lambda x: x["left"])

    # finally, sorting connected components by increasing y-value
    connected_components.sort(key=lambda x: x["top"])
    columns = list(chain(*[comp["column"] for comp in connected_components]))

    # old to new mapping
    columns_dict = {k: col["id"] for k, col in enumerate(columns)}
    _blocks = [(columns_dict[x[0]], x[1]) for x in reading_blocks]
    _blocks.sort(key=lambda x: x[0])
    reading_blocks = [(idx + 1, block[1]) for idx, block in enumerate(_blocks)]
    return reading_blocks


@pipeline_component_registry.register("TextOrderService")
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

    .. code-block:: python

        order = TextOrderService(text_container=names.C.WORD,
                                 floating_text_block_names=[names.C.TITLE, names.C.TEXT, names.C.LIST],
                                 text_block_names=[names.C.TITLE, names.C.TEXT, names.C.LIST, names.C.CELL,
                                                   names.C.HEAD, names.C.BODY])
    """

    def __init__(
        self,
        text_container: str,
        floating_text_block_names: Optional[Union[str, Sequence[str]]] = None,
        text_block_names: Optional[Union[str, Sequence[str]]] = None,
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
                                              part of the subsequent sub process. Note however, that if the text
                                              container is on word level rather than line level, the results will not be
                                              very convincing
        """
        if isinstance(floating_text_block_names, str):
            floating_text_block_names = [floating_text_block_names]
        elif floating_text_block_names is None:
            floating_text_block_names = []
        if isinstance(text_block_names, str):
            text_block_names = [text_block_names]
        elif text_block_names is None:
            text_block_names = []

        self._text_container = text_container
        self._floating_text_block_names = floating_text_block_names
        self._text_block_names = text_block_names
        self._text_containers_to_text_block = text_containers_to_text_block
        self._init_sanity_checks()
        super().__init__("text_order")

    def serve(self, dp: Image) -> None:
        # select all text blocks that are considered to be relevant for page text. This does not include some layout
        # items that have to be considered independently (e.g. tables). Order the blocks by column wise reading order
        text_block_anns = dp.get_annotation(category_names=self._floating_text_block_names)
        number_text_block_anns_orig = len(text_block_anns)
        # maybe add all text containers that are not mapped to a text block
        if self._text_containers_to_text_block:
            text_ann_ids = list(
                chain(*[text_block.get_relationship(Relationships.child) for text_block in text_block_anns])
            )
            text_container_anns = dp.get_annotation(category_names=self._text_container)
            text_container_anns = [ann for ann in text_container_anns if ann.annotation_id not in text_ann_ids]
            text_block_anns.extend(text_container_anns)

        # estimating reading columns. We will only do this if we have some text blocks that are no text_containers
        # (number_text_block_anns_orig >0) or if the text container is not a word. Otherwise, we will have to skip that
        # part
        if self._text_container != LayoutType.word or number_text_block_anns_orig:
            raw_reading_order_list = _reading_columns(dp, text_block_anns, 0.05, 2.0)

            for raw_reading_order in raw_reading_order_list:
                self.dp_manager.set_category_annotation(
                    Relationships.reading_order, raw_reading_order[0], Relationships.reading_order, raw_reading_order[1]
                )

        # next we select all blocks that might contain text. We sort all text within these blocks
        block_anns = dp.get_annotation(category_names=self._text_block_names)
        for text_block in block_anns:
            text_container_ann_ids = text_block.get_relationship(Relationships.child)
            text_container_anns = dp.get_annotation(
                annotation_ids=text_container_ann_ids,
                category_names=self._text_container,
            )
            raw_reading_order_list = _reading_lines(dp.image_id, text_container_anns)
            for raw_reading_order in raw_reading_order_list:
                self.dp_manager.set_category_annotation(
                    Relationships.reading_order, raw_reading_order[0], Relationships.reading_order, raw_reading_order[1]
                )

        # this is the setting where we order words without having text blocks
        if not block_anns:
            text_container_anns = dp.get_annotation(category_names=self._text_container)
            # some OCR systems return textline and blocks. If they are available we will sort first by block, then by
            # line and finally by center x coord.
            if text_container_anns:
                text_container_ann = text_container_anns[0]
                if WordType.block and WordType.text_line in text_container_ann.sub_categories:
                    text_container_position = [
                        (
                            int(ann.get_sub_category(WordType.block).category_id),
                            int(ann.get_sub_category(WordType.text_line).category_id),
                            ann.bounding_box.cx,  # type: ignore
                            ann.annotation_id,
                        )
                        for ann in text_container_anns
                    ]
                    text_container_position.sort(key=lambda x: (x[0], x[1], x[2]))
                    for position, element in enumerate(text_container_position):
                        self.dp_manager.set_category_annotation(
                            Relationships.reading_order, position, Relationships.reading_order, element[3]
                        )
                else:
                    # Last try. We only form lines without and define a reading from this
                    raw_reading_order_list = _reading_lines(dp.image_id, text_container_anns)
                    for raw_reading_order in raw_reading_order_list:
                        self.dp_manager.set_category_annotation(
                            Relationships.reading_order,
                            raw_reading_order[0],
                            Relationships.reading_order,
                            raw_reading_order[1],
                        )

    def clone(self) -> PipelineComponent:
        return self.__class__(
            copy(self._text_container),
            deepcopy(self._floating_text_block_names),
            deepcopy(self._text_block_names),
            deepcopy(self._text_containers_to_text_block),
        )

    def _init_sanity_checks(self) -> None:
        assert self._text_container in [LayoutType.word, LayoutType.line], (
            f"text_container must be either {LayoutType.word} or " f"{LayoutType.line}"
        )
        assert set(self._floating_text_block_names) <= set(
            self._text_block_names
        ), "floating_text_block_names must be a subset of text_block_names"
        if (
            self._text_container == LayoutType.word
            and self._text_containers_to_text_block
            and not self._floating_text_block_names
        ):
            logger.info(
                "Choosing %s text_container while choosing no text_blocks will give no sensible "
                "results. Choose %s text_container if you do not have text_blocks available.",
                LayoutType.word,
                LayoutType.line,
            )

    def get_meta_annotation(self) -> JsonDict:
        anns_with_reading_order = list(deepcopy(self._floating_text_block_names))
        anns_with_reading_order.extend([LayoutType.word, LayoutType.line])
        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {category: {Relationships.reading_order} for category in anns_with_reading_order}),
                ("relationships", {}),
                ("summaries", []),
            ]
        )
