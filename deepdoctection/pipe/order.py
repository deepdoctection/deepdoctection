# -*- coding: utf-8 -*-
# File: order.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Ordering text and layout segments
"""
from __future__ import annotations

import os
from abc import ABC
from copy import copy
from itertools import chain
from logging import DEBUG
from typing import Any, Optional, Sequence, Union

import numpy as np

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import BoundingBox, merge_boxes
from ..datapoint.image import Image, MetaAnnotation
from ..datapoint.view import IMAGE_DEFAULTS
from ..extern.base import DetectionResult
from ..extern.tp.tpfrcnn.utils.np_box_ops import ioa as np_ioa
from ..pipe.base import PipelineComponent
from ..pipe.registry import pipeline_component_registry
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import LayoutType, ObjectTypes, Relationships, TypeOrStr, get_type


class OrderGenerator:
    """
    Class for implementing text ordering logic and tasks that have preparational character.

    This includes logic for grouping word type `ImageAnnotation` into text lines, splitting text lines into sub-lines
     (by detecting gaps between words), as well as ordering text blocks (e.g., titles, tables, etc.).
    """

    def __init__(self, starting_point_tolerance: float, broken_line_tolerance: float, height_tolerance: float):
        """
        Args:
            starting_point_tolerance: Threshold to identify if two text blocks belong to one column. To check if two
                                      text blocks belong to the same column, one condition says that x-coordinates of
                                      vertices should not differ more than this threshold.
            broken_line_tolerance: Threshold to identify if two consecutive words belonging to one line should be in two
                                   different sub-lines (because they belong to two different text columns).
            height_tolerance: Threshold to identify if two columns lying over each other belong together or need to be
                              separated. Scaling factor of relative text block height.
        """
        self.starting_point_tolerance = starting_point_tolerance
        self.broken_line_tolerance = broken_line_tolerance
        self.height_tolerance = height_tolerance
        self.ioa_column_threshold = 0.9
        self.columns_detect_result: Optional[Sequence[DetectionResult]] = None

    @staticmethod
    def group_words_into_lines(
        word_anns: Sequence[ImageAnnotation], image_id: Optional[str] = None
    ) -> list[tuple[int, int, str]]:
        """
        Arranges words into horizontal text lines and sorts text lines vertically to provide an enumeration of words
        used for establishing the reading order.

        Using this reading order arrangement makes sense only for words within a rectangle and needs to be revised in
        more complex appearances.

        Example:
            ```python
            group_words_into_lines(word_anns, image_id)
            ```

        Args:
            word_anns: Sequence of `ImageAnnotation` representing words.
            image_id: Optional image ID.

        Returns:
            List of triplets for every word annotation: (word reading order position, text line position, word
            annotation id).


        """
        reading_lines = []
        rows: list[dict[str, float]] = []
        for word in word_anns:
            bounding_box = word.get_bounding_box(image_id)
            row_found = False
            for idx, row in enumerate(rows):
                row_cy = (row["upper"] + row["lower"]) / 2
                # word belongs to row if center lies within the upper and lower bounds of the row or if the center y
                # coordinate lies within the upper and lower bounds of the word bounding boxes.
                # If word belongs to bound we do not update any row bounds. Thus, row bound are determined by the
                # first word that defines the row
                if (row["upper"] < bounding_box.cy < row["lower"]) or (bounding_box.uly < row_cy < bounding_box.lry):
                    reading_lines.append((idx, word.annotation_id, bounding_box.cx))
                    row_found = True
                    break

            # condition above not satisfied for any row, thus word defines a new row
            if not row_found:
                rows.append({"upper": bounding_box.uly, "lower": bounding_box.lry})
                reading_lines.append((len(rows) - 1, word.annotation_id, bounding_box.cx))

        rows_dict = {k: rows[k] for k in range(len(rows))}
        rows_dict = {
            idx: key[0]  # type:ignore
            for idx, key in enumerate(sorted(rows_dict.items(), key=lambda it: it[1]["upper"]))
        }
        reading_lines.sort(key=lambda x: (rows_dict[x[0]], x[2]))
        number_rows = len(rows_dict)
        if logger.isEnabledFor(DEBUG):
            logger.debug(
                LoggingRecord(
                    "group_words_into_lines",
                    {"number_rows": number_rows, "reading_lines": reading_lines, "rows_dict": rows_dict},
                )
            )
        return [(idx + 1, number_rows - word[0], word[1]) for idx, word in enumerate(reading_lines)]

    @staticmethod
    def group_lines_into_lines(
        line_anns: Sequence[ImageAnnotation], image_id: Optional[str] = None
    ) -> list[tuple[int, int, str]]:
        """
        Sorts reading lines.

        Returns for a list of `ImageAnnotation` a list of tuples, each tuple containing the reading order and the
        `annotation_id` for each list element.

        Args:
            line_anns: Sequence of text line `ImageAnnotation`.
            image_id: Image ID of underlying image (to get the bounding boxes).

        Returns:
            List of tuples (reading_order, reading_order, annotation_id).

        Example:
            ```python
            group_lines_into_lines(line_anns, image_id)
            ```
        """
        reading_lines = []
        for ann in line_anns:
            bounding_box = ann.get_bounding_box(image_id)
            reading_lines.append((bounding_box.cy, ann.annotation_id))
        reading_lines.sort(key=lambda x: x[0])
        logger.debug(LoggingRecord("group_lines_into_lines", {"reading_lines": reading_lines}))
        return [(idx + 1, idx + 1, line[1]) for idx, line in enumerate(reading_lines)]

    @staticmethod
    def _connected_components(columns: list[BoundingBox]) -> list[dict[str, Any]]:
        # building connected components of columns
        connected_components: list[dict[str, Any]] = []
        for idx, col in enumerate(columns):
            col_dict = {"id": idx, "box": col}
            component_found = False
            for comp in connected_components:
                if (
                    comp["top"] < col.uly < comp["bottom"]
                    or comp["top"] < col.lry < comp["bottom"]
                    or col.uly < comp["top"] < col.lry
                    or col.uly < comp["bottom"] < col.lry
                ):
                    comp["top"] = min(comp["top"], col.uly)
                    comp["bottom"] = max(comp["bottom"], col.lry)
                    comp["left"] = col.ulx
                    comp["column"].append(col_dict)
                    component_found = True
                    break
            if not component_found:
                connected_components.append({"top": col.uly, "bottom": col.lry, "left": col.ulx, "column": [col_dict]})

            # next, sorting columns in connected components by increasing x-value. In order to be tolerant to
            # nearby values we are rounding values we want to sort
            for comp in connected_components:
                for column in comp["column"]:
                    column["box"].ulx = round(column["box"].ulx, 2)
                    column["box"].uly = round(column["box"].uly, 2)
                comp["column"].sort(key=lambda x: (x["box"].ulx, x["box"].uly))

            # finally, sorting connected components by increasing y-value
            connected_components.sort(key=lambda x: x["top"])
        if logger.isEnabledFor(DEBUG):
            logger.debug(LoggingRecord("_connected_components", {"connected_components": str(connected_components)}))
        return connected_components

    def order_blocks(
        self, anns: list[ImageAnnotation], image_width: float, image_height: float, image_id: Optional[str] = None
    ) -> Sequence[tuple[int, str]]:
        """
        Determines a text ordering of text blocks.

        These text blocks should be larger sections than just words. It will first try to detect columns, then try to
        consolidate columns, and finally try to detect connected components of columns. A connected component of columns
        is a group of columns that lie next to each other. Having two connected components lying over each other will
        infer a reading order where the upper block of the connected component will be read first, followed by text
        blocks of columns of the second.

        Example:
            ```python
            order_blocks(anns, image_width, image_height, image_id)
            ```

        Args:
            anns: List of `ImageAnnotation` with all elements to sort.
            image_width: Image width (to re-calculate bounding boxes into relative coordinates).
            image_height: Image height (to re-calculate bounding boxes into relative coordinates).
            image_id: Image ID.

        Returns:
            List of tuples with reading order position and `annotation_id`.
        """
        if not anns:
            return []
        reading_blocks = []
        columns: list[BoundingBox] = []
        anns.sort(
            key=lambda x: (
                x.bounding_box.transform(image_width, image_height).cy,  # type: ignore
                x.bounding_box.transform(image_width, image_height).cx,  # type: ignore
            )
        )
        for ann in anns:
            bounding_box = ann.get_bounding_box(image_id)
            if bounding_box.absolute_coords:
                rel_coords_box = bounding_box.transform(image_width, image_height)
            else:
                rel_coords_box = bounding_box

            column_found = False
            for idx, col in enumerate(columns):
                # if the x-coordinate left and right is within starting_point_tolerance (first_condition and
                # second_condition) or if x-coordinate left and right is within the left or right border of the column
                # then the annotation will belong to this column and column left/right will be re-adjusted
                first_condition = all(
                    (
                        col.ulx - self.starting_point_tolerance < rel_coords_box.ulx,
                        rel_coords_box.lrx < col.lrx + self.starting_point_tolerance,
                    )
                )
                second_condition = all(
                    (
                        rel_coords_box.ulx - self.starting_point_tolerance < col.ulx,
                        col.lrx < rel_coords_box.lrx + self.starting_point_tolerance,
                    )
                )
                # broken line condition
                third_condition = abs(rel_coords_box.ulx - col.lrx) < self.broken_line_tolerance
                fourth_condition = abs(rel_coords_box.uly - col.lry) < self.height_tolerance * rel_coords_box.height
                fifth_condition = abs(rel_coords_box.lry - col.uly) < self.height_tolerance * rel_coords_box.height

                if (first_condition and (fourth_condition or fifth_condition)) or (  # pylint: disable=R0916
                    second_condition
                    and (fourth_condition or fifth_condition)
                    or (third_condition and (fourth_condition or fifth_condition))
                ):
                    reading_blocks.append((idx, ann.annotation_id))
                    # update the top and right with the new line added.
                    col.ulx = min(rel_coords_box.ulx, col.ulx)
                    col.uly = min(rel_coords_box.uly, col.uly)
                    col.lrx = max(rel_coords_box.lrx, col.lrx)
                    col.lry = max(rel_coords_box.lry, col.lry)
                    column_found = True
                    break

            if not column_found:
                columns.append(
                    BoundingBox(
                        absolute_coords=False,
                        ulx=rel_coords_box.ulx,
                        uly=rel_coords_box.uly,
                        lrx=rel_coords_box.lrx,
                        lry=rel_coords_box.lry,
                    )
                )
                # update the top and right with the new reading block added.
                reading_blocks.append((len(columns) - 1, ann.annotation_id))
        self.columns_detect_result = self._make_column_detect_results(columns)
        consoldiated_cols = self._consolidate_columns(columns)

        consolidated_columns = []
        for idx, _ in enumerate(columns):
            if columns[consoldiated_cols[idx]] not in consolidated_columns:
                consolidated_columns.append(columns[consoldiated_cols[idx]])

        reading_blocks = [(consoldiated_cols.get(x[0], x[0]), x[1]) for x in reading_blocks]

        connected_components = self._connected_components(consolidated_columns)
        columns_box = list(chain(*[comp["column"] for comp in connected_components]))

        # old to new mapping
        columns_dict = {col["id"]: k for k, col in enumerate(columns_box)}
        blocks = [(columns_dict.get(x[0], consoldiated_cols.get(x[0])), x[1]) for x in reading_blocks]
        blocks.sort(key=lambda x: x[0])  # type: ignore
        sorted_blocks = []
        max_block_number = max(list(columns_dict.values()))
        filtered_blocks: Sequence[tuple[int, str]]
        for idx in range(max_block_number + 1):
            filtered_blocks = list(filter(lambda x: x[0] == idx, blocks))  # type: ignore # pylint: disable=W0640
            sorted_blocks.extend(self._sort_anns_grouped_by_blocks(filtered_blocks, anns, image_width, image_height))
        reading_blocks = [(idx + 1, block[1]) for idx, block in enumerate(sorted_blocks)]

        if logger.isEnabledFor(DEBUG):
            logger.debug(
                LoggingRecord(
                    "order_blocks",
                    {
                        "consolidated_cols": str(consoldiated_cols),
                        "columns": str(columns),
                        "reading_blocks": str(reading_blocks),
                    },
                )
            )
        return reading_blocks

    def _consolidate_columns(self, columns: list[BoundingBox]) -> dict[int, int]:
        if not columns:
            return {}
        np_boxes = np.array([col.to_list(mode="xyxy") for col in columns])
        ioa_matrix = np.transpose(np_ioa(np_boxes, np_boxes))
        np.fill_diagonal(ioa_matrix, 0)
        output = ioa_matrix > self.ioa_column_threshold
        child_index, parent_index = output.nonzero()
        column_dict = dict(zip(child_index, parent_index))
        column_dict = {int(key): int(val) for key, val in column_dict.items()}
        counter = 0
        for idx, _ in enumerate(columns):
            if idx not in column_dict:
                column_dict[idx] = counter
                counter += 1
        if logger.isEnabledFor(DEBUG):
            logger.debug(LoggingRecord("consolidated columns", copy(column_dict)))
        return column_dict

    @staticmethod
    def _sort_anns_grouped_by_blocks(
        block: Sequence[tuple[int, str]], anns: Sequence[ImageAnnotation], image_width: float, image_height: float
    ) -> list[tuple[int, str]]:
        if not block:
            return []
        anns_and_blocks_numbers = list(zip(*block))
        ann_ids = anns_and_blocks_numbers[1]
        block_number = anns_and_blocks_numbers[0][0]
        block_anns = [ann for ann in anns if ann.annotation_id in ann_ids]
        block_anns.sort(
            key=lambda x: (
                round(x.bounding_box.transform(image_width, image_height).uly, 2),  # type: ignore
                round(x.bounding_box.transform(image_width, image_height).ulx, 2),  # type: ignore
            )
        )
        return [(block_number, ann.annotation_id) for ann in block_anns]

    @staticmethod
    def _make_column_detect_results(columns: Sequence[BoundingBox]) -> Sequence[DetectionResult]:
        column_detect_result_list = []
        if os.environ.get("LOG_LEVEL", "INFO") == "DEBUG":
            for box in columns:
                column_detect_result_list.append(
                    DetectionResult(
                        box=box.to_list(mode="xyxy"),
                        absolute_coords=box.absolute_coords,
                        class_id=99,
                        class_name=LayoutType.COLUMN,
                    )
                )
        return column_detect_result_list


class TextLineGenerator:
    """
    Class for generating synthetic text lines from words.

    Possible to break text lines into sub-lines by using a paragraph break threshold. This allows detection of a
    multi-column structure just by observing sub-lines.


    """

    def __init__(self, make_sub_lines: bool, paragraph_break: Optional[float] = None):
        """
        Args:
            make_sub_lines: Whether to build sub-lines from lines.
            paragraph_break: Threshold of two consecutive words. If distance is larger than threshold, two sub-lines
                will be built. Relative coordinates are used to calculate the distance between two consecutive words.
                A reasonable value is `0.035`.

        Raises:
            ValueError: If `make_sub_lines` is `True` and `paragraph_break` is `None`.
        """
        if make_sub_lines and paragraph_break is None:
            raise ValueError("You must specify paragraph_break when setting make_sub_lines to True")
        self.make_sub_lines = make_sub_lines
        self.paragraph_break = paragraph_break

    def _make_detect_result(self, box: BoundingBox, relationships: dict[str, list[str]]) -> DetectionResult:
        return DetectionResult(
            box=box.to_list(mode="xyxy"),
            class_name=LayoutType.LINE,
            absolute_coords=box.absolute_coords,
            relationships=relationships,
        )

    def create_detection_result(
        self,
        word_anns: Sequence[ImageAnnotation],
        image_width: float,
        image_height: float,
        image_id: Optional[str] = None,
        highest_level: bool = True,
    ) -> Sequence[DetectionResult]:
        """
        Creates detection result of lines (or sub-lines) from given word type `ImageAnnotation`.

        Example:
            ```python
            create_detection_result(word_anns, image_width, image_height, image_id)
            ```

        Args:
            word_anns: List of given word type `ImageAnnotation`.
            image_width: Image width.
            image_height: Image height.
            image_id: Image ID.
            highest_level: Whether this is the highest level of line creation.

        Returns:
            Sequence of `DetectionResult`.
        """
        if not word_anns:
            return []
        # every list now non-empty
        word_anns_dict = {ann.annotation_id: ann for ann in word_anns}
        # list of  (word index, text line, word annotation_id)
        word_order_list = OrderGenerator.group_words_into_lines(word_anns, image_id)
        number_rows = max(word[1] for word in word_order_list)
        if number_rows == 1 and not highest_level:
            return []
        detection_result_list = []
        for number_row in range(1, number_rows + 1):
            # list of  (word index, text line, word annotation_id) for text line equal to number_row
            ann_meta_per_row = [ann_meta for ann_meta in word_order_list if ann_meta[1] == number_row]
            ann_ids = [ann_meta[2] for ann_meta in ann_meta_per_row]
            anns_per_row = [word_anns_dict[ann_id] for ann_id in ann_ids]
            anns_per_row.sort(key=lambda x: x.get_bounding_box(image_id).ulx)

            if len(anns_per_row) < 2 or not self.make_sub_lines:
                # either row has only one word or all words should belong to one line
                boxes = [ann.get_bounding_box(image_id) for ann in anns_per_row]
                merge_box = merge_boxes(*boxes)
                detection_result = self._make_detect_result(
                    merge_box, {"child": [ann.annotation_id for ann in anns_per_row]}
                )
                detection_result_list.append(detection_result)
            else:
                for idx, ann in enumerate(anns_per_row):
                    if idx == 0:
                        sub_line = [ann]
                        sub_line_ann_ids = [ann.annotation_id]
                        continue

                    prev_box = anns_per_row[idx - 1].get_bounding_box(image_id)
                    current_box = ann.get_bounding_box(image_id)

                    if prev_box.absolute_coords:
                        prev_box = prev_box.transform(image_width, image_height)
                    if current_box.absolute_coords:
                        current_box = current_box.transform(image_width, image_height)

                    # If distance between boxes is lower than paragraph break, same sub-line
                    if current_box.ulx - prev_box.lrx < self.paragraph_break:  # type: ignore
                        sub_line.append(ann)
                        sub_line_ann_ids.append(ann.annotation_id)
                    else:
                        # We need to iterate maybe more than one time, because sub-lines may have more than one line
                        # if having been split. Take fore example a multi-column layout where a sub-line has
                        # two lines because of a column break and fonts twice as large as the other column.
                        detection_results = self.create_detection_result(
                            sub_line, image_width, image_height, image_id, False
                        )
                        if detection_results:
                            detection_result_list.extend(detection_results)
                        else:
                            boxes = [ann.get_bounding_box(image_id) for ann in sub_line]
                            merge_box = merge_boxes(*boxes)
                            detection_result = self._make_detect_result(merge_box, {"child": sub_line_ann_ids})
                            detection_result_list.append(detection_result)
                            sub_line = [ann]
                            sub_line_ann_ids = [ann.annotation_id]

                    if idx == len(anns_per_row) - 1:
                        detection_results = self.create_detection_result(
                            sub_line, image_width, image_height, image_id, False
                        )
                        if detection_results:
                            detection_result_list.extend(detection_results)
                        else:
                            boxes = [ann.get_bounding_box(image_id) for ann in sub_line]
                            merge_box = merge_boxes(*boxes)
                            detection_result = self._make_detect_result(merge_box, {"child": sub_line_ann_ids})
                            detection_result_list.append(detection_result)

        return detection_result_list


class TextLineServiceMixin(PipelineComponent, ABC):
    """
    This class is used to create text lines similar to `TextOrderService`.

    It uses the logic of the `TextOrderService` but modifies it to suit its needs. It specifically uses the
     `_create_lines_for_words` method and modifies the `serve` method.


    """

    def __init__(
        self,
        name: str,
        include_residual_text_container: bool = True,
        paragraph_break: Optional[float] = None,
    ):
        """
        Args:
            name: Name of the service.
            include_residual_text_container: Whether to include residual text containers.
            paragraph_break: Paragraph break threshold.
        """
        self.include_residual_text_container = include_residual_text_container
        self.text_line_generator = TextLineGenerator(self.include_residual_text_container, paragraph_break)
        super().__init__(name)

    def _create_lines_for_words(self, word_anns: Sequence[ImageAnnotation]) -> Sequence[ImageAnnotation]:
        """
        Creates lines for words using the `TextLineGenerator` instance.

        Args:
            word_anns: Sequence of `ImageAnnotation`.

        Returns:
            Sequence of `ImageAnnotation`.
        """
        detection_result_list = self.text_line_generator.create_detection_result(
            word_anns,
            self.dp_manager.datapoint.width,
            self.dp_manager.datapoint.height,
            self.dp_manager.datapoint.image_id,
        )
        line_anns = []
        for detect_result in detection_result_list:
            ann_id = self.dp_manager.set_image_annotation(detect_result)
            if ann_id:
                line_ann = self.dp_manager.get_annotation(ann_id)
                child_ann_id_list = detect_result.relationships["child"]  # type: ignore
                for child_ann_id in child_ann_id_list:
                    line_ann.dump_relationship(Relationships.CHILD, child_ann_id)
                line_anns.append(line_ann)
        return line_anns


class TextLineService(TextLineServiceMixin):
    """
    Some OCR systems do not identify lines of text but only provide text boxes for words.

    This is not sufficient for certain applications. This service determines rule-based text lines based on word boxes.
    One difficulty is that text lines are not continuous but are interrupted, for example, in multi-column layouts.
    These interruptions are taken into account insofar as the gap between two words on almost the same page height must
    not be too large.

    The service constructs new `ImageAnnotation` of the category `LayoutType.line` and forms relations between the text
    lines and the words contained in the text lines. The reading order is not arranged.


    """

    def __init__(self, paragraph_break: Optional[float] = None):
        """
        Args:
            paragraph_break: Threshold of two consecutive words. If distance is larger than threshold, two
                             sub-lines will be built.
        """
        super().__init__(
            name="text_line",
            include_residual_text_container=True,
            paragraph_break=paragraph_break,
        )

    def clone(self) -> TextLineService:
        """
        This method returns a new instance of the class with the same configuration.
        """
        return self.__class__(self.text_line_generator.paragraph_break)

    def serve(self, dp: Image) -> None:
        text_container_anns = dp.get_annotation(category_names=LayoutType.WORD)
        self._create_lines_for_words(text_container_anns)

    def get_meta_annotation(self) -> MetaAnnotation:
        """
        This method returns metadata about the annotations created by this pipeline component.
        """
        return MetaAnnotation(
            image_annotations=(LayoutType.LINE,),
            sub_categories={},
            relationships={LayoutType.LINE: {Relationships.CHILD}},
            summaries=(),
        )


@pipeline_component_registry.register("TextOrderService")
class TextOrderService(TextLineServiceMixin):
    """
    Reading order of words within floating text blocks as well as reading order of blocks within simple text blocks.

    To understand the difference between floating text blocks and simple text blocks, consider a page containing an
    article and a table. Table cells are text blocks that contain words which must be sorted. However, they do not
    belong to floating text that encircle a table. They are rather an element that is supposed to be read independently.

    A heuristic argument for its ordering is used where the underlying assumption is the reading order from left
    to right.

    - For the reading order within a text block, text containers (i.e., image annotations that contain character
      sub-annotations) are sorted based on their bounding box center and then lines are formed: Each word induces a new
      line, provided that its center is not in a line that has already been created by an already processed word. The
      entire block width is defined as the line width and the upper or lower line limit of the word bounding box as the
      upper or lower line limit. The reading order of the words is from left to right within a line. The reading order
      of the lines is from top to bottom.

    - For the reading order of text blocks within a page, the blocks are sorted using a similar procedure, with the
      difference that columns are formed instead of lines. Column lengths are defined as the length of the entire page
      and the left and right text block boundaries as the left and right column boundaries.

    A category annotation per word is generated, which fixes the order per word in the block, as well as a category
    annotation per block, which saves the reading order of the block per page.

    The blocks are defined in `text_block_categories` and text blocks that should be considered when generating
    narrative text must be added in `floating_text_block_categories`.

    Example:

        ```python
        order = TextOrderService(
            text_container="word",
            text_block_categories=["title", "text", "list", "cell", "head", "body"],
            floating_text_block_categories=["title", "text", "list"]
        )
        ```

    Note:
        The blocks are defined in `text_block_categories` and text blocks that should be considered when generating
        narrative text must be added in `floating_text_block_categories`.
    """

    def __init__(
        self,
        text_container: str,
        text_block_categories: Optional[Union[str, Sequence[TypeOrStr]]] = None,
        floating_text_block_categories: Optional[Union[str, Sequence[TypeOrStr]]] = None,
        include_residual_text_container: bool = True,
        starting_point_tolerance: float = 0.005,
        broken_line_tolerance: float = 0.003,
        height_tolerance: float = 2.0,
        paragraph_break: Optional[float] = 0.035,
    ):
        """
        Args:
            text_container: `Name` of an image annotation that has a CHARS sub-category. These annotations will be
                            ordered within all text blocks.
            text_block_categories: `Name` of image annotation that have a relation with text containers and where text
                                   containers need to be sorted. Defaults to `IMAGE_DEFAULTS["text_block_categories"]`.
            floating_text_block_categories: Name of image annotation that belong to floating text. These annotations
                                            form the highest hierarchy of text blocks that will be ordered to generate a
                                            narrative output of text. Defaults to
                                            `IMAGE_DEFAULTS["floating_text_block_categories"]`.
            include_residual_text_container: Text containers with no parent text block (e.g., not matched with any
                                             parent annotation in `MatchingService`) will not be assigned with a
                                             reading. (Reading order will only be assigned to image annotations that are
                                             `floating_text_block_categories` or text containers matched with text block
                                             annotations.) Setting `include_residual_text_container=True` will build
                                             synthetic text lines from text containers and regard these text lines as
                                             floating text blocks.
            starting_point_tolerance: Threshold to identify if two text blocks belong to one column. To check if two
                                      text blocks belong to the same column, one condition says that x-coordinates of
                                      vertices should not differ more than this threshold.
            broken_line_tolerance: Threshold to identify if two consecutive words belonging to one line should be in two
                                   different sub-lines (because they belong to two different text columns).
            height_tolerance: Threshold to identify if two columns lying over each other belong together or need to be
                              separated. Scaling factor of relative text block height.
            paragraph_break: Threshold of two consecutive words. If distance is larger than threshold, two sublines
                             will be built.
        """
        self.text_container = get_type(text_container)
        if isinstance(text_block_categories, (str, ObjectTypes)):
            text_block_categories = (get_type(text_block_categories),)
        if text_block_categories is None:
            text_block_categories = IMAGE_DEFAULTS.TEXT_BLOCK_CATEGORIES
        self.text_block_categories = tuple((get_type(category) for category in text_block_categories))
        if isinstance(floating_text_block_categories, (str, ObjectTypes)):
            floating_text_block_categories = (get_type(floating_text_block_categories),)
        if floating_text_block_categories is None:
            floating_text_block_categories = IMAGE_DEFAULTS.FLOATING_TEXT_BLOCK_CATEGORIES
        self.floating_text_block_categories = tuple((get_type(category) for category in floating_text_block_categories))
        if include_residual_text_container:
            self.floating_text_block_categories = self.floating_text_block_categories + (LayoutType.LINE,)
        self.include_residual_text_container = include_residual_text_container
        self.order_generator = OrderGenerator(starting_point_tolerance, broken_line_tolerance, height_tolerance)
        self.text_line_generator = TextLineGenerator(self.include_residual_text_container, paragraph_break)
        super().__init__(
            name="text_order",
            include_residual_text_container=include_residual_text_container,
            paragraph_break=paragraph_break,
        )
        self._init_sanity_checks()

    def serve(self, dp: Image) -> None:
        text_container_anns = dp.get_annotation(category_names=self.text_container)
        text_block_anns = dp.get_annotation(category_names=self.text_block_categories)
        if self.include_residual_text_container:
            mapped_text_container_ids = list(
                chain(*[text_block.get_relationship(Relationships.CHILD) for text_block in text_block_anns])
            )
            residual_text_container_anns = [
                ann for ann in text_container_anns if ann.annotation_id not in mapped_text_container_ids
            ]
            if self.text_container == LayoutType.WORD:
                text_block_anns.extend(self._create_lines_for_words(residual_text_container_anns))
            else:
                text_block_anns.extend(residual_text_container_anns)
        for text_block_ann in text_block_anns:
            self.order_text_in_text_block(text_block_ann)
        floating_text_block_anns = dp.get_annotation(category_names=self.floating_text_block_categories)
        self.order_blocks(floating_text_block_anns)
        self._create_columns()

    def _create_columns(self) -> None:
        if logger.isEnabledFor(DEBUG) and self.order_generator.columns_detect_result:
            for idx, detect_result in enumerate(self.order_generator.columns_detect_result):
                annotation_id = self.dp_manager.set_image_annotation(detect_result)
                if annotation_id:
                    self.dp_manager.set_category_annotation(
                        Relationships.READING_ORDER, idx, Relationships.READING_ORDER, annotation_id
                    )

    def order_text_in_text_block(self, text_block_ann: ImageAnnotation) -> None:
        """
        Orders text within a text block.

        It will take all child-like text containers (determined by a `MatchingOrderService`) from a block and order
        all items line-wise.

        Args:
            text_block_ann: Text block annotation (category one of `text_block_categories`).
        """
        text_container_ids = text_block_ann.get_relationship(Relationships.CHILD)
        text_container_ann = self.dp_manager.datapoint.get_annotation(
            annotation_ids=text_container_ids, category_names=self.text_container
        )
        if self.text_container == LayoutType.WORD:
            word_order_list = self.order_generator.group_words_into_lines(
                text_container_ann, self.dp_manager.datapoint.image_id
            )
        else:
            word_order_list = self.order_generator.group_lines_into_lines(
                text_container_ann, self.dp_manager.datapoint.image_id
            )
        for word_order in word_order_list:
            self.dp_manager.set_category_annotation(
                Relationships.READING_ORDER, word_order[0], Relationships.READING_ORDER, word_order[2]
            )

    def order_blocks(self, text_block_anns: list[ImageAnnotation]) -> None:
        """
        Orders text blocks using the internal order generator.

        Args:
            text_block_anns: List of `ImageAnnotation`.
        """
        block_order_list = self.order_generator.order_blocks(
            text_block_anns, self.dp_manager.datapoint.width, self.dp_manager.datapoint.height
        )
        for word_order in block_order_list:
            self.dp_manager.set_category_annotation(
                Relationships.READING_ORDER, word_order[0], Relationships.READING_ORDER, word_order[1]
            )

    def _init_sanity_checks(self) -> None:
        assert self.text_container in (LayoutType.WORD, LayoutType.LINE), (
            f"text_container must be either {LayoutType.WORD} or " f"{LayoutType.LINE}"
        )
        add_category = []
        if self.include_residual_text_container:
            add_category.append(LayoutType.LINE)

        if set(self.floating_text_block_categories) <= set(self.text_block_categories + tuple(add_category)):
            logger.warning(
                "In most cases floating_text_block_categories must be a subset of text_block_categories. "
                "Adding categories to floating_text_block_categories, that do not belong to "
                "text_block_categories makes only sense for categories set have CHILD relationships with"
                " annotations that belong to text_block_categories."
            )

    def get_meta_annotation(self) -> MetaAnnotation:
        add_category = [self.text_container]
        image_annotations: list[ObjectTypes] = []
        if self.include_residual_text_container and self.text_container == LayoutType.WORD:
            add_category.append(LayoutType.LINE)
            image_annotations.append(LayoutType.LINE)
        anns_with_reading_order = list(copy(self.floating_text_block_categories)) + add_category
        return MetaAnnotation(
            image_annotations=tuple(image_annotations),
            sub_categories={  # type: ignore
                category: {Relationships.READING_ORDER: {Relationships.READING_ORDER}}
                for category in anns_with_reading_order
            }
            | {self.text_container: {Relationships.READING_ORDER: {Relationships.READING_ORDER}}},
            relationships={},
            summaries=(),
        )

    def clone(self) -> TextOrderService:
        return self.__class__(
            self.text_container,
            self.text_block_categories,
            self.floating_text_block_categories,
            self.include_residual_text_container,
            self.order_generator.starting_point_tolerance,
            self.order_generator.broken_line_tolerance,
            self.order_generator.height_tolerance,
            self.text_line_generator.paragraph_break,
        )

    def clear_predictor(self) -> None:
        pass
