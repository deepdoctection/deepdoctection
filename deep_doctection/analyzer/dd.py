# -*- coding: utf-8 -*-
# File: dd.py

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
Module for Deep-Doctection Analyzer
"""

import os
from typing import Optional, List, Union


from ..pipe.base import PipelineComponent, PredictorPipelineComponent
from ..pipe.cell import SubImageLayoutService
from ..pipe.common import MatchingService
from ..pipe.layout import ImageLayoutService
from ..pipe.segment import TableSegmentationService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.text import TextExtractionService, TextOrderService
from ..pipe.doctectionpipe import DoctectionPipe
from ..extern.tpdetect import TPFrcnnDetector
from ..extern.tessocr import TesseractOcrDetector
from ..extern.model import ModelDownloadManager
from ..utils.metacfg import set_config_by_yaml
from ..utils.settings import names
from ..utils.systools import get_package_path
from ..utils.logger import logger

_DD_ONE = "configs/dd/conf_dd_one.yaml"


def get_dd_analyzer(
        tables: bool = True, ocr: bool = True, table_refinement: bool = True, language: Optional[str] = None
) -> DoctectionPipe:
    """
    Factory function for creating the built-in **Deep-Doctection Analyzer**.

    The Standard Analyzer is a pipeline that comprises the following analysis components:

    - Document analysis with object recognition and classification of:

        * TITLE
        * TEXT
        * LIST
        * TABLE
        * FIGURE

    - Table recognition including line and column segmentation as well as detection of cells that run over several
      rows or columns.

    - OCR using Tesseract as well as text assignment to the document assignment.

    - Determination of the reading order for complex structured documents.

    You can optionally switch off table recognition and ocr related components.

    :param tables: Will do full table recognition. Default set to True
    :param table_refinement: Will rearrange cells such that generating html is possible
    :param ocr: Will do ocr, matching with layout and ordering words. Default set to True
    :param language: Select a specific language. Pre-selecting layout will increase ocr precision.
    """

    logger.info("Will establish with table: %s and ocr: %s", tables, ocr)
    # setup path
    p_path = get_package_path()
    cfg = set_config_by_yaml(os.path.join(p_path, _DD_ONE))
    logger.info("Deep Doctection Analyzer Config: ------------------------------------------\n %s", str(cfg))
    pipe_component_list: List[Union[PipelineComponent, PredictorPipelineComponent]] = []

    # setup layout
    categories_layout = {"1": names.C.TEXT, "2": names.C.TITLE, "3": names.C.LIST, "4": names.C.TAB, "5": names.C.FIG}
    layout_config_path = os.path.join(p_path, cfg.CONFIG.TPLAYOUT)
    layout_weights_path = ModelDownloadManager.maybe_download_weights(cfg.WEIGHTS.TPLAYOUT)
    assert layout_weights_path is not None
    d_layout = TPFrcnnDetector(layout_config_path, layout_weights_path, categories_layout)
    layout = ImageLayoutService(d_layout, to_image=True, crop_image=True)
    pipe_component_list.append(layout)

    # setup tables
    if tables:
        categories_cell = {"1": names.C.CELL}
        cell_config_path = os.path.join(p_path, cfg.CONFIG.TPCELL)
        cell_weights_path = ModelDownloadManager.maybe_download_weights(cfg.WEIGHTS.TPCELL)
        d_cell = TPFrcnnDetector(
            cell_config_path,
            cell_weights_path,
            categories_cell,
        )
        cell = SubImageLayoutService(d_cell, names.C.TAB, {1: 6}, True)
        pipe_component_list.append(cell)

        categories_item = {"1": names.C.ROW, "2": names.C.COL}
        item_config_path = os.path.join(p_path, cfg.CONFIG.TPITEM)
        item_weights_path = ModelDownloadManager.maybe_download_weights(cfg.WEIGHTS.TPITEM)
        d_item = TPFrcnnDetector(item_config_path, item_weights_path, categories_item)
        item = SubImageLayoutService(d_item, names.C.TAB, {1: 7, 2: 8}, True)
        pipe_component_list.append(item)

        table_segmentation = TableSegmentationService(
            cfg.SEGMENTATION.ASSIGNMENT_RULE,
            cfg.SEGMENTATION.IOU_THRESHOLD_ROWS,
            cfg.SEGMENTATION.IOU_THRESHOLD_COLS,
            cfg.SEGMENTATION.IOA_THRESHOLD_ROWS,
            cfg.SEGMENTATION.IOA_THRESHOLD_COLS,
            cfg.SEGMENTATION.FULL_TABLE_TILING,
            cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
            cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
        )
        pipe_component_list.append(table_segmentation)

        if table_refinement:
            table_segmentation_refinement = TableSegmentationRefinementService()
            pipe_component_list.append(table_segmentation_refinement)

    # setup ocr
    if ocr:
        tess_ocr_config_path = os.path.join(p_path, cfg.CONFIG.TESS_OCR)
        d_tess_ocr = TesseractOcrDetector(
            tess_ocr_config_path, config_overwrite=[f"LANGUAGES={language}"] if language is not None else None
        )
        text = TextExtractionService(d_tess_ocr, None, {1: 9})
        pipe_component_list.append(text)

        match = MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=names.C.WORD,
            matching_rule=cfg.WORD_MATCHING.RULE,
            iou_threshold=cfg.WORD_MATCHING.IOU_THRESHOLD,
            ioa_threshold=cfg.WORD_MATCHING.IOA_THRESHOLD,
        )
        pipe_component_list.append(match)

        order = TextOrderService()
        pipe_component_list.append(order)

    pipe = DoctectionPipe(pipeline_component_list=pipe_component_list)

    return pipe