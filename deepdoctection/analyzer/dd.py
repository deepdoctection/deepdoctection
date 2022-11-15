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
Module for **deep**doctection analyzer.
"""

import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union

from ..extern.model import ModelCatalog, ModelDownloadManager
from ..extern.tessocr import TesseractOcrDetector
from ..pipe.base import PipelineComponent
from ..pipe.cell import SubImageLayoutService
from ..pipe.common import MatchingService
from ..pipe.doctectionpipe import DoctectionPipe
from ..pipe.layout import ImageLayoutService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.segment import TableSegmentationService
from ..pipe.text import TextExtractionService, TextOrderService
from ..utils.file_utils import pytorch_available, tensorpack_available, tf_available
from ..utils.fs import mkdir_p
from ..utils.logger import logger
from ..utils.metacfg import AttrDict, set_config_by_yaml
from ..utils.settings import CellType, LayoutType
from ..utils.systools import get_configs_dir_path, get_package_path

if tf_available() and tensorpack_available():
    from tensorpack.utils.gpu import get_num_gpu  # pylint: disable=E0401

    from ..extern.tp.tfutils import disable_tp_layer_logging
    from ..extern.tpdetect import TPFrcnnDetector

if pytorch_available():
    from torch import cuda

    from ..extern.d2detect import D2FrcnnDetector


__all__ = ["get_dd_analyzer", "build_analyzer"]

_DD_ONE = "deepdoctection/configs/conf_dd_one.yaml"
_TESSERACT = "deepdoctection/configs/conf_tesseract.yaml"


def _auto_select_lib_and_device() -> Tuple[str, str]:
    """
    Select the DL library and subsequently the device. In summary:

    If TF is available, use TF unless a GPU is not available, in which case choose PT. If CUDA is not available and PT
    is not installed raise ImportError.
    """
    if tf_available() and tensorpack_available():
        if get_num_gpu() >= 1:
            return "tf", "cuda"
        if pytorch_available():
            return "pt", "cpu"
        raise ModuleNotFoundError("Install Pytorch and Torchvision to run with a CPU")
    if pytorch_available():
        if cuda.is_available():
            return "pt", "gpu"
        return "pt", "cpu"
    raise ModuleNotFoundError("Install Tensorflow or Pytorch before building analyzer")


def _maybe_copy_config_to_cache(file_name: str, force_copy: bool = True) -> str:
    """
    Initial copying of config file from the package dir into the config cache.

    :return: path to the copied file_name
    """

    absolute_path_source = os.path.join(get_package_path(), file_name)
    absolute_path = os.path.join(get_configs_dir_path(), os.path.join("dd", os.path.split(file_name)[1]))
    mkdir_p(os.path.split(absolute_path)[0])
    if not os.path.isfile(absolute_path) or force_copy:
        copyfile(absolute_path_source, absolute_path)
    return absolute_path


def build_analyzer(cfg: AttrDict) -> DoctectionPipe:
    """
    Builds the analyzer with a given config

    :param cfg: A configuration
    :return: Analyzer pipeline
    """
    pipe_component_list: List[PipelineComponent] = []

    d_layout: Union[D2FrcnnDetector, TPFrcnnDetector]
    if cfg.LIB == "tf":
        layout_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.TPLAYOUT)
        layout_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.TPLAYOUT)
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.TPLAYOUT)
        categories_layout = profile.categories
        assert categories_layout is not None
        assert layout_weights_path is not None
        d_layout = TPFrcnnDetector(layout_config_path, layout_weights_path, categories_layout)
    else:
        layout_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2LAYOUT)
        layout_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2LAYOUT)
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2LAYOUT)
        categories_layout = profile.categories
        assert categories_layout is not None
        assert layout_weights_path is not None
        d_layout = D2FrcnnDetector(layout_config_path, layout_weights_path, categories_layout, device=cfg.DEVICE)
    layout = ImageLayoutService(d_layout, to_image=True, crop_image=True)
    pipe_component_list.append(layout)

    # setup tables service
    if cfg.TAB:
        d_cell: Optional[Union[D2FrcnnDetector, TPFrcnnDetector]]
        d_item: Union[D2FrcnnDetector, TPFrcnnDetector]
        if cfg.LIB == "tf":
            cell_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.TPCELL)
            cell_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.TPCELL)
            profile = ModelCatalog.get_profile(cfg.WEIGHTS.TPCELL)
            categories_cell = profile.categories
            assert categories_cell is not None
            d_cell = TPFrcnnDetector(
                cell_config_path,
                cell_weights_path,
                categories_cell,
            )
            item_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.TPITEM)
            item_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.TPITEM)
            profile = ModelCatalog.get_profile(cfg.WEIGHTS.TPITEM)
            categories_item = profile.categories
            assert categories_item is not None
            d_item = TPFrcnnDetector(item_config_path, item_weights_path, categories_item)
        else:
            cell_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2CELL)
            cell_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2CELL)
            profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2CELL)
            categories_cell = profile.categories
            assert categories_cell is not None
            d_cell = D2FrcnnDetector(cell_config_path, cell_weights_path, categories_cell, device=cfg.DEVICE)
            item_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2ITEM)
            item_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2ITEM)
            profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2ITEM)
            categories_item = profile.categories
            assert categories_item is not None
            d_item = D2FrcnnDetector(item_config_path, item_weights_path, categories_item, device=cfg.DEVICE)

        cell = SubImageLayoutService(d_cell, LayoutType.table, {1: 6}, True)
        pipe_component_list.append(cell)

        item = SubImageLayoutService(d_item, LayoutType.table, {1: 7, 2: 8}, True)
        pipe_component_list.append(item)

        table_segmentation = TableSegmentationService(
            cfg.SEGMENTATION.ASSIGNMENT_RULE,
            cfg.SEGMENTATION.IOU_THRESHOLD_ROWS
            if cfg.SEGMENTATION.ASSIGNMENT_RULE in ["iou"]
            else cfg.SEGMENTATION.IOA_THRESHOLD_ROWS,
            cfg.SEGMENTATION.IOU_THRESHOLD_COLS
            if cfg.SEGMENTATION.ASSIGNMENT_RULE in ["iou"]
            else cfg.SEGMENTATION.IOA_THRESHOLD_COLS,
            cfg.SEGMENTATION.FULL_TABLE_TILING,
            cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
            cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
        )
        pipe_component_list.append(table_segmentation)

        if cfg.TAB_REF:
            table_segmentation_refinement = TableSegmentationRefinementService()
            pipe_component_list.append(table_segmentation_refinement)

    # setup ocr
    if cfg.OCR:

        tess_ocr_config_path = get_configs_dir_path() / cfg.CONFIG.TESS_OCR
        d_tess_ocr = TesseractOcrDetector(
            tess_ocr_config_path, config_overwrite=[f"LANGUAGES={cfg.LANG}"] if cfg.LANG is not None else None
        )
        text = TextExtractionService(d_tess_ocr)
        pipe_component_list.append(text)

        match = MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=LayoutType.word,
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.IOU_THRESHOLD
            if cfg.WORD_MATCHING.RULE in ["iou"]
            else cfg.WORD_MATCHING.IOA_THRESHOLD,
        )
        pipe_component_list.append(match)

        order = TextOrderService(
            text_container=LayoutType.word,
            floating_text_block_names=[LayoutType.title, LayoutType.text, LayoutType.list],
            text_block_names=[
                LayoutType.title,
                LayoutType.text,
                LayoutType.list,
                LayoutType.cell,
                CellType.header,
                CellType.body,
            ],
        )
        pipe_component_list.append(order)

    pipe = DoctectionPipe(pipeline_component_list=pipe_component_list)

    return pipe


def get_dd_analyzer(
    tables: bool = True, ocr: bool = True, table_refinement: bool = True, language: Optional[str] = None
) -> DoctectionPipe:
    """
    Factory function for creating the built-in **deep**doctection analyzer.

    The Standard Analyzer is a pipeline that comprises the following analysis components:

    - Document analysis with object recognition and classification of:

        * title
        * text
        * list
        * table
        * figure

    - Table recognition including line and column segmentation as well as detection of cells that run over several
      rows or columns.

    - OCR using Tesseract as well as text assignment to the document assignment.

    - Determination of the reading order for complex structured documents.

    You can optionally switch off table recognition and ocr related components.

    :param tables: Will do full table recognition. Default set to True
    :param table_refinement: Will rearrange cells such that generating html is possible
    :param ocr: Will do ocr, matching with layout and ordering words. Default set to True
    :param language: Select a specific language. Pre-selecting layout will increase ocr precision.

    :return: A DoctectionPipe instance with the given configs
    """

    lib, device = _auto_select_lib_and_device()
    dd_one_config_path = _maybe_copy_config_to_cache(_DD_ONE)
    _maybe_copy_config_to_cache(_TESSERACT)

    # Set up of the configuration and logging
    cfg = set_config_by_yaml(dd_one_config_path)

    cfg.freeze(freezed=False)
    cfg.LIB = lib
    cfg.DEVICE = device
    cfg.TAB = tables
    cfg.TAB_REF = table_refinement
    cfg.OCR = ocr
    cfg.LANG = language
    cfg.freeze()

    logger.info("Config: \n %s", str(cfg), cfg.to_dict())

    # will silent all TP loggings while building the tower
    if tensorpack_available():
        disable_tp_layer_logging()

    return build_analyzer(cfg)
