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

-factory build_analyzer for a given config

-user factory with a reduced config setting
"""

import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union

from ..extern.model import ModelCatalog, ModelDownloadManager
from ..extern.tessocr import TesseractOcrDetector
from ..extern.pdftext import PdfPlumberTextDetector
from ..extern.doctrocr import DoctrTextlineDetector, DoctrTextRecognizer
from ..extern.texocr import TextractOcrDetector
from ..pipe.base import PipelineComponent
from ..pipe.cell import DetectResultGenerator, SubImageLayoutService
from ..pipe.common import MatchingService, PageParsingService
from ..pipe.doctectionpipe import DoctectionPipe
from ..pipe.layout import ImageLayoutService
from ..pipe.order import TextOrderService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.segment import TableSegmentationService, PubtablesSegmentationService
from ..pipe.text import TextExtractionService
from ..utils.file_utils import pytorch_available, tensorpack_available, tf_available
from ..utils.fs import mkdir_p
from ..utils.logger import logger
from ..utils.metacfg import AttrDict, set_config_by_yaml
from ..utils.settings import LayoutType
from ..utils.systools import get_configs_dir_path, get_package_path
from ..utils.transform import PadTransform

if tf_available() and tensorpack_available():
    from tensorpack.utils.gpu import get_num_gpu  # pylint: disable=E0401

    from ..extern.tp.tfutils import disable_tp_layer_logging
    from ..extern.tpdetect import TPFrcnnDetector

if pytorch_available():
    from torch import cuda

    from ..extern.d2detect import D2FrcnnDetector
    from ..extern.hfdetr import HFDetrDerivedDetector


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

def _build_detector(cfg, mode) -> Union["D2FrcnnDetector", "TPFrcnnDetector", "HFDetrDerivedDetector"]:
    weights = getattr(cfg.TF,mode).WEIGHTS if cfg.LIB == "tf" else getattr(cfg.PT,mode).WEIGHTS
    filter_categories = getattr(getattr(cfg.TF,mode),"FILTER") if cfg.LIB == "tf" else getattr(getattr(cfg.PT,mode),"FILTER")
    config_path = ModelCatalog.get_full_path_configs(weights)
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    profile = ModelCatalog.get_profile(weights)
    categories = profile.categories
    assert categories is not None
    if profile.model_wrapper in ("TPFrcnnDetector",):
        return TPFrcnnDetector(config_path, weights_path, categories,
                               filter_categories=filter_categories)
    if profile.model_wrapper in ("D2FrcnnDetector",):
        return D2FrcnnDetector(config_path, weights_path, categories,
                               device=cfg.DEVICE,
                               filter_categories=filter_categories)
    if profile.model_wrapper in ("HFDetrDerivedDetector",):
        preprocessor_config = ModelCatalog.get_full_path_preprocessor_configs(weights)
        return HFDetrDerivedDetector(config_path,
                                            weights_path,
                                            preprocessor_config,
                                            categories,
                                            filter_categories=filter_categories)
    raise ValueError("You have chosen profile.model_wrapper: %s which is not allowed", profile.model_wrapper)

def _build_padder(cfg, mode):
    top, right, bottom, left = getattr(cfg.PT, mode).PAD.TOP, getattr(cfg.PT, mode).PAD.RIGHT, \
        getattr(cfg.PT, mode).PAD.BOTTOM, getattr(cfg.PT, mode).PAD.LEFT
    return PadTransform(top=top, right=right, bottom=bottom, left=left)

def _build_service(detector, cfg, mode):
    padder = None
    if detector.name == "HFDetrDerivedDetector":
        padder = _build_padder(cfg, mode)
    return ImageLayoutService(detector, to_image=True, crop_image=True, padder= padder)

def _build_sub_image_service(detector, mode):
    exclude_category_ids = []
    if mode == "ITEM":
        if detector.name == "HFDetrDerivedDetector":
            exclude_category_ids.extend(["1", "3", "4", "5", "6"])
    detect_result_generator = DetectResultGenerator(detector.categories,
                                                    exclude_category_ids=exclude_category_ids)
    return SubImageLayoutService(detector, LayoutType.table, None, detect_result_generator)

def _build_ocr(cfg):
    if cfg.OCR.USE_TESSERACT:
        ocr_config_path = get_configs_dir_path() / cfg.OCR.CONFIG.TESSERACT
        return TesseractOcrDetector(
            ocr_config_path, config_overwrite=[f"LANGUAGES={cfg.LANGUAGES}"] if cfg.LANGUAGES is not None else None
        )
    if cfg.OCR.USE_DOCTR:
        weights = cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.TF if cfg.LIB == "tf" else cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.PT
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        profile = ModelCatalog.get_profile(weights)
        return DoctrTextRecognizer(profile.architecture,weights_path,cfg.DEVICE)
    if cfg.OCR.USE_TEXTRACT:
        return TextractOcrDetector()
    raise ValueError("You have set USE_OCR=True but any of USE_TESSERACT, USE_DOCTR, USE_TEXTRACT is set to False")

def _build_doctr_word(cfg):
    weights = cfg.OCR.WEIGHTS.DOCTR_WORD.TF if cfg.LIB == "tf" else cfg.OCR.WEIGHTS.DOCTR_WORD.PT
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    profile = ModelCatalog.get_profile(weights)
    return DoctrTextlineDetector(profile.architecture, weights_path, profile.categories, cfg.DEVICE)

def build_analyzer(cfg: AttrDict) -> DoctectionPipe:
    """
    Builds the analyzer with a given config

    :param cfg: A configuration
    :return: Analyzer pipeline
    """
    pipe_component_list: List[PipelineComponent] = []

    if cfg.USE_LAYOUT:
         d_layout = _build_detector(cfg, "LAYOUT")
         layout = _build_service(d_layout, cfg, "LAYOUT")
         pipe_component_list.append(layout)

    # setup tables service
    if cfg.USE_TABLE_SEGMENTATION:
        d_item = _build_detector(cfg, "ITEM")
        item = _build_sub_image_service(d_item, "ITEM")
        pipe_component_list.append(item)

        d_cell = _build_detector(cfg, "CELL")
        cell = _build_sub_image_service(d_cell, "CELL")
        pipe_component_list.append(cell)

        if d_item.name == "HFDetrDerivedDetector":
            pubtables = PubtablesSegmentationService(cfg.SEGMENTATION.ASSIGNMENT_RULE,
                                                     cfg.SEGMENTATION.THRESHOLD_ROWS,
                                                     cfg.SEGMENTATION.THRESHOLD_COLS,
                                                     cfg.SEGMENTATION.FULL_TABLE_TILING,
                                                     cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                                                     cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                                                     cfg.SEGMENTATION.CELL_CATEGORY_ID,
                                                     stretch_rule=cfg.SEGMENTATION.STRETCH_RULE)
            pipe_component_list.append(pubtables)
        else:
            table_segmentation = TableSegmentationService(
                cfg.SEGMENTATION.ASSIGNMENT_RULE,
                cfg.SEGMENTATION.THRESHOLD_ROWS,
                cfg.SEGMENTATION.THRESHOLD_COLS,
                cfg.SEGMENTATION.FULL_TABLE_TILING,
                cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
            )
            pipe_component_list.append(table_segmentation)

            if cfg.USE_TABLE_REFINEMENT:
                table_segmentation_refinement = TableSegmentationRefinementService()
                pipe_component_list.append(table_segmentation_refinement)

    if cfg.USE_PDF_MINER:
        pdf_text = PdfPlumberTextDetector()
        d_text = TextExtractionService(pdf_text)
        pipe_component_list.append(d_text)

    # setup ocr
    if cfg.USE_OCR:
        # the extra mile for DocTr
        if cfg.OCR.USE_DOCTR:
            d_word = _build_doctr_word(cfg)
            word = ImageLayoutService(d_word, to_image=True, crop_image=True, skip_if_layout_extracted=True)
            pipe_component_list.append(word)

        ocr = _build_ocr(cfg)
        text = TextExtractionService(ocr, skip_if_text_extracted=True)
        pipe_component_list.append(text)

        match = MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=LayoutType.word,
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.THRESHOLD)
        pipe_component_list.append(match)

        order = TextOrderService(
            text_container=LayoutType.word,
            text_block_categories=cfg.TEXT_ORDERING.TEXT_BLOCK_CATEGORIES,
            floating_text_block_categories=cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            include_residual_text_container=cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
            starting_point_tolerance=cfg.TEXT_ORDERING.STARTING_POINT_TOLERANCE,
            broken_line_tolerance=cfg.TEXT_ORDERING.BROKEN_LINE_TOLERANCE,
            height_tolerance=cfg.TEXT_ORDERING.HEIGHT_TOLERANCE,
            paragraph_break=cfg.TEXT_ORDERING.PARAGRAPH_BREAK,
        )
        pipe_component_list.append(order)

    page_parsing_service = PageParsingService(text_container=LayoutType.word,
                                     floating_text_block_categories=cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
                                     include_residual_text_container=cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER)
    pipe = DoctectionPipe(pipeline_component_list=pipe_component_list,page_parsing_service=page_parsing_service)

    return pipe


def get_dd_analyzer(
    tables: bool = True, ocr: bool = True, table_refinement: bool = True, language: Optional[str] = None,
        reset_config_file: bool = False
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
    dd_one_config_path = _maybe_copy_config_to_cache(_DD_ONE, reset_config_file)
    _maybe_copy_config_to_cache(_TESSERACT)

    # Set up of the configuration and logging
    cfg = set_config_by_yaml(dd_one_config_path)

    cfg.freeze(freezed=False)
    cfg.LIB = lib
    cfg.DEVICE = device
    cfg.USE_TABLE_SEGMENTATION = tables
    cfg.USE_TABLE_REFINEMENT = table_refinement
    cfg.USE_OCR = ocr
    cfg.LANGUAGES = language
    cfg.freeze()

    logger.info("Config: \n %s", str(cfg), cfg.to_dict())

    # will silent all TP loggings while building the tower
    if tensorpack_available():
        disable_tp_layer_logging()

    return build_analyzer(cfg)
