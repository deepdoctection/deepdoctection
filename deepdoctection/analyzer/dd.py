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

import ast
import os
from os import environ
from shutil import copyfile
from typing import List, Optional, Union

from ..extern.base import ObjectDetector
from ..extern.doctrocr import DoctrTextlineDetector, DoctrTextRecognizer
from ..extern.model import ModelCatalog, ModelDownloadManager
from ..extern.pdftext import PdfPlumberTextDetector
from ..extern.tessocr import TesseractOcrDetector
from ..extern.texocr import TextractOcrDetector
from ..pipe.base import PipelineComponent
from ..pipe.cell import DetectResultGenerator, SubImageLayoutService
from ..pipe.common import AnnotationNmsService, MatchingService, PageParsingService
from ..pipe.doctectionpipe import DoctectionPipe
from ..pipe.layout import ImageLayoutService
from ..pipe.order import TextOrderService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.segment import PubtablesSegmentationService, TableSegmentationService
from ..pipe.text import TextExtractionService
from ..pipe.text_refine import TextRefinementService
from ..utils.detection_types import Pathlike
from ..utils.env_info import get_device
from ..utils.file_utils import (
    boto3_available,
    detectron2_available,
    pytorch_available,
    tensorpack_available,
    tf_available,
)
from ..utils.fs import get_configs_dir_path, get_package_path, mkdir_p
from ..utils.logger import LoggingRecord, logger
from ..utils.metacfg import AttrDict, set_config_by_yaml
from ..utils.settings import CellType, LayoutType
from ..utils.transform import PadTransform

if tf_available() and tensorpack_available():
    from ..extern.tp.tfutils import disable_tp_layer_logging
    from ..extern.tpdetect import TPFrcnnDetector

if pytorch_available():
    from ..extern.d2detect import D2FrcnnDetector, D2FrcnnTracingDetector
    from ..extern.hfdetr import HFDetrDerivedDetector

if boto3_available():
    from botocore.config import Config  # type: ignore


__all__ = [
    "maybe_copy_config_to_cache",
    "config_sanity_checks",
    "build_detector",
    "build_padder",
    "build_service",
    "build_sub_image_service",
    "build_ocr",
    "build_doctr_word",
    "get_dd_analyzer",
    "build_analyzer",
]

_DD_ONE = "deepdoctection/configs/conf_dd_one.yaml"
_TESSERACT = "deepdoctection/configs/conf_tesseract.yaml"


def maybe_copy_config_to_cache(
    package_path: Pathlike, configs_dir_path: Pathlike, file_name: str, force_copy: bool = True
) -> str:
    """
    Initial copying of various files
    :param package_path: base path to directory of source file `file_name`
    :param configs_dir_path: base path to target directory
    :param file_name: file to copy
    :param force_copy: If file is already in target directory, will re-copy the file

    :return: path to the copied file_name
    """

    absolute_path_source = os.path.join(package_path, file_name)
    absolute_path = os.path.join(configs_dir_path, os.path.join("dd", os.path.split(file_name)[1]))
    mkdir_p(os.path.split(absolute_path)[0])
    if not os.path.isfile(absolute_path) or force_copy:
        copyfile(absolute_path_source, absolute_path)
    return absolute_path


def config_sanity_checks(cfg: AttrDict) -> None:
    """Some config sanity checks"""
    
    if cfg.USE_PDF_MINER and cfg.USE_OCR and cfg.OCR.USE_DOCTR:
        raise ValueError("Configuration USE_PDF_MINER= True and USE_OCR=True and USE_DOCTR=True is not allowed")
    if cfg.USE_OCR:
        if cfg.OCR.USE_TESSERACT + cfg.OCR.USE_DOCTR + cfg.OCR.USE_TEXTRACT != 1:
            raise ValueError(
                "Choose either OCR.USE_TESSERACT=True or OCR.USE_DOCTR=True or OCR.USE_TEXTRACT=True "
                "and set the other two to False. Only one OCR system can be activated."
            )

def build_detector(
    cfg: AttrDict, mode: str
) -> Union["D2FrcnnDetector", "TPFrcnnDetector", "HFDetrDerivedDetector", "D2FrcnnTracingDetector"]:
    """Building a D2-Detector, a TP-Detector as Detr-Detector or a D2-Torch Tracing Detector according to
    the config

    :param cfg: Config
    :param mode: either `LAYOUT`,`CELL` or `ITEM`
    """
    weights = (
        getattr(cfg.TF, mode).WEIGHTS
        if cfg.LIB == "TF"
        else (getattr(cfg.PT, mode).WEIGHTS if detectron2_available() else getattr(cfg.PT, mode).WEIGHTS_TS)
    )
    filter_categories = (
        getattr(getattr(cfg.TF, mode), "FILTER") if cfg.LIB == "TF" else getattr(getattr(cfg.PT, mode), "FILTER")
    )
    config_path = ModelCatalog.get_full_path_configs(weights)
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    profile = ModelCatalog.get_profile(weights)
    categories = profile.categories
    assert categories is not None
    if profile.model_wrapper in ("TPFrcnnDetector",):
        return TPFrcnnDetector(config_path, weights_path, categories, filter_categories=filter_categories)
    if profile.model_wrapper in ("D2FrcnnDetector",):
        return D2FrcnnDetector(
            config_path, weights_path, categories, device=cfg.DEVICE, filter_categories=filter_categories
        )
    if profile.model_wrapper in ("D2FrcnnTracingDetector",):
        return D2FrcnnTracingDetector(config_path, weights_path, categories, filter_categories=filter_categories)
    if profile.model_wrapper in ("HFDetrDerivedDetector",):
        preprocessor_config = ModelCatalog.get_full_path_preprocessor_configs(weights)
        return HFDetrDerivedDetector(
            config_path,
            weights_path,
            preprocessor_config,
            categories,
            device=cfg.DEVICE,
            filter_categories=filter_categories,
        )
    raise TypeError(
        f"You have chosen profile.model_wrapper: {profile.model_wrapper} which is not allowed. Please check "
        f"compatability with your deep learning framework"
    )


def build_padder(cfg: AttrDict, mode: str) -> PadTransform:
    """Building a padder according to the config

    :param cfg: Config
    :param mode: either `LAYOUT`,`CELL` or `ITEM`
    :return `PadTransform` instance
    """
    top, right, bottom, left = (
        getattr(cfg.PT, mode).PAD.TOP,
        getattr(cfg.PT, mode).PAD.RIGHT,
        getattr(cfg.PT, mode).PAD.BOTTOM,
        getattr(cfg.PT, mode).PAD.LEFT,
    )
    return PadTransform(top=top, right=right, bottom=bottom, left=left)


def build_service(detector: ObjectDetector, cfg: AttrDict, mode: str) -> ImageLayoutService:
    """Building a layout service with a given detector

    :param detector: will be passed to the `ImageLayoutService`
    :param cfg: Configuration
    :param mode: either `LAYOUT`,`CELL` or `ITEM`
    :return `ImageLayoutService` instance
    """
    padder = None
    if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
        padder = build_padder(cfg, mode)
    return ImageLayoutService(detector, to_image=True, crop_image=True, padder=padder)


def build_sub_image_service(detector: ObjectDetector, cfg: AttrDict, mode: str) -> SubImageLayoutService:
    """
    Building a sub image layout service with a given detector

    :param detector: will be passed to the `SubImageLayoutService`
    :param cfg: Configuration
    :param mode: either `LAYOUT`,`CELL` or `ITEM`
    :return: `SubImageLayoutService` instance
    """
    exclude_category_ids = []
    padder = None
    if mode == "ITEM":
        if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
            exclude_category_ids.extend(["1", "3", "4", "5", "6"])
            padder = build_padder(cfg, mode)
    detect_result_generator = DetectResultGenerator(detector.categories, exclude_category_ids=exclude_category_ids)
    return SubImageLayoutService(
        detector, [LayoutType.table, LayoutType.table_rotated], None, detect_result_generator, padder
    )


def build_ocr(cfg: AttrDict) -> Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]:
    """
    Building OCR predictor
    :param cfg: Config
    """
    if cfg.OCR.USE_TESSERACT:
        ocr_config_path = get_configs_dir_path() / cfg.OCR.CONFIG.TESSERACT
        return TesseractOcrDetector(
            ocr_config_path, config_overwrite=[f"LANGUAGES={cfg.LANGUAGE}"] if cfg.LANGUAGE is not None else None
        )
    if cfg.OCR.USE_DOCTR:
        weights = cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.TF if cfg.LIB == "TF" else cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.PT
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        profile = ModelCatalog.get_profile(weights)
        # get_full_path_configs will complete the path even if the model is not registered
        config_path = ModelCatalog.get_full_path_configs(weights) if profile.config is not None else None
        if profile.architecture is None:
            raise ValueError("model profile.architecture must be specified")
        return DoctrTextRecognizer(
            profile.architecture, weights_path, cfg.DEVICE, lib=cfg.LIB, path_config_json=config_path
        )
    if cfg.OCR.USE_TEXTRACT:
        credentials_kwargs = {
            "aws_access_key_id": environ.get("ACCESS_KEY"),
            "aws_secret_access_key": environ.get("SECRET_KEY"),
            "config": Config(region_name=environ.get("REGION")),
        }
        return TextractOcrDetector(**credentials_kwargs)
    raise ValueError("You have set USE_OCR=True but any of USE_TESSERACT, USE_DOCTR, USE_TEXTRACT is set to False")


def build_doctr_word(cfg: AttrDict) -> DoctrTextlineDetector:
    """Building `DoctrTextlineDetector` instance"""
    weights = cfg.OCR.WEIGHTS.DOCTR_WORD.TF if cfg.LIB == "TF" else cfg.OCR.WEIGHTS.DOCTR_WORD.PT
    weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
    profile = ModelCatalog.get_profile(weights)
    if profile.architecture is None:
        raise ValueError("model profile.architecture must be specified")
    if profile.categories is None:
        raise ValueError("model profile.categories must be specified")
    return DoctrTextlineDetector(profile.architecture, weights_path, profile.categories, cfg.DEVICE, lib=cfg.LIB)


def build_analyzer(cfg: AttrDict) -> DoctectionPipe:
    """
    Builds the analyzer with a given config

    :param cfg: A configuration
    :return: Analyzer pipeline
    """
    pipe_component_list: List[PipelineComponent] = []

    if cfg.USE_LAYOUT:
        d_layout = build_detector(cfg, "LAYOUT")
        layout = build_service(d_layout, cfg, "LAYOUT")
        pipe_component_list.append(layout)

    # setup layout nms service
    if cfg.LAYOUT_NMS_PAIRS.COMBINATIONS and cfg.USE_LAYOUT:
        if not detectron2_available() and cfg.LIB == "PT":
            raise ModuleNotFoundError("LAYOUT_NMS_PAIRS is only available for detectron2")
        if not isinstance(cfg.LAYOUT_NMS_PAIRS.COMBINATIONS, list) and not isinstance(
            cfg.LAYOUT_NMS_PAIRS.COMBINATIONS[0], list
        ):
            raise ValueError("LAYOUT_NMS_PAIRS mus be a list of lists")
        layout_nms_serivce = AnnotationNmsService(
            cfg.LAYOUT_NMS_PAIRS.COMBINATIONS, cfg.LAYOUT_NMS_PAIRS.THRESHOLDS, cfg.LAYOUT_NMS_PAIRS.PRIORITY
        )
        pipe_component_list.append(layout_nms_serivce)

    # setup tables service
    if cfg.USE_TABLE_SEGMENTATION:
        d_item = build_detector(cfg, "ITEM")
        item = build_sub_image_service(d_item, cfg, "ITEM")
        pipe_component_list.append(item)

        if d_item.__class__.__name__ not in ("HFDetrDerivedDetector",):
            d_cell = build_detector(cfg, "CELL")
            cell = build_sub_image_service(d_cell, cfg, "CELL")
            pipe_component_list.append(cell)

        if d_item.__class__.__name__ in ("HFDetrDerivedDetector",):
            pubtables = PubtablesSegmentationService(
                cfg.SEGMENTATION.ASSIGNMENT_RULE,
                cfg.SEGMENTATION.THRESHOLD_ROWS,
                cfg.SEGMENTATION.THRESHOLD_COLS,
                cfg.SEGMENTATION.FULL_TABLE_TILING,
                cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                cfg.SEGMENTATION.CELL_CATEGORY_ID,
                LayoutType.table,
                [
                    CellType.spanning,
                    CellType.row_header,
                    CellType.column_header,
                    CellType.projected_row_header,
                    LayoutType.cell,
                ],
                [
                    CellType.spanning,
                    CellType.row_header,
                    CellType.column_header,
                    CellType.projected_row_header,
                ],
                [LayoutType.row, LayoutType.column],
                [CellType.row_number, CellType.column_number],
                stretch_rule=cfg.SEGMENTATION.STRETCH_RULE,
            )
            pipe_component_list.append(pubtables)
        else:
            table_segmentation = TableSegmentationService(
                cfg.SEGMENTATION.ASSIGNMENT_RULE,
                cfg.SEGMENTATION.THRESHOLD_ROWS,
                cfg.SEGMENTATION.THRESHOLD_COLS,
                cfg.SEGMENTATION.FULL_TABLE_TILING,
                cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                LayoutType.table,
                [CellType.header, CellType.body, LayoutType.cell],
                [LayoutType.row, LayoutType.column],
                [CellType.row_number, CellType.column_number],
                cfg.SEGMENTATION.STRETCH_RULE,
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
            d_word = build_doctr_word(cfg)
            word = ImageLayoutService(d_word, to_image=True, crop_image=True, skip_if_layout_extracted=True)
            pipe_component_list.append(word)

        ocr = build_ocr(cfg)
        skip_if_text_extracted = cfg.USE_PDF_MINER
        extract_from_roi = LayoutType.word if cfg.OCR.USE_DOCTR else None
        text = TextExtractionService(
            ocr, skip_if_text_extracted=skip_if_text_extracted, extract_from_roi=extract_from_roi
        )
        pipe_component_list.append(text)

    if cfg.USE_PDF_MINER or cfg.USE_OCR:
        match = MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=LayoutType.word,
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.THRESHOLD,
            max_parent_only=cfg.WORD_MATCHING.MAX_PARENT_ONLY,
        )
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
                
        # Text Refinement Service 
        if cfg.USE_TEXT_REFINEMENT:
            categories_to_refine = [LayoutType.text, LayoutType.title] 
            text_refine = TextRefinementService(
                use_spellcheck_refinement=cfg.TEXT_REFINEMENT.USE_SPELLCHECKER_REFINEMENT,
                use_nlp_refinement=cfg.TEXT_REFINEMENT.USE_NLP_REFINEMENT,
                text_refinement_threshold=cfg.TEXT_REFINEMENT.TEXT_REFINEMENT_THRESHOLD,
                nlp_refinement_model_name=cfg.TEXT_REFINEMENT.NLP_REFINEMENT.MLM_MODEL, # TODO: Add support for custom models
                categories_to_refine=categories_to_refine) 
            
            pipe_component_list.append(text_refine)

    page_parsing_service = PageParsingService(
        text_container=LayoutType.word,
        floating_text_block_categories=cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
        include_residual_text_container=cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
    )
    pipe = DoctectionPipe(pipeline_component_list=pipe_component_list, page_parsing_service=page_parsing_service)

    return pipe


def get_dd_analyzer(
    reset_config_file: bool = False,
    config_overwrite: Optional[List[str]] = None,
    path_config_file: Optional[Pathlike] = None,
) -> DoctectionPipe:
    """
    Factory function for creating the built-in **deep**doctection analyzer.

    The Standard Analyzer is a pipeline that comprises the following analysis components:

    - Document layout analysis

    - Table segmentation

    - Text extraction/OCR

    - Reading order

    We refer to the various notebooks and docs for running an analyzer and changing the configs.

    :param reset_config_file: This will copy the `.yaml` file with default variables to the `.cache` and therefore
                              resetting all configurations if set to `True`.
    :param config_overwrite: Passing a list of string arguments and values to overwrite the `.yaml` configuration with
                             highest priority, e.g. ["USE_TABLE_SEGMENTATION=False",
                                                     "USE_OCR=False",
                                                     "TF.LAYOUT.WEIGHTS=my_fancy_pytorch_model"]
    :param path_config_file: Path to a custom config file. Can be outside of the .cache directory.
    :return: A DoctectionPipe instance with given configs
    """
    config_overwrite = [] if config_overwrite is None else config_overwrite
    lib = "TF" if ast.literal_eval(os.environ.get("USE_TENSORFLOW", "False")) else "PT"
    device = get_device(False)
    dd_one_config_path = maybe_copy_config_to_cache(
        get_package_path(), get_configs_dir_path(), _DD_ONE, reset_config_file
    )
    maybe_copy_config_to_cache(get_package_path(), get_configs_dir_path(), _TESSERACT)

    # Set up of the configuration and logging
    cfg = set_config_by_yaml(dd_one_config_path if not path_config_file else path_config_file)

    cfg.freeze(freezed=False)
    cfg.LANGUAGE = None
    cfg.LIB = lib
    cfg.DEVICE = device
    cfg.freeze()

    if config_overwrite:
        cfg.update_args(config_overwrite)

    config_sanity_checks(cfg)
    logger.info(LoggingRecord(f"Config: \n {str(cfg)}", cfg.to_dict()))  # type: ignore

    # will silent all TP logging while building the tower
    if tensorpack_available():
        disable_tp_layer_logging()

    return build_analyzer(cfg)
