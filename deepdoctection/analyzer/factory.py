# -*- coding: utf-8 -*-
# File: factory.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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

from os import environ
from typing import Union

from lazy_imports import try_import

from ..extern.base import ImageTransformer, ObjectDetector, PdfMiner
from ..extern.d2detect import D2FrcnnDetector, D2FrcnnTracingDetector
from ..extern.doctrocr import DoctrTextlineDetector, DoctrTextRecognizer
from ..extern.hfdetr import HFDetrDerivedDetector
from ..extern.model import ModelCatalog, ModelDownloadManager
from ..extern.pdftext import PdfPlumberTextDetector
from ..extern.tessocr import TesseractOcrDetector, TesseractRotationTransformer
from ..extern.texocr import TextractOcrDetector
from ..extern.tpdetect import TPFrcnnDetector
from ..pipe.base import PipelineComponent
from ..pipe.common import AnnotationNmsService, IntersectionMatcher, MatchingService, PageParsingService
from ..pipe.doctectionpipe import DoctectionPipe
from ..pipe.layout import ImageLayoutService
from ..pipe.order import TextOrderService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.segment import PubtablesSegmentationService, TableSegmentationService
from ..pipe.sub_layout import DetectResultGenerator, SubImageLayoutService
from ..pipe.text import TextExtractionService
from ..pipe.transform import SimpleTransformService
from ..utils.file_utils import detectron2_available
from ..utils.fs import get_configs_dir_path
from ..utils.settings import LayoutType, Relationships
from ..utils.transform import PadTransform

with try_import() as image_guard:
    from botocore.config import Config  # type: ignore


__all__ = [
    "ServiceFactory",
]

from ._config import cfg


class ServiceFactory:
    @staticmethod
    def build_layout_detector(
        mode: str,
    ) -> Union[D2FrcnnDetector, TPFrcnnDetector, HFDetrDerivedDetector, D2FrcnnTracingDetector]:
        """Building a D2-Detector, a TP-Detector as Detr-Detector or a D2-Torch Tracing Detector according to
        the config

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
        categories = profile.categories if profile.categories is not None else {}

        if profile.model_wrapper in ("TPFrcnnDetector",):
            return TPFrcnnDetector(
                path_yaml=config_path,
                path_weights=weights_path,
                categories=categories,
                filter_categories=filter_categories,
            )
        if profile.model_wrapper in ("D2FrcnnDetector",):
            return D2FrcnnDetector(
                path_yaml=config_path,
                path_weights=weights_path,
                categories=categories,
                device=cfg.DEVICE,
                filter_categories=filter_categories,
            )
        if profile.model_wrapper in ("D2FrcnnTracingDetector",):
            return D2FrcnnTracingDetector(
                path_yaml=config_path,
                path_weights=weights_path,
                categories=categories,
                filter_categories=filter_categories,
            )
        if profile.model_wrapper in ("HFDetrDerivedDetector",):
            preprocessor_config = ModelCatalog.get_full_path_preprocessor_configs(weights)
            return HFDetrDerivedDetector(
                path_config_json=config_path,
                path_weights=weights_path,
                path_feature_extractor_config_json=preprocessor_config,
                categories=categories,
                device=cfg.DEVICE,
                filter_categories=filter_categories,
            )
        raise TypeError(
            f"You have chosen profile.model_wrapper: {profile.model_wrapper} which is not allowed. Please check "
            f"compatability with your deep learning framework"
        )

    @staticmethod
    def build_rotation_detector() -> TesseractRotationTransformer:
        return TesseractRotationTransformer()

    @staticmethod
    def build_transform_service(transform_predictor: ImageTransformer) -> SimpleTransformService:
        return SimpleTransformService(transform_predictor)

    @staticmethod
    def build_padder(mode: str) -> PadTransform:
        """Building a padder according to the config

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

    @staticmethod
    def build_layout_service(detector: ObjectDetector, mode: str) -> ImageLayoutService:
        """Building a layout service with a given detector

        :param detector: will be passed to the `ImageLayoutService`
        :param mode: either `LAYOUT`,`CELL` or `ITEM`
        :return `ImageLayoutService` instance
        """
        padder = None
        if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
            padder = ServiceFactory.build_padder(mode=mode)
        return ImageLayoutService(layout_detector=detector, to_image=True, crop_image=True, padder=padder)

    @staticmethod
    def build_layout_nms_service() -> AnnotationNmsService:
        """Building a NMS service for layout annotations"""
        if not detectron2_available() and cfg.LIB == "PT":
            raise ModuleNotFoundError("LAYOUT_NMS_PAIRS is only available for detectron2")
        if not isinstance(cfg.LAYOUT_NMS_PAIRS.COMBINATIONS, list) and not isinstance(
            cfg.LAYOUT_NMS_PAIRS.COMBINATIONS[0], list
        ):
            raise ValueError("LAYOUT_NMS_PAIRS must be a list of lists")
        return AnnotationNmsService(
            nms_pairs=cfg.LAYOUT_NMS_PAIRS.COMBINATIONS,
            thresholds=cfg.LAYOUT_NMS_PAIRS.THRESHOLDS,
            priority=cfg.LAYOUT_NMS_PAIRS.PRIORITY,
        )

    @staticmethod
    def build_sub_image_service(detector: ObjectDetector, mode: str) -> SubImageLayoutService:
        """
        Building a sub image layout service with a given detector

        :param detector: will be passed to the `SubImageLayoutService`
        :param mode: either `LAYOUT`,`CELL` or `ITEM`
        :return: `SubImageLayoutService` instance
        """
        exclude_category_ids = []
        padder = None
        if mode == "ITEM":
            if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
                exclude_category_ids.extend([1, 3, 4, 5, 6])
                padder = ServiceFactory.build_padder(mode)
        detect_result_generator = DetectResultGenerator(
            categories=detector.categories.categories, exclude_category_ids=exclude_category_ids
        )
        return SubImageLayoutService(
            sub_image_detector=detector,
            sub_image_names=[LayoutType.TABLE, LayoutType.TABLE_ROTATED],
            category_id_mapping=None,
            detect_result_generator=detect_result_generator,
            padder=padder,
        )

    @staticmethod
    def build_ocr_detector() -> Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]:
        """
        Building OCR predictor
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
                architecture=profile.architecture,
                path_weights=weights_path,
                device=cfg.DEVICE,
                lib=cfg.LIB,
                path_config_json=config_path,
            )
        if cfg.OCR.USE_TEXTRACT:
            credentials_kwargs = {
                "aws_access_key_id": environ.get("ACCESS_KEY", None),
                "aws_secret_access_key": environ.get("SECRET_KEY", None),
                "config": Config(region_name=environ.get("REGION", None)),
            }
            return TextractOcrDetector(**credentials_kwargs)
        raise ValueError("You have set USE_OCR=True but any of USE_TESSERACT, USE_DOCTR, USE_TEXTRACT is set to False")

    @staticmethod
    def build_doctr_word_detector() -> DoctrTextlineDetector:
        """Building `DoctrTextlineDetector` instance"""
        weights = cfg.OCR.WEIGHTS.DOCTR_WORD.TF if cfg.LIB == "TF" else cfg.OCR.WEIGHTS.DOCTR_WORD.PT
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        profile = ModelCatalog.get_profile(weights)
        if profile.architecture is None:
            raise ValueError("model profile.architecture must be specified")
        if profile.categories is None:
            raise ValueError("model profile.categories must be specified")
        return DoctrTextlineDetector(profile.architecture, weights_path, profile.categories, cfg.DEVICE, lib=cfg.LIB)

    @staticmethod
    def build_table_segmentation_service(
        detector: ObjectDetector,
    ) -> Union[PubtablesSegmentationService, TableSegmentationService]:
        table_segmentation: Union[PubtablesSegmentationService, TableSegmentationService]
        if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
            table_segmentation = PubtablesSegmentationService(
                segment_rule=cfg.SEGMENTATION.ASSIGNMENT_RULE,
                threshold_rows=cfg.SEGMENTATION.THRESHOLD_ROWS,
                threshold_cols=cfg.SEGMENTATION.THRESHOLD_COLS,
                tile_table_with_items=cfg.SEGMENTATION.FULL_TABLE_TILING,
                remove_iou_threshold_rows=cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                remove_iou_threshold_cols=cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                cell_class_id=cfg.SEGMENTATION.CELL_CATEGORY_ID,
                table_name=cfg.SEGMENTATION.TABLE_NAME,
                cell_names=cfg.SEGMENTATION.PUBTABLES_CELL_NAMES,
                spanning_cell_names=cfg.SEGMENTATION.PUBTABLES_SPANNING_CELL_NAMES,
                item_names=cfg.SEGMENTATION.PUBTABLES_ITEM_NAMES,
                sub_item_names=cfg.SEGMENTATION.PUBTABLES_SUB_ITEM_NAMES,
                stretch_rule=cfg.SEGMENTATION.STRETCH_RULE,
            )

        else:
            table_segmentation = TableSegmentationService(
                segment_rule=cfg.SEGMENTATION.ASSIGNMENT_RULE,
                threshold_rows=cfg.SEGMENTATION.THRESHOLD_ROWS,
                threshold_cols=cfg.SEGMENTATION.THRESHOLD_COLS,
                tile_table_with_items=cfg.SEGMENTATION.FULL_TABLE_TILING,
                remove_iou_threshold_rows=cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                remove_iou_threshold_cols=cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                table_name=cfg.SEGMENTATION.TABLE_NAME,
                cell_names=cfg.SEGMENTATION.CELL_NAMES,
                item_names=cfg.SEGMENTATION.ITEM_NAMES,
                sub_item_names=cfg.SEGMENTATION.SUB_ITEM_NAMES,
                stretch_rule=cfg.SEGMENTATION.STRETCH_RULE,
            )
        return table_segmentation

    @staticmethod
    def build_table_refinement_service() -> TableSegmentationRefinementService:
        return TableSegmentationRefinementService(
            [cfg.SEGMENTATION.TABLE_NAME],
            cfg.SEGMENTATION.PUBTABLES_CELL_NAMES,
        )

    @staticmethod
    def build_pdf_text_detector() -> PdfPlumberTextDetector:
        return PdfPlumberTextDetector(x_tolerance=cfg.PDF_MINER.X_TOLERANCE, y_tolerance=cfg.PDF_MINER.Y_TOLERANCE)

    @staticmethod
    def build_pdf_miner_text_service(detector: PdfMiner) -> TextExtractionService:
        return TextExtractionService(detector)

    @staticmethod
    def build_doctr_word_detector_service(detector: DoctrTextlineDetector) -> ImageLayoutService:
        return ImageLayoutService(
            layout_detector=detector, to_image=True, crop_image=True, skip_if_layout_extracted=True
        )

    @staticmethod
    def build_text_extraction_service(detector: Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]) \
            -> TextExtractionService:
        return TextExtractionService(
            detector,
            skip_if_text_extracted=cfg.USE_PDF_MINER,
            extract_from_roi=cfg.TEXT_CONTAINER if cfg.OCR.USE_DOCTR else None,
        )

    @staticmethod
    def build_word_matching_service() -> MatchingService:
        matcher = IntersectionMatcher(
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.THRESHOLD,
            max_parent_only=cfg.WORD_MATCHING.MAX_PARENT_ONLY,
        )
        return MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=cfg.TEXT_CONTAINER,
            matcher=matcher,
            relationship_key=Relationships.CHILD,
        )

    @staticmethod
    def build_text_order_service() -> TextOrderService:
        return TextOrderService(
            text_container=cfg.TEXT_CONTAINER,
            text_block_categories=cfg.TEXT_ORDERING.TEXT_BLOCK_CATEGORIES,
            floating_text_block_categories=cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            include_residual_text_container=cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
            starting_point_tolerance=cfg.TEXT_ORDERING.STARTING_POINT_TOLERANCE,
            broken_line_tolerance=cfg.TEXT_ORDERING.BROKEN_LINE_TOLERANCE,
            height_tolerance=cfg.TEXT_ORDERING.HEIGHT_TOLERANCE,
            paragraph_break=cfg.TEXT_ORDERING.PARAGRAPH_BREAK,
        )

    @staticmethod
    def build_page_parsing_service() -> PageParsingService:
        return PageParsingService(
            text_container=cfg.TEXT_CONTAINER,
            floating_text_block_categories=cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            include_residual_text_container=cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
        )

    @staticmethod
    def build_analyzer() -> DoctectionPipe:
        """
        Builds the analyzer with a given config

        :return: Analyzer pipeline
        """
        pipe_component_list: list[PipelineComponent] = []

        if cfg.USE_ROTATOR:
            rotation_detector = ServiceFactory.build_rotation_detector()
            transform_service = ServiceFactory.build_transform_service(transform_predictor=rotation_detector)
            pipe_component_list.append(transform_service)

        if cfg.USE_LAYOUT:
            layout_detector = ServiceFactory.build_layout_detector(mode="LAYOUT")
            layout_service = ServiceFactory.build_layout_service(detector=layout_detector, mode="LAYOUT")
            pipe_component_list.append(layout_service)

        # setup layout nms service
        if cfg.USE_LAYOUT_NMS:
            layout_nms_service = ServiceFactory.build_layout_nms_service()
            pipe_component_list.append(layout_nms_service)

        # setup tables service
        if cfg.USE_TABLE_SEGMENTATION:
            item_detector = ServiceFactory.build_layout_detector(mode="ITEM")
            item_service = ServiceFactory.build_sub_image_service(detector=item_detector, mode="ITEM")
            pipe_component_list.append(item_service)

            if item_detector.__class__.__name__ not in ("HFDetrDerivedDetector",):
                cell_detector = ServiceFactory.build_layout_detector(mode="CELL")
                cell_service = ServiceFactory.build_sub_image_service(detector=cell_detector, mode="CELL")
                pipe_component_list.append(cell_service)

            table_segmentation_service = ServiceFactory.build_table_segmentation_service(detector=item_detector)
            pipe_component_list.append(table_segmentation_service)

            if cfg.USE_TABLE_REFINEMENT:
                table_refinement_service = ServiceFactory.build_table_refinement_service()
                pipe_component_list.append(table_refinement_service)

        if cfg.USE_PDF_MINER:
            pdf_miner = ServiceFactory.build_pdf_text_detector()
            d_text = ServiceFactory.build_pdf_miner_text_service(pdf_miner)
            pipe_component_list.append(d_text)

        # setup ocr
        if cfg.USE_OCR:
            # the extra mile for DocTr
            if cfg.OCR.USE_DOCTR:
                word_detector = ServiceFactory.build_doctr_word_detector()
                word_service = ServiceFactory.build_doctr_word_detector_service(word_detector)
                pipe_component_list.append(word_service)

            ocr_detector = ServiceFactory.build_ocr_detector()
            text_extraction_service = ServiceFactory.build_text_extraction_service(ocr_detector)
            pipe_component_list.append(text_extraction_service)

        if cfg.USE_PDF_MINER or cfg.USE_OCR:
            matching_service = ServiceFactory.build_word_matching_service()
            pipe_component_list.append(matching_service)

            text_order_service = ServiceFactory.build_text_order_service()
            pipe_component_list.append(text_order_service)

        page_parsing_service = ServiceFactory.build_page_parsing_service()

        return DoctectionPipe(pipeline_component_list=pipe_component_list, page_parsing_service=page_parsing_service)
