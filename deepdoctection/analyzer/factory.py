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

"""
`ServiceFactory` for building analyzers
"""

from __future__ import annotations

from os import environ
from typing import TYPE_CHECKING, Union

from lazy_imports import try_import

from ..extern.base import ImageTransformer, ObjectDetector, PdfMiner
from ..extern.d2detect import D2FrcnnDetector, D2FrcnnTracingDetector
from ..extern.doctrocr import DoctrTextlineDetector, DoctrTextRecognizer
from ..extern.hfdetr import HFDetrDerivedDetector
from ..extern.hflayoutlm import (
    HFLayoutLmSequenceClassifier,
    HFLayoutLmTokenClassifier,
    HFLayoutLmv2SequenceClassifier,
    HFLayoutLmv2TokenClassifier,
    HFLayoutLmv3SequenceClassifier,
    HFLayoutLmv3TokenClassifier,
    HFLiltSequenceClassifier,
    HFLiltTokenClassifier,
    get_tokenizer_from_model_class,
)
from ..extern.hflm import HFLmSequenceClassifier, HFLmTokenClassifier
from ..extern.model import ModelCatalog, ModelDownloadManager
from ..extern.pdftext import PdfPlumberTextDetector
from ..extern.tessocr import TesseractOcrDetector, TesseractRotationTransformer
from ..extern.texocr import TextractOcrDetector
from ..extern.tpdetect import TPFrcnnDetector
from ..pipe.base import PipelineComponent
from ..pipe.common import (
    AnnotationNmsService,
    FamilyCompound,
    IntersectionMatcher,
    MatchingService,
    NeighbourMatcher,
    PageParsingService,
)
from ..pipe.doctectionpipe import DoctectionPipe
from ..pipe.layout import ImageLayoutService, skip_if_category_or_service_extracted
from ..pipe.lm import LMSequenceClassifierService, LMTokenClassifierService
from ..pipe.order import TextOrderService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.segment import PubtablesSegmentationService, TableSegmentationService
from ..pipe.sub_layout import DetectResultGenerator, SubImageLayoutService
from ..pipe.text import TextExtractionService
from ..pipe.transform import SimpleTransformService
from ..utils.error import DependencyError
from ..utils.fs import get_configs_dir_path
from ..utils.metacfg import AttrDict
from ..utils.settings import CellType, LayoutType, Relationships
from ..utils.transform import PadTransform

with try_import() as image_guard:
    from botocore.config import Config  # type: ignore

if TYPE_CHECKING:
    from ..extern.hflayoutlm import LayoutSequenceModels, LayoutTokenModels
    from ..extern.hflm import LmSequenceModels, LmTokenModels


__all__ = [
    "ServiceFactory",
]


class ServiceFactory:
    """
    Factory class for building various components of the deepdoctection analyzer pipeline.

    This class uses the `cfg` configuration object from `config.py`, which is an instance of the `AttrDict` class.
    The configuration is not passed explicitly in an `__init__` method but is accessed directly within the methods.

    The class provides static methods to build different services and detectors required for the pipeline, such as
    layout detectors, OCR detectors, table segmentation services, and more. The methods disentangle the creation
    of predictors (e.g., `ObjectDetector`, `TextRecognizer`) from the configuration, allowing for flexible and
    modular construction of the pipeline components.

    Extending the Class:
        This class can be extended by using inheritance and adding new methods or overriding existing ones.
        To extend the configuration attributes, you can modify the `cfg` object in `config.py` to include new
        settings or parameters required for the new methods.
    """

    @staticmethod
    def _build_layout_detector(
        config: AttrDict,
        mode: str,
    ) -> Union[D2FrcnnDetector, TPFrcnnDetector, HFDetrDerivedDetector, D2FrcnnTracingDetector]:
        """
        Building a D2-Detector, a TP-Detector as Detr-Detector or a D2-Torch Tracing Detector according to
        the config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """
        if config.LIB is None:
            raise DependencyError("At least one of the env variables DD_USE_TF or DD_USE_TORCH must be set.")

        weights = (
            getattr(config.TF, mode).WEIGHTS
            if config.LIB == "TF"
            else (
                getattr(config.PT, mode).WEIGHTS
                if getattr(config.PT.ENFORCE_WEIGHTS, mode)
                else getattr(config.PT, mode).WEIGHTS_TS
            )
        )
        filter_categories = (
            getattr(getattr(config.TF, mode), "FILTER")
            if config.LIB == "TF"
            else getattr(getattr(config.PT, mode), "FILTER")
        )
        config_path = ModelCatalog.get_full_path_configs(weights)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        profile = ModelCatalog.get_profile(weights)
        if config.LIB == "PT" and profile.padding is not None:
            getattr(config.PT, mode).PADDING = profile.padding
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
                device=config.DEVICE,
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
                device=config.DEVICE,
                filter_categories=filter_categories,
            )
        raise TypeError(
            f"You have chosen profile.model_wrapper: {profile.model_wrapper} which is not allowed. Please check "
            f"compatability with your deep learning framework"
        )

    @staticmethod
    def build_layout_detector(
        config: AttrDict, mode: str
    ) -> Union[D2FrcnnDetector, TPFrcnnDetector, HFDetrDerivedDetector, D2FrcnnTracingDetector]:
        """
        Building a layout detector according to the config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """
        return ServiceFactory._build_layout_detector(config, mode)

    @staticmethod
    def _build_rotation_detector() -> TesseractRotationTransformer:
        """
        Building a rotation detector.

        Returns:
            TesseractRotationTransformer: Rotation detector instance.
        """
        return TesseractRotationTransformer()

    @staticmethod
    def build_rotation_detector() -> TesseractRotationTransformer:
        """
        Building a rotation detector.

        Returns:
            TesseractRotationTransformer: Rotation detector instance.
        """
        return ServiceFactory._build_rotation_detector()

    @staticmethod
    def _build_transform_service(transform_predictor: ImageTransformer) -> SimpleTransformService:
        """
        Building a transform service with a given predictor.

        Args:
            transform_predictor: Predictor for image transformation.

        Returns:
            SimpleTransformService: Transform service instance.
        """
        return SimpleTransformService(transform_predictor)

    @staticmethod
    def build_transform_service(transform_predictor: ImageTransformer) -> SimpleTransformService:
        """
        Building a transform service with a given predictor.

        Args:
            transform_predictor: Predictor for image transformation.

        Returns:
            SimpleTransformService: Transform service instance.
        """
        return ServiceFactory._build_transform_service(transform_predictor)

    @staticmethod
    def _build_padder(config: AttrDict, mode: str) -> PadTransform:
        """
        Building a padder according to the config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.

        Returns:
            PadTransform: `PadTransform` instance.
        """
        top, right, bottom, left = (
            getattr(config.PT, mode).PAD.TOP,
            getattr(config.PT, mode).PAD.RIGHT,
            getattr(config.PT, mode).PAD.BOTTOM,
            getattr(config.PT, mode).PAD.LEFT,
        )
        return PadTransform(pad_top=top, pad_right=right, pad_bottom=bottom, pad_left=left)  #

    @staticmethod
    def build_padder(config: AttrDict, mode: str) -> PadTransform:
        """
        Building a padder according to the config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.

        Returns:
            PadTransform: `PadTransform` instance.
        """
        return ServiceFactory._build_padder(config, mode)

    @staticmethod
    def _build_layout_service(config: AttrDict, detector: ObjectDetector, mode: str) -> ImageLayoutService:
        """
        Building a layout service with a given detector.

        Args:
            config: Configuration object.
            detector: Will be passed to the `ImageLayoutService`.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.

        Returns:
            ImageLayoutService: `ImageLayoutService` instance.
        """
        padder = None
        if getattr(config.PT, mode).PADDING:
            padder = ServiceFactory.build_padder(config, mode=mode)
        return ImageLayoutService(layout_detector=detector, to_image=True, crop_image=True, padder=padder)

    @staticmethod
    def build_layout_service(config: AttrDict, detector: ObjectDetector, mode: str) -> ImageLayoutService:
        """
        Building a layout service with a given detector.

        Args:
            config: Configuration object.
            detector: Will be passed to the `ImageLayoutService`.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.

        Returns:
            ImageLayoutService: `ImageLayoutService` instance.
        """
        return ServiceFactory._build_layout_service(config, detector, mode)

    @staticmethod
    def _build_layout_nms_service(config: AttrDict) -> AnnotationNmsService:
        """
        Building a NMS service for layout annotations.

        Args:
            config: Configuration object.

        Returns:
            AnnotationNmsService: NMS service instance.
        """
        if not isinstance(config.LAYOUT_NMS_PAIRS.COMBINATIONS, list) and not isinstance(
            config.LAYOUT_NMS_PAIRS.COMBINATIONS[0], list
        ):
            raise ValueError("LAYOUT_NMS_PAIRS must be a list of lists")
        return AnnotationNmsService(
            nms_pairs=config.LAYOUT_NMS_PAIRS.COMBINATIONS,
            thresholds=config.LAYOUT_NMS_PAIRS.THRESHOLDS,
            priority=config.LAYOUT_NMS_PAIRS.PRIORITY,
        )

    @staticmethod
    def build_layout_nms_service(config: AttrDict) -> AnnotationNmsService:
        """
        Building a NMS service for layout annotations.

        Args:
            config: Configuration object.

        Returns:
            AnnotationNmsService: NMS service instance.
        """
        return ServiceFactory._build_layout_nms_service(config)

    @staticmethod
    def _build_sub_image_service(config: AttrDict, detector: ObjectDetector, mode: str) -> SubImageLayoutService:
        """
        Building a sub image layout service with a given detector.

        Args:
            config: Configuration object.
            detector: Will be passed to the `SubImageLayoutService`.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.

        Returns:
            SubImageLayoutService: `SubImageLayoutService` instance.
        """
        exclude_category_names = []
        padder = None
        if mode == "ITEM":
            if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
                exclude_category_names.extend(
                    [LayoutType.TABLE, CellType.COLUMN_HEADER, CellType.PROJECTED_ROW_HEADER, CellType.SPANNING]
                )
                padder = ServiceFactory.build_padder(config, mode)
        detect_result_generator = DetectResultGenerator(
            categories_name_as_key=detector.categories.get_categories(as_dict=True, name_as_key=True),
            exclude_category_names=exclude_category_names,
        )
        return SubImageLayoutService(
            sub_image_detector=detector,
            sub_image_names=[LayoutType.TABLE, LayoutType.TABLE_ROTATED],
            detect_result_generator=detect_result_generator,
            padder=padder,
        )

    @staticmethod
    def build_sub_image_service(config: AttrDict, detector: ObjectDetector, mode: str) -> SubImageLayoutService:
        """
        Building a sub image layout service with a given detector.

        Args:
            config: Configuration object.
            detector: Will be passed to the `SubImageLayoutService`.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.

        Returns:
            SubImageLayoutService: `SubImageLayoutService` instance.
        """
        return ServiceFactory._build_sub_image_service(config, detector, mode)

    @staticmethod
    def _build_ocr_detector(config: AttrDict) -> Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]:
        """
        Building OCR predictor.

        Args:
            config: Configuration object.

        Returns:
            Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]: OCR detector instance.
        """
        if config.OCR.USE_TESSERACT:
            ocr_config_path = get_configs_dir_path() / config.OCR.CONFIG.TESSERACT
            return TesseractOcrDetector(
                ocr_config_path,
                config_overwrite=[f"LANGUAGES={config.LANGUAGE}"] if config.LANGUAGE is not None else None,
            )
        if config.OCR.USE_DOCTR:
            if config.LIB is None:
                raise DependencyError("At least one of the env variables DD_USE_TF or DD_USE_TORCH must be set.")
            weights = (
                config.OCR.WEIGHTS.DOCTR_RECOGNITION.TF
                if config.LIB == "TF"
                else (config.OCR.WEIGHTS.DOCTR_RECOGNITION.PT)
            )
            weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
            profile = ModelCatalog.get_profile(weights)
            # get_full_path_configs will complete the path even if the model is not registered
            config_path = ModelCatalog.get_full_path_configs(weights) if profile.config is not None else None
            if profile.architecture is None:
                raise ValueError("model profile.architecture must be specified")
            return DoctrTextRecognizer(
                architecture=profile.architecture,
                path_weights=weights_path,
                device=config.DEVICE,
                lib=config.LIB,
                path_config_json=config_path,
            )
        if config.OCR.USE_TEXTRACT:
            credentials_kwargs = {
                "aws_access_key_id": environ.get("AWS_ACCESS_KEY", None),
                "aws_secret_access_key": environ.get("AWS_SECRET_KEY", None),
                "config": Config(region_name=environ.get("AWS_REGION", None)),
            }
            return TextractOcrDetector(**credentials_kwargs)
        raise ValueError("You have set USE_OCR=True but any of USE_TESSERACT, USE_DOCTR, USE_TEXTRACT is set to False")

    @staticmethod
    def build_ocr_detector(config: AttrDict) -> Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]:
        """
        Building OCR predictor.

        Args:
            config: Configuration object.

        Returns:
            Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]: OCR detector instance.
        """
        return ServiceFactory._build_ocr_detector(config)

    @staticmethod
    def build_doctr_word_detector(config: AttrDict) -> DoctrTextlineDetector:
        """
        Building `DoctrTextlineDetector` instance.

        Args:
            config: Configuration object.

        Returns:
            DoctrTextlineDetector: Textline detector instance.
        """
        if config.LIB is None:
            raise DependencyError("At least one of the env variables DD_USE_TF or DD_USE_TORCH must be set.")
        weights = config.OCR.WEIGHTS.DOCTR_WORD.TF if config.LIB == "TF" else config.OCR.WEIGHTS.DOCTR_WORD.PT
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        profile = ModelCatalog.get_profile(weights)
        if profile.architecture is None:
            raise ValueError("model profile.architecture must be specified")
        if profile.categories is None:
            raise ValueError("model profile.categories must be specified")
        return DoctrTextlineDetector(
            profile.architecture, weights_path, profile.categories, config.DEVICE, lib=config.LIB
        )

    @staticmethod
    def _build_table_segmentation_service(
        config: AttrDict,
        detector: ObjectDetector,
    ) -> Union[PubtablesSegmentationService, TableSegmentationService]:
        """
        Build and return a table segmentation service based on the provided detector.

        Note:
            Depending on the type of the detector, this method will return either a `PubtablesSegmentationService` or a
            `TableSegmentationService` instance. The selection is made as follows:

            - If the detector is an instance of `HFDetrDerivedDetector`, a `PubtablesSegmentationService` is created and
              returned. This service uses specific configuration parameters for segmentation, such as assignment rules,
              thresholds, and cell names defined in the `cfg` object.
            - For other detector types, a `TableSegmentationService` is created and returned. This service also uses
              configuration parameters from the `cfg` object but is tailored for different segmentation needs.

        Args:
            config: Configuration object.
            detector: An instance of `ObjectDetector` used to determine the type of table segmentation service to build.

        Returns:
            Table segmentation service instance.
        """
        table_segmentation: Union[PubtablesSegmentationService, TableSegmentationService]
        if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
            table_segmentation = PubtablesSegmentationService(
                segment_rule=config.SEGMENTATION.ASSIGNMENT_RULE,
                threshold_rows=config.SEGMENTATION.THRESHOLD_ROWS,
                threshold_cols=config.SEGMENTATION.THRESHOLD_COLS,
                tile_table_with_items=config.SEGMENTATION.FULL_TABLE_TILING,
                remove_iou_threshold_rows=config.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                remove_iou_threshold_cols=config.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                table_name=config.SEGMENTATION.TABLE_NAME,
                cell_names=config.SEGMENTATION.PUBTABLES_CELL_NAMES,
                spanning_cell_names=config.SEGMENTATION.PUBTABLES_SPANNING_CELL_NAMES,
                item_names=config.SEGMENTATION.PUBTABLES_ITEM_NAMES,
                sub_item_names=config.SEGMENTATION.PUBTABLES_SUB_ITEM_NAMES,
                item_header_cell_names=config.SEGMENTATION.PUBTABLES_ITEM_HEADER_CELL_NAMES,
                item_header_thresholds=config.SEGMENTATION.PUBTABLES_ITEM_HEADER_THRESHOLDS,
                stretch_rule=config.SEGMENTATION.STRETCH_RULE,
            )

        else:
            table_segmentation = TableSegmentationService(
                segment_rule=config.SEGMENTATION.ASSIGNMENT_RULE,
                threshold_rows=config.SEGMENTATION.THRESHOLD_ROWS,
                threshold_cols=config.SEGMENTATION.THRESHOLD_COLS,
                tile_table_with_items=config.SEGMENTATION.FULL_TABLE_TILING,
                remove_iou_threshold_rows=config.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
                remove_iou_threshold_cols=config.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
                table_name=config.SEGMENTATION.TABLE_NAME,
                cell_names=config.SEGMENTATION.CELL_NAMES,
                item_names=config.SEGMENTATION.ITEM_NAMES,
                sub_item_names=config.SEGMENTATION.SUB_ITEM_NAMES,
                stretch_rule=config.SEGMENTATION.STRETCH_RULE,
            )
        return table_segmentation

    @staticmethod
    def build_table_segmentation_service(
        config: AttrDict,
        detector: ObjectDetector,
    ) -> Union[PubtablesSegmentationService, TableSegmentationService]:
        """
        Build and return a table segmentation service based on the provided detector.

        Note:
            Depending on the type of the detector, this method will return either a `PubtablesSegmentationService` or a
            `TableSegmentationService` instance. The selection is made as follows:

            - If the detector is an instance of `HFDetrDerivedDetector`, a `PubtablesSegmentationService` is created and
              returned. This service uses specific configuration parameters for segmentation, such as assignment rules,
              thresholds, and cell names defined in the `cfg` object.
            - For other detector types, a `TableSegmentationService` is created and returned. This service also uses
              configuration parameters from the `cfg` object but is tailored for different segmentation needs.

        Args:
            config: Configuration object.
            detector: An instance of `ObjectDetector` used to determine the type of table segmentation service to build.

        Returns:
            Table segmentation service instance.
        """
        return ServiceFactory._build_table_segmentation_service(config, detector)

    @staticmethod
    def _build_table_refinement_service(config: AttrDict) -> TableSegmentationRefinementService:
        """
        Building a table segmentation refinement service.

        Args:
            config: Configuration object.

        Returns:
            TableSegmentationRefinementService: Refinement service instance.
        """
        return TableSegmentationRefinementService(
            [config.SEGMENTATION.TABLE_NAME],
            config.SEGMENTATION.PUBTABLES_CELL_NAMES,
        )

    @staticmethod
    def build_table_refinement_service(config: AttrDict) -> TableSegmentationRefinementService:
        """
        Building a table segmentation refinement service.

        Args:
            config: Configuration object.

        Returns:
            TableSegmentationRefinementService: Refinement service instance.
        """
        return ServiceFactory._build_table_refinement_service(config)

    @staticmethod
    def _build_pdf_text_detector(config: AttrDict) -> PdfPlumberTextDetector:
        """
        Building a PDF text detector.

        Args:
            config: Configuration object.

        Returns:
            PdfPlumberTextDetector: PDF text detector instance.
        """
        return PdfPlumberTextDetector(
            x_tolerance=config.PDF_MINER.X_TOLERANCE, y_tolerance=config.PDF_MINER.Y_TOLERANCE
        )

    @staticmethod
    def build_pdf_text_detector(config: AttrDict) -> PdfPlumberTextDetector:
        """
        Building a PDF text detector.

        Args:
            config: Configuration object.

        Returns:
            PdfPlumberTextDetector: PDF text detector instance.
        """
        return ServiceFactory._build_pdf_text_detector(config)

    @staticmethod
    def _build_pdf_miner_text_service(detector: PdfMiner) -> TextExtractionService:
        """
        Building a PDFMiner text extraction service.

        Args:
            detector: PdfMiner instance.

        Returns:
            TextExtractionService: Text extraction service instance.
        """
        return TextExtractionService(detector)

    @staticmethod
    def build_pdf_miner_text_service(detector: PdfMiner) -> TextExtractionService:
        """
        Building a PDFMiner text extraction service.

        Args:
            detector: PdfMiner instance.

        Returns:
            TextExtractionService: Text extraction service instance.
        """
        return ServiceFactory._build_pdf_miner_text_service(detector)

    @staticmethod
    def _build_doctr_word_detector_service(detector: DoctrTextlineDetector) -> ImageLayoutService:
        """
        Building a Doctr word detector service.

        Args:
            detector: DoctrTextlineDetector instance.

        Returns:
            ImageLayoutService: Word detector service instance.
        """
        return ImageLayoutService(layout_detector=detector, to_image=True, crop_image=True)

    @staticmethod
    def build_doctr_word_detector_service(detector: DoctrTextlineDetector) -> ImageLayoutService:
        """
        Building a Doctr word detector service.

        Args:
            detector: DoctrTextlineDetector instance.

        Returns:
            ImageLayoutService: Word detector service instance.
        """
        return ServiceFactory._build_doctr_word_detector_service(detector)

    @staticmethod
    def _build_text_extraction_service(
        config: AttrDict, detector: Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]
    ) -> TextExtractionService:
        """
        Building a text extraction service.

        Args:
            config: Configuration object.
            detector: OCR detector instance.

        Returns:
            TextExtractionService: Text extraction service instance.
        """
        return TextExtractionService(
            detector,
            extract_from_roi=config.TEXT_CONTAINER if config.OCR.USE_DOCTR else None,
        )

    @staticmethod
    def build_text_extraction_service(
        config: AttrDict, detector: Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]
    ) -> TextExtractionService:
        """
        Building a text extraction service.

        Args:
            config: Configuration object.
            detector: OCR detector instance.

        Returns:
            TextExtractionService: Text extraction service instance.
        """
        return ServiceFactory._build_text_extraction_service(config, detector)

    @staticmethod
    def _build_word_matching_service(config: AttrDict) -> MatchingService:
        """
        Building a word matching service.

        Args:
            config: Configuration object.

        Returns:
            MatchingService: Word matching service instance.
        """
        matcher = IntersectionMatcher(
            matching_rule=config.WORD_MATCHING.RULE,
            threshold=config.WORD_MATCHING.THRESHOLD,
            max_parent_only=config.WORD_MATCHING.MAX_PARENT_ONLY,
        )
        family_compounds = [
            FamilyCompound(
                parent_categories=config.WORD_MATCHING.PARENTAL_CATEGORIES,
                child_categories=config.TEXT_CONTAINER,
                relationship_key=Relationships.CHILD,
            ),
            FamilyCompound(
                parent_categories=[LayoutType.LIST],
                child_categories=[LayoutType.LIST_ITEM],
                relationship_key=Relationships.CHILD,
                create_synthetic_parent=True,
                synthetic_parent=LayoutType.LIST,
            ),
        ]
        return MatchingService(
            family_compounds=family_compounds,
            matcher=matcher,
        )

    @staticmethod
    def build_word_matching_service(config: AttrDict) -> MatchingService:
        """
        Building a word matching service.

        Args:
            config: Configuration object.

        Returns:
            MatchingService: Word matching service instance.
        """
        return ServiceFactory._build_word_matching_service(config)

    @staticmethod
    def _build_layout_link_matching_service(config: AttrDict) -> MatchingService:
        """
        Building a layout link matching service.

        Args:
            config: Configuration object.

        Returns:
            MatchingService: Layout link matching service instance.
        """
        neighbor_matcher = NeighbourMatcher()
        family_compounds = [
            FamilyCompound(
                parent_categories=config.LAYOUT_LINK.PARENTAL_CATEGORIES,
                child_categories=config.LAYOUT_LINK.CHILD_CATEGORIES,
                relationship_key=Relationships.LAYOUT_LINK,
            )
        ]
        return MatchingService(
            family_compounds=family_compounds,
            matcher=neighbor_matcher,
        )

    @staticmethod
    def build_layout_link_matching_service(config: AttrDict) -> MatchingService:
        """
        Building a layout link matching service.

        Args:
            config: Configuration object.

        Returns:
            MatchingService: Layout link matching service instance.
        """
        return ServiceFactory._build_layout_link_matching_service(config)

    @staticmethod
    def _build_line_matching_service(config: AttrDict) -> MatchingService:
        """
        Building a line matching service.

        Args:
            config: Configuration object.

        Returns:
            MatchingService: Line matching service instance.
        """
        matcher = IntersectionMatcher(
            matching_rule=config.WORD_MATCHING.RULE,
            threshold=config.WORD_MATCHING.THRESHOLD,
            max_parent_only=config.WORD_MATCHING.MAX_PARENT_ONLY,
        )
        family_compounds = [
            FamilyCompound(
                parent_categories=[LayoutType.LIST],
                child_categories=[LayoutType.LINE],
                relationship_key=Relationships.CHILD,
            ),
        ]
        return MatchingService(
            family_compounds=family_compounds,
            matcher=matcher,
        )

    @staticmethod
    def build_line_matching_service(config: AttrDict) -> MatchingService:
        """
        Building a line matching service.

        Args:
            config: Configuration object.

        Returns:
            MatchingService: Line matching service instance.
        """
        return ServiceFactory._build_line_matching_service(config)

    @staticmethod
    def _build_text_order_service(config: AttrDict) -> TextOrderService:
        """
        Building a text order service.

        Args:
            config: Configuration object.

        Returns:
            TextOrderService: Text order service instance.
        """
        return TextOrderService(
            text_container=config.TEXT_CONTAINER,
            text_block_categories=config.TEXT_ORDERING.TEXT_BLOCK_CATEGORIES,
            floating_text_block_categories=config.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            include_residual_text_container=config.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
            starting_point_tolerance=config.TEXT_ORDERING.STARTING_POINT_TOLERANCE,
            broken_line_tolerance=config.TEXT_ORDERING.BROKEN_LINE_TOLERANCE,
            height_tolerance=config.TEXT_ORDERING.HEIGHT_TOLERANCE,
            paragraph_break=config.TEXT_ORDERING.PARAGRAPH_BREAK,
        )

    @staticmethod
    def build_text_order_service(config: AttrDict) -> TextOrderService:
        """
        Building a text order service.

        Args:
            config: Configuration object.

        Returns:
            TextOrderService: Text order service instance.
        """
        return ServiceFactory._build_text_order_service(config)

    @staticmethod
    def _build_sequence_classifier(config: AttrDict) -> Union[LayoutSequenceModels, LmSequenceModels]:
        """
        Builds and returns a sequence classifier instance.

        Args:
            config: Configuration object that determines the type of sequence classifier to construct.

        Returns:
            A sequence classifier instance constructed according to the specified configuration.
        """
        config_path = ModelCatalog.get_full_path_configs(config.LM_SEQUENCE_CLASS.WEIGHTS)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(config.LM_SEQUENCE_CLASS.WEIGHTS)
        profile = ModelCatalog.get_profile(config.LM_SEQUENCE_CLASS.WEIGHTS)
        categories = profile.categories if profile.categories is not None else {}
        use_xlm_tokenizer = "xlm_tokenizer" == profile.architecture

        if profile.model_wrapper in ("HFLayoutLmSequenceClassifier",):
            return HFLayoutLmSequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
                use_xlm_tokenizer=use_xlm_tokenizer,
            )
        if profile.model_wrapper in ("HFLayoutLmv2SequenceClassifier",):
            return HFLayoutLmv2SequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
                use_xlm_tokenizer=use_xlm_tokenizer,
            )
        if profile.model_wrapper in ("HFLayoutLmv3SequenceClassifier",):
            return HFLayoutLmv3SequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
                use_xlm_tokenizer=use_xlm_tokenizer,
            )
        if profile.model_wrapper in ("HFLiltSequenceClassifier",):
            return HFLiltSequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
                use_xlm_tokenizer=use_xlm_tokenizer,
            )
        if profile.model_wrapper in ("HFLmSequenceClassifier",):
            return HFLmSequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
                use_xlm_tokenizer=use_xlm_tokenizer,
            )
        raise ValueError(f"Unsupported model wrapper: {profile.model_wrapper}")

    @staticmethod
    def build_sequence_classifier(config: AttrDict) -> Union[LayoutSequenceModels, LmSequenceModels]:
        """
        Builds and returns a sequence classifier instance.

        Args:
            config: Configuration object that determines the type of sequence classifier to construct.

        Returns:
            A sequence classifier instance constructed according to the specified configuration.
        """
        return ServiceFactory._build_sequence_classifier(config)

    @staticmethod
    def _build_sequence_classifier_service(
        config: AttrDict, sequence_classifier: Union[LayoutSequenceModels, LmSequenceModels]
    ) -> LMSequenceClassifierService:
        """
        Building a sequence classifier service.

        Args:
            config: Configuration object.
            sequence_classifier: Sequence classifier instance.

        Returns:
            LMSequenceClassifierService: Text order service instance.
        """
        tokenizer_fast = get_tokenizer_from_model_class(
            sequence_classifier.model.__class__.__name__, sequence_classifier.use_xlm_tokenizer
        )

        return LMSequenceClassifierService(
            tokenizer=tokenizer_fast,
            language_model=sequence_classifier,
            use_other_as_default_category=config.LM_SEQUENCE_CLASS.USE_OTHER_AS_DEFAULT_CATEGORY,
        )

    @staticmethod
    def build_sequence_classifier_service(
        config: AttrDict, sequence_classifier: Union[LayoutSequenceModels, LmSequenceModels]
    ) -> LMSequenceClassifierService:
        """
        Building a sequence classifier service.

        Args:
            config: Configuration object.
            sequence_classifier: Sequence classifier instance.

        Returns:
            LMSequenceClassifierService: Text order service instance.
        """
        return ServiceFactory._build_sequence_classifier_service(config, sequence_classifier)

    @staticmethod
    def _build_token_classifier(config: AttrDict) -> Union[LayoutTokenModels, LmTokenModels]:
        """
        Builds and returns a token classifier model.

        Args:
            config: Configuration object.

        Returns:
            The instantiated token classifier model.
        """
        config_path = ModelCatalog.get_full_path_configs(config.LM_TOKEN_CLASS.WEIGHTS)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(config.LM_TOKEN_CLASS.WEIGHTS)
        profile = ModelCatalog.get_profile(config.LM_TOKEN_CLASS.WEIGHTS)
        categories = profile.categories if profile.categories is not None else {}
        use_xlm_tokenizer = "xlm_tokenizer" == profile.architecture
        if profile.model_wrapper in ("HFLayoutLmTokenClassifier",):
            return HFLayoutLmTokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
                use_xlm_tokenizer=use_xlm_tokenizer,
            )
        if profile.model_wrapper in ("HFLayoutLmv2TokenClassifier",):
            return HFLayoutLmv2TokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
            )
        if profile.model_wrapper in ("HFLayoutLmv3TokenClassifier",):
            return HFLayoutLmv3TokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
            )
        if profile.model_wrapper in ("HFLiltTokenClassifier",):
            return HFLiltTokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=config.DEVICE,
            )
        if profile.model_wrapper in ("HFLmTokenClassifier",):
            return HFLmTokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
            )
        raise ValueError(f"Unsupported model wrapper: {profile.model_wrapper}")

    @staticmethod
    def build_token_classifier(config: AttrDict) -> Union[LayoutTokenModels, LmTokenModels]:
        """
        Builds and returns a token classifier model.

        Args:
            config: Configuration object.

        Returns:
            The instantiated token classifier model.
        """
        return ServiceFactory._build_token_classifier(config)

    @staticmethod
    def _build_token_classifier_service(
        config: AttrDict, token_classifier: Union[LayoutTokenModels, LmTokenModels]
    ) -> LMTokenClassifierService:
        """
        Building a token classifier service.

        Args:
            config: Configuration object.
            token_classifier: Token classifier instance.

        Returns:
             A LMTokenClassifierService instance.
        """
        tokenizer_fast = get_tokenizer_from_model_class(
            token_classifier.model.__class__.__name__, token_classifier.use_xlm_tokenizer
        )

        return LMTokenClassifierService(
            tokenizer=tokenizer_fast,
            language_model=token_classifier,
            use_other_as_default_category=config.LM_TOKEN_CLASS.USE_OTHER_AS_DEFAULT_CATEGORY,
            segment_positions=config.LM_TOKEN_CLASS.SEGMENT_POSITIONS,
            sliding_window_stride=config.LM_TOKEN_CLASS.SLIDING_WINDOW_STRIDE,
        )

    @staticmethod
    def build_token_classifier_service(
        config: AttrDict, token_classifier: Union[LayoutTokenModels, LmTokenModels]
    ) -> LMTokenClassifierService:
        """
        Building a token classifier service.

        Args:
            config: Configuration object.
            token_classifier: Token classifier instance.

        Returns:
             A LMTokenClassifierService instance.
        """
        return ServiceFactory._build_token_classifier_service(config, token_classifier)

    @staticmethod
    def _build_page_parsing_service(config: AttrDict) -> PageParsingService:
        """
        Building a page parsing service.

        Args:
            config: Configuration object.

        Returns:
            PageParsingService: Page parsing service instance.
        """
        return PageParsingService(
            text_container=config.TEXT_CONTAINER,
            floating_text_block_categories=config.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            include_residual_text_container=config.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
        )

    @staticmethod
    def build_page_parsing_service(config: AttrDict) -> PageParsingService:
        """
        Building a page parsing service.

        Args:
            config: Configuration object.

        Returns:
            PageParsingService: Page parsing service instance.
        """
        return ServiceFactory._build_page_parsing_service(config)

    @staticmethod
    def build_analyzer(config: AttrDict) -> DoctectionPipe:
        """
        Builds the analyzer with a given config.

        Args:
            config: Configuration object.

        Returns:
            DoctectionPipe: Analyzer pipeline instance.
        """
        pipe_component_list: list[PipelineComponent] = []

        if config.USE_ROTATOR:
            rotation_detector = ServiceFactory.build_rotation_detector()
            transform_service = ServiceFactory.build_transform_service(transform_predictor=rotation_detector)
            pipe_component_list.append(transform_service)

        if config.USE_LAYOUT:
            layout_detector = ServiceFactory.build_layout_detector(config, mode="LAYOUT")
            layout_service = ServiceFactory.build_layout_service(config, detector=layout_detector, mode="LAYOUT")
            pipe_component_list.append(layout_service)

        # setup layout nms service
        if config.USE_LAYOUT_NMS:
            layout_nms_service = ServiceFactory.build_layout_nms_service(config)
            pipe_component_list.append(layout_nms_service)

        # setup tables service
        if config.USE_TABLE_SEGMENTATION:
            item_detector = ServiceFactory.build_layout_detector(config, mode="ITEM")
            item_service = ServiceFactory.build_sub_image_service(config, detector=item_detector, mode="ITEM")
            pipe_component_list.append(item_service)

            if item_detector.__class__.__name__ not in ("HFDetrDerivedDetector",):
                cell_detector = ServiceFactory.build_layout_detector(config, mode="CELL")
                cell_service = ServiceFactory.build_sub_image_service(config, detector=cell_detector, mode="CELL")
                pipe_component_list.append(cell_service)

            table_segmentation_service = ServiceFactory.build_table_segmentation_service(config, detector=item_detector)
            pipe_component_list.append(table_segmentation_service)

            if config.USE_TABLE_REFINEMENT:
                table_refinement_service = ServiceFactory.build_table_refinement_service(config)
                pipe_component_list.append(table_refinement_service)

        d_text_service_id = ""
        if config.USE_PDF_MINER:
            pdf_miner = ServiceFactory.build_pdf_text_detector(config)
            d_text = ServiceFactory.build_pdf_miner_text_service(pdf_miner)
            d_text_service_id = d_text.service_id
            pipe_component_list.append(d_text)

        # setup ocr
        if config.USE_OCR:
            # the extra mile for DocTr
            if config.OCR.USE_DOCTR:
                word_detector = ServiceFactory.build_doctr_word_detector(config)
                word_service = ServiceFactory.build_doctr_word_detector_service(word_detector)
                word_service.set_inbound_filter(skip_if_category_or_service_extracted(service_ids=d_text_service_id))
                pipe_component_list.append(word_service)

            ocr_detector = ServiceFactory.build_ocr_detector(config)
            text_extraction_service = ServiceFactory.build_text_extraction_service(config, ocr_detector)
            text_extraction_service.set_inbound_filter(
                skip_if_category_or_service_extracted(service_ids=d_text_service_id)
            )
            pipe_component_list.append(text_extraction_service)

        if config.USE_PDF_MINER or config.USE_OCR:
            matching_service = ServiceFactory.build_word_matching_service(config)
            pipe_component_list.append(matching_service)

            text_order_service = ServiceFactory.build_text_order_service(config)
            pipe_component_list.append(text_order_service)

        if config.USE_LAYOUT_LINK:
            layout_link_matching_service = ServiceFactory.build_layout_link_matching_service(config)
            pipe_component_list.append(layout_link_matching_service)

        if config.USE_LINE_MATCHER:
            line_list_matching_service = ServiceFactory.build_line_matching_service(config)
            pipe_component_list.append(line_list_matching_service)

        if config.USE_LM_SEQUENCE_CLASS:
            sequence_classifier = ServiceFactory.build_sequence_classifier(config)
            sequence_classifier_service = ServiceFactory.build_sequence_classifier_service(config, sequence_classifier)
            pipe_component_list.append(sequence_classifier_service)

        if config.USE_LM_TOKEN_CLASS:
            token_classifier = ServiceFactory.build_token_classifier(config)
            token_classifier_service = ServiceFactory.build_token_classifier_service(config, token_classifier)
            pipe_component_list.append(token_classifier_service)

        page_parsing_service = ServiceFactory.build_page_parsing_service(config)

        return DoctectionPipe(pipeline_component_list=pipe_component_list, page_parsing_service=page_parsing_service)
