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

from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union

from lazy_imports import try_import

from dd_core.utils.env_info import SETTINGS
from dd_core.utils.error import DependencyError
from dd_core.utils.metacfg import AttrDict
from dd_core.utils.object_types import CellType, LayoutType, ObjectTypes, Relationships
from dd_core.utils.transform import PadTransform

from ..extern.base import ImageTransformer, ObjectDetector, PdfMiner
from ..extern.d2detect import D2FrcnnDetector, D2FrcnnTracingDetector
from ..extern.doctrocr import DocTrRotationTransformer, DoctrTextlineDetector, DoctrTextRecognizer
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
)
from ..extern.hflm import HFLmLanguageDetector, HFLmSequenceClassifier, HFLmTokenClassifier
from ..extern.model import ModelCatalog, ModelDownloadManager, ModelProfile
from ..extern.pdftext import PdfPlumberTextDetector
from ..extern.tessocr import TesseractOcrDetector, TesseractRotationTransformer
from ..extern.texocr import TextractOcrDetector
from ..extern.azurediocr import AzureDocIntelOcrDetector
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
from ..pipe.language import LanguageDetectionService
from ..pipe.layout import ImageLayoutService, skip_if_category_or_service_extracted
from ..pipe.lm import LMSequenceClassifierService, LMTokenClassifierService
from ..pipe.order import TextOrderService
from ..pipe.refine import TableSegmentationRefinementService
from ..pipe.segment import PubtablesSegmentationService, TableSegmentationService
from ..pipe.sub_layout import DetectResultGenerator, SubImageLayoutService
from ..pipe.text import TextExtractionService
from ..pipe.transform import SimpleTransformService

with try_import() as boto_guard:
    from botocore.config import Config  # type: ignore

with try_import() as transformer_guard:
    from transformers import AutoTokenizer

if TYPE_CHECKING:
    from ..extern.hflayoutlm import LayoutSequenceModels, LayoutTokenModels
    from ..extern.hflm import LmSequenceModels, LmTokenModels

    RotationTransformer = Union[TesseractRotationTransformer, DocTrRotationTransformer]

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
    def _get_layout_detector_kwargs_from_config(config: AttrDict, mode: str) -> dict[str, Any]:
        """
        Extracting layout detector kwargs from config.
        Building a D2-Detector, a Detr-Detector or a D2-Torch Tracing Detector according to
        the config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """
        if config.LIB is None:
            raise DependencyError("At least DD_USE_TORCH must be set.")

        weights = (
            getattr(config, mode).WEIGHTS if getattr(config.ENFORCE_WEIGHTS, mode) else getattr(config, mode).WEIGHTS_TS
        )
        filter_categories = getattr(getattr(config, mode), "FILTER")
        profile = ModelCatalog.get_profile(weights)

        if config.LIB == "PT" and profile.padding is not None:
            getattr(config, mode).PADDING = profile.padding
            getattr(config.PT, mode).PADDING = profile.padding

        device = config.DEVICE

        return {
            "weights": weights,
            "filter_categories": filter_categories,
            "profile": profile,
            "device": device,
            "lib": config.LIB,
        }

    @staticmethod
    def _build_layout_detector(
        weights: str,
        filter_categories: list[str],
        profile: ModelProfile,
        device: Literal["cpu", "cuda"],
        lib: Literal["TF", None],
    ) -> Union[D2FrcnnDetector, HFDetrDerivedDetector, D2FrcnnTracingDetector]:
        """
        Building a D2-Detector, a TP-Detector as Detr-Detector or a D2-Torch Tracing Detector according to
        the config.

        Args:
            weights: Weights for the layout detector.
            filter_categories: Categories to filter during detection.
            profile: Model profile for the layout detector.
            device: Device to use for computation.
            lib: Deep learning library to use.
        """
        if lib is None:
            raise DependencyError("At least DD_USE_TORCH must be set.")

        config_path = ModelCatalog.get_full_path_configs(weights)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        categories = profile.categories if profile.categories is not None else {}

        if profile.model_wrapper in ("D2FrcnnDetector",):
            return D2FrcnnDetector(
                path_yaml=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
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
                device=device,
                filter_categories=filter_categories,
            )
        raise TypeError(
            f"You have chosen profile.model_wrapper: {profile.model_wrapper} which is not allowed. Please check "
            f"compatability with your deep learning framework"
        )

    @staticmethod
    def build_layout_detector(
        config: AttrDict, mode: str
    ) -> Union[D2FrcnnDetector, HFDetrDerivedDetector, D2FrcnnTracingDetector]:
        """
        Building a layout detector according to the config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """
        layout_detector_kwargs = ServiceFactory._get_layout_detector_kwargs_from_config(config, mode)
        return ServiceFactory._build_layout_detector(**layout_detector_kwargs)

    @staticmethod
    def _build_rotation_detector(rotator_name: Literal["tesseract", "doctr"]) -> RotationTransformer:
        """
        Building a rotation detector.

        Returns:
            TesseractRotationTransformer: Rotation detector instance.
        """

        if rotator_name == "tesseract":
            return TesseractRotationTransformer()
        if rotator_name == "doctr":
            return DocTrRotationTransformer()
        raise ValueError(
            f"You have chosen rotator_name: {rotator_name} which is not allowed. Only tesseract or "
            f"doctr are allowed."
        )

    @staticmethod
    def build_rotation_detector(rotator_name: Literal["tesseract", "doctr"]) -> RotationTransformer:
        """
        Building a rotation detector.

        Returns:
            TesseractRotationTransformer: Rotation detector instance.
        """
        return ServiceFactory._build_rotation_detector(rotator_name)

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
    def _get_padder_kwargs_from_config(config: AttrDict, mode: str) -> dict[str, Any]:
        """
        Extracting padder kwargs from config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """
        return {
            "top": getattr(config, mode).PAD.TOP,
            "right": getattr(config, mode).PAD.RIGHT,
            "bottom": getattr(config, mode).PAD.BOTTOM,
            "left": getattr(config, mode).PAD.LEFT,
        }

    @staticmethod
    def _build_padder(top: int, right: int, bottom: int, left: int) -> PadTransform:
        """
        Building a padder according to the config.

        Args:
            top: Padding on the top side.
            right: Padding on the right side.
            bottom: Padding on the bottom side.
            left: Padding on the left side.

        Returns:
            PadTransform: `PadTransform` instance.
        """
        return PadTransform(pad_top=top, pad_right=right, pad_bottom=bottom, pad_left=left)

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
        padder_kwargs = ServiceFactory._get_padder_kwargs_from_config(config, mode)
        return ServiceFactory._build_padder(**padder_kwargs)

    @staticmethod
    def _get_layout_service_kwargs_from_config(config: AttrDict, mode: str) -> dict[str, Any]:
        """
        Extracting layout service kwargs from config.

        Args:
            config: Configuration object.
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """
        padder = None
        if getattr(config, mode).PADDING:
            padder = ServiceFactory.build_padder(config, mode=mode)
        return {
            "padder": padder,
        }

    @staticmethod
    def _build_layout_service(detector: ObjectDetector, padder: PadTransform) -> ImageLayoutService:
        """
        Building a layout service with a given detector.

        Args:
            detector: Will be passed to the `ImageLayoutService`.
            padder: PadTransform instance.

        Returns:
            ImageLayoutService: `ImageLayoutService` instance.
        """
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
        layout_service_kwargs = ServiceFactory._get_layout_service_kwargs_from_config(config, mode)
        return ServiceFactory._build_layout_service(detector, **layout_service_kwargs)

    @staticmethod
    def _get_layout_nms_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting layout NMS service kwargs from config.

        Args:
            config: Configuration object.
        """

        if not isinstance(config.LAYOUT_NMS_PAIRS.COMBINATIONS, list) and not isinstance(
            config.LAYOUT_NMS_PAIRS.COMBINATIONS[0], list
        ):
            raise ValueError("LAYOUT_NMS_PAIRS must be a list of lists")

        return {
            "nms_pairs": config.LAYOUT_NMS_PAIRS.COMBINATIONS,
            "thresholds": config.LAYOUT_NMS_PAIRS.THRESHOLDS,
            "priority": config.LAYOUT_NMS_PAIRS.PRIORITY,
        }

    @staticmethod
    def _build_layout_nms_service(
        nms_pairs: Sequence[Sequence[Union[ObjectTypes, str]]],
        thresholds: Union[float, Sequence[float]],
        priority: Sequence[Union[ObjectTypes, str, None]],
    ) -> AnnotationNmsService:
        """
        Building a NMS service for layout annotations.

        Args:
            nms_pairs: Pairs of categories for NMS.
            thresholds: NMS thresholds.
            priority: Priority of categories.

        Returns:
            AnnotationNmsService: NMS service instance.
        """

        return AnnotationNmsService(
            nms_pairs=nms_pairs,
            thresholds=thresholds,
            priority=priority,
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
        nms_service_kwargs = ServiceFactory._get_layout_nms_service_kwargs_from_config(config)
        return ServiceFactory._build_layout_nms_service(**nms_service_kwargs)

    @staticmethod
    def _get_sub_image_layout_service_kwargs_from_config(detector: ObjectDetector, mode: str) -> dict[str, Any]:
        """
        Extracting sub image service kwargs from config.

        Args:
            mode: Either `LAYOUT`, `CELL`, or `ITEM`.
        """

        exclude_category_names = []
        if mode == "ITEM":
            if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
                exclude_category_names.extend(
                    [LayoutType.TABLE, CellType.COLUMN_HEADER, CellType.PROJECTED_ROW_HEADER, CellType.SPANNING]
                )
        return {"exclude_category_names": exclude_category_names}

    @staticmethod
    def _build_sub_image_service(
        detector: ObjectDetector, padder: Optional[PadTransform], exclude_category_names: list[ObjectTypes]
    ) -> SubImageLayoutService:
        """
        Building a sub image layout service with a given detector.

        Args:
            detector: Will be passed to the `SubImageLayoutService`.
            padder: PadTransform instance.
            exclude_category_names: Category names to exclude during detection.

        Returns:
            SubImageLayoutService: `SubImageLayoutService` instance.
        """
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
        padder = None
        if mode == "ITEM":
            padder = ServiceFactory.build_padder(config, mode)
        sub_image_layout_service_kwargs = ServiceFactory._get_sub_image_layout_service_kwargs_from_config(
            detector, mode
        )
        return ServiceFactory._build_sub_image_service(detector, padder, **sub_image_layout_service_kwargs)

    @staticmethod
    def _get_ocr_detector_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting OCR detector kwargs from config.

        Args:
            config: Configuration object.
        """
        ocr_config_path = None
        weights = None
        languages = None
        credentials_kwargs = {}
        use_tesseract = False
        use_doctr = False
        use_textract = False
        use_azure_di = False

        if config.OCR.USE_TESSERACT:
            use_tesseract = True
            ocr_config_path = SETTINGS.CONFIGS_DIR / config.OCR.CONFIG.TESSERACT
            languages = [f"LANGUAGES={config.LANGUAGE}"] if config.LANGUAGE is not None else None

        if config.OCR.USE_DOCTR:
            use_doctr = True
            if config.LIB is None:
                raise DependencyError("At least DD_USE_TORCH must be set.")
            weights = config.OCR.WEIGHTS.DOCTR_RECOGNITION
        if config.OCR.USE_TEXTRACT:
            use_textract = True
            if SETTINGS.AWS_REGION and SETTINGS.AWS_ACCESS_KEY_ID and SETTINGS.AWS_SECRET_ACCESS_KEY:
                credentials_kwargs = {
                    "aws_access_key_id": SETTINGS.AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": SETTINGS.AWS_SECRET_ACCESS_KEY,
                    "config": Config(region_name=SETTINGS.AWS_REGION),
                }
        if config.OCR.USE_AZURE_DI:
            use_azure_di = True
            if SETTINGS.AZURE_DI_ENDPOINT and SETTINGS.AZURE_DI_KEY:
                credentials_kwargs = {
                    "endpoint": SETTINGS.AZURE_DI_ENDPOINT,
                    "api_key": SETTINGS.AZURE_DI_KEY,
                }

        return {
            "use_tesseract": use_tesseract,
            "use_doctr": use_doctr,
            "use_textract": use_textract,
            "use_azure_di": use_azure_di,
            "ocr_config_path": ocr_config_path,
            "languages": languages,
            "weights": weights,
            "credentials_kwargs": credentials_kwargs,
            "lib": config.LIB,
            "device": config.DEVICE,
        }

    @staticmethod
    def _build_ocr_detector(
        use_tesseract: bool,
        use_doctr: bool,
        use_textract: bool,
        use_azure_di: bool,
        ocr_config_path: str,
        languages: Union[list[str], None],
        weights: str,
        credentials_kwargs: dict[str, Any],
        lib: Literal["TF", "PT", None],
        device: Literal["cuda", "cpu"],
    ) -> Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector, AzureDocIntelOcrDetector]:
        """
        Building OCR predictor.

        Args:
            use_tesseract: Whether to use Tesseract OCR.
            use_doctr: Whether to use Doctr OCR.
            use_textract: Whether to use Textract OCR.
            use_azure_di: Whether to use Azure Document Intelligence OCR.
            ocr_config_path: Path to OCR config.
            languages: Languages for OCR.
            weights: Weights for Doctr OCR.
            credentials_kwargs: Credentials for Textract or Azure DI OCR.
            lib: Deep learning library to use.
            device: Device to use for computation.

        Returns:
            Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector, AzureDocIntelOcrDetector]: OCR detector instance.
        """
        if use_tesseract:
            return TesseractOcrDetector(
                ocr_config_path,
                config_overwrite=languages,
            )
        if use_doctr:
            if lib is None:
                raise DependencyError("At least one of the env variables DD_USE_TF or DD_USE_TORCH must be set.")
            weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
            profile = ModelCatalog.get_profile(weights)
            # get_full_path_configs will complete the path even if the model is not registered
            config_path = ModelCatalog.get_full_path_configs(weights) if profile.config is not None else None
            if profile.architecture is None:
                raise ValueError("model profile.architecture must be specified")
            return DoctrTextRecognizer(
                architecture=profile.architecture,
                path_weights=weights_path,
                device=device,
                path_config_json=config_path,
            )
        if use_textract:
            return TextractOcrDetector(**credentials_kwargs)
        if use_azure_di:
            return AzureDocIntelOcrDetector(**credentials_kwargs)
        raise ValueError("You have set USE_OCR=True but any of USE_TESSERACT, USE_DOCTR, USE_TEXTRACT, USE_AZURE_DI is set to False")

    @staticmethod
    def build_ocr_detector(config: AttrDict) -> Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector, AzureDocIntelOcrDetector]:
        """
        Building OCR predictor.

        Args:
            config: Configuration object.

        Returns:
            Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector]: OCR detector instance.
        """
        ocr_detector_kwargs = ServiceFactory._get_ocr_detector_kwargs_from_config(config)
        return ServiceFactory._build_ocr_detector(**ocr_detector_kwargs)

    @staticmethod
    def _get_doctr_word_detector_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting Doctr word detector kwargs from config.

        Args:
            config: Configuration object.
        """
        weights = config.OCR.WEIGHTS.DOCTR_WORD
        profile = ModelCatalog.get_profile(weights)
        return {
            "weights": weights,
            "profile": profile,
            "device": config.DEVICE,
            "lib": config.LIB,
        }

    @staticmethod
    def _build_doctr_word_detector(
        weights: str, profile: ModelProfile, device: Literal["cuda", "cpu"], lib: Literal["PT"]
    ) -> DoctrTextlineDetector:
        """
        Building `DoctrTextlineDetector` instance.

        Args:
            weights: Weights for Doctr word detector.
            profile: Model profile for Doctr word detector.
            device: Device to use for computation.
            lib: Deep learning library to use.

        Returns:
            DoctrTextlineDetector: Textline detector instance.
        """
        if lib is None:
            raise DependencyError("At least one of the env variable DD_USE_TORCH must be set.")
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(weights)
        if profile.architecture is None:
            raise ValueError("model profile.architecture must be specified")
        if profile.categories is None:
            raise ValueError("model profile.categories must be specified")
        return DoctrTextlineDetector(profile.architecture, weights_path, profile.categories, device)

    @staticmethod
    def build_doctr_word_detector(config: AttrDict) -> DoctrTextlineDetector:
        """
        Building `DoctrTextlineDetector` instance.

        Args:
            config: Configuration object.

        Returns:
            DoctrTextlineDetector: Textline detector instance.
        """
        doctr_word_detector_kwargs = ServiceFactory._get_doctr_word_detector_kwargs_from_config(config)
        return ServiceFactory._build_doctr_word_detector(**doctr_word_detector_kwargs)

    @staticmethod
    def _get_table_segmentation_service_kwargs_from_config(config: AttrDict, detector_name: str) -> dict[str, Any]:
        """
        Extracting table segmentation service kwargs from config.

        Args:
            config: Configuration object.
            detector_name: An instance name of `ObjectDetector`.
        """
        return {
            "segment_rule": config.SEGMENTATION.ASSIGNMENT_RULE,
            "threshold_rows": config.SEGMENTATION.THRESHOLD_ROWS,
            "threshold_cols": config.SEGMENTATION.THRESHOLD_COLS,
            "tile_table_with_items": config.SEGMENTATION.FULL_TABLE_TILING,
            "remove_iou_threshold_rows": config.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
            "remove_iou_threshold_cols": config.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
            "table_name": config.SEGMENTATION.TABLE_NAME,
            "cell_names": (
                config.SEGMENTATION.PUBTABLES_CELL_NAMES
                if detector_name in ("HFDetrDerivedDetector",)
                else config.SEGMENTATION.CELL_NAMES
            ),
            "spanning_cell_names": config.SEGMENTATION.PUBTABLES_SPANNING_CELL_NAMES,
            "item_names": (
                config.SEGMENTATION.PUBTABLES_ITEM_NAMES
                if detector_name in ("HFDetrDerivedDetector",)
                else config.SEGMENTATION.ITEM_NAMES
            ),
            "sub_item_names": (
                config.SEGMENTATION.PUBTABLES_SUB_ITEM_NAMES
                if detector_name in ("HFDetrDerivedDetector",)
                else config.SEGMENTATION.SUB_ITEM_NAMES
            ),
            "item_header_cell_names": config.SEGMENTATION.PUBTABLES_ITEM_HEADER_CELL_NAMES,
            "item_header_thresholds": config.SEGMENTATION.PUBTABLES_ITEM_HEADER_THRESHOLDS,
            "stretch_rule": config.SEGMENTATION.STRETCH_RULE,
        }

    @staticmethod
    def _build_table_segmentation_service(
        detector: ObjectDetector,
        segment_rule: Literal["iou", "ioa"],
        threshold_rows: float,
        threshold_cols: float,
        tile_table_with_items: bool,
        remove_iou_threshold_rows: float,
        remove_iou_threshold_cols: float,
        table_name: Union[ObjectTypes, str],
        cell_names: Sequence[Union[ObjectTypes, str]],
        spanning_cell_names: Sequence[Union[ObjectTypes, str]],
        item_names: Sequence[Union[ObjectTypes, str]],
        sub_item_names: Sequence[Union[ObjectTypes, str]],
        item_header_cell_names: Sequence[Union[ObjectTypes, str]],
        item_header_thresholds: Sequence[float],
        stretch_rule: Literal["left", "equal"],
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
            detector: An instance of `ObjectDetector` used to determine the type of table segmentation service to build.
            segment_rule: Rule for segmenting tables.
            threshold_rows: Threshold for row segmentation.
            threshold_cols: Threshold for column segmentation.
            tile_table_with_items: Whether to tile the table with items.
            remove_iou_threshold_rows: IOU threshold for removing rows.
            remove_iou_threshold_cols: IOU threshold for removing columns.
            table_name: Name of the table object type.
            cell_names: Names of the cell object types.
            spanning_cell_names: Names of the spanning cell object types.
            item_names: Names of the item object types.
            sub_item_names: Names of the sub-item object types.
            item_header_cell_names: Names of the item header cell object types.
            item_header_thresholds: Thresholds for item header segmentation.
            stretch_rule: Rule for stretching cells.

        Returns:
            Table segmentation service instance.
        """
        table_segmentation: Union[PubtablesSegmentationService, TableSegmentationService]
        if detector.__class__.__name__ in ("HFDetrDerivedDetector",):
            table_segmentation = PubtablesSegmentationService(
                segment_rule=segment_rule,
                threshold_rows=threshold_rows,
                threshold_cols=threshold_cols,
                tile_table_with_items=tile_table_with_items,
                remove_iou_threshold_rows=remove_iou_threshold_rows,
                remove_iou_threshold_cols=remove_iou_threshold_cols,
                table_name=table_name,
                cell_names=cell_names,
                spanning_cell_names=spanning_cell_names,
                item_names=item_names,
                sub_item_names=sub_item_names,
                item_header_cell_names=item_header_cell_names,
                item_header_thresholds=item_header_thresholds,
                stretch_rule=stretch_rule,
            )

        else:
            table_segmentation = TableSegmentationService(
                segment_rule=segment_rule,
                threshold_rows=threshold_rows,
                threshold_cols=threshold_cols,
                tile_table_with_items=tile_table_with_items,
                remove_iou_threshold_rows=remove_iou_threshold_rows,
                remove_iou_threshold_cols=remove_iou_threshold_cols,
                table_name=table_name,
                cell_names=cell_names,
                item_names=item_names,
                sub_item_names=sub_item_names,
                stretch_rule=stretch_rule,
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
        table_segmentation_service_kwargs = ServiceFactory._get_table_segmentation_service_kwargs_from_config(
            config, detector.__class__.__name__
        )
        return ServiceFactory._build_table_segmentation_service(detector, **table_segmentation_service_kwargs)

    @staticmethod
    def _get_table_refinement_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting table segmentation refinement service kwargs from config.

        Args:
            config: Configuration object.
        """

        return {
            "table_names": [config.SEGMENTATION.TABLE_NAME],
            "cell_names": config.SEGMENTATION.PUBTABLES_CELL_NAMES,
        }

    @staticmethod
    def _build_table_refinement_service(
        table_names: Sequence[ObjectTypes], cell_names: Sequence[ObjectTypes]
    ) -> TableSegmentationRefinementService:
        """
        Building a table segmentation refinement service.

        Args:
            table_names: Names of the table object types.
            cell_names: Names of the cell object types.

        Returns:
            TableSegmentationRefinementService: Refinement service instance.
        """
        return TableSegmentationRefinementService(table_names=table_names, cell_names=cell_names)

    @staticmethod
    def build_table_refinement_service(config: AttrDict) -> TableSegmentationRefinementService:
        """
        Building a table segmentation refinement service.

        Args:
            config: Configuration object.

        Returns:
            TableSegmentationRefinementService: Refinement service instance.
        """
        table_refinement_service_kwargs = ServiceFactory._get_table_refinement_service_kwargs_from_config(config)
        return ServiceFactory._build_table_refinement_service(**table_refinement_service_kwargs)

    @staticmethod
    def _get_pdf_text_detector_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting PDF text detector kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "x_tolerance": config.PDF_MINER.X_TOLERANCE,
            "y_tolerance": config.PDF_MINER.Y_TOLERANCE,
        }

    @staticmethod
    def _build_pdf_text_detector(x_tolerance: int, y_tolerance: int) -> PdfPlumberTextDetector:
        """
        Building a PDF text detector.

        Args:
            x_tolerance: X tolerance for text extraction.
            y_tolerance: Y tolerance for text extraction.

        Returns:
            PdfPlumberTextDetector: PDF text detector instance.
        """
        return PdfPlumberTextDetector(x_tolerance=x_tolerance, y_tolerance=y_tolerance)

    @staticmethod
    def build_pdf_text_detector(config: AttrDict) -> PdfPlumberTextDetector:
        """
        Building a PDF text detector.

        Args:
            config: Configuration object.

        Returns:
            PdfPlumberTextDetector: PDF text detector instance.
        """
        pdf_text_detector_kwargs = ServiceFactory._get_pdf_text_detector_kwargs_from_config(config)
        return ServiceFactory._build_pdf_text_detector(**pdf_text_detector_kwargs)

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
    def _get_text_extraction_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting text extraction service kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "extract_from_roi": config.TEXT_CONTAINER if config.OCR.USE_DOCTR else None,
        }

    @staticmethod
    def _build_text_extraction_service(
        detector: Union[TesseractOcrDetector, DoctrTextRecognizer, TextractOcrDetector],
        extract_from_roi: Union[Sequence[ObjectTypes], ObjectTypes, None] = None,
    ) -> TextExtractionService:
        """
        Building a text extraction service.

        Args:
            detector: OCR detector instance.
            extract_from_roi: ROI categories to extract text from.

        Returns:
            TextExtractionService: Text extraction service instance.
        """
        return TextExtractionService(detector, extract_from_roi=extract_from_roi)

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
        text_extraction_service_kwargs = ServiceFactory._get_text_extraction_service_kwargs_from_config(config)
        return ServiceFactory._build_text_extraction_service(detector, **text_extraction_service_kwargs)

    @staticmethod
    def _get_word_matching_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting word matching service kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "matching_rule": config.WORD_MATCHING.RULE,
            "threshold": config.WORD_MATCHING.THRESHOLD,
            "max_parent_only": config.WORD_MATCHING.MAX_PARENT_ONLY,
            "parental_categories": config.WORD_MATCHING.PARENTAL_CATEGORIES,
            "text_container": config.TEXT_CONTAINER,
        }

    @staticmethod
    def _build_word_matching_service(
        matching_rule: Literal["iou", "ioa"],
        threshold: float,
        max_parent_only: bool,
        parental_categories: Union[Sequence[ObjectTypes], ObjectTypes, None],
        text_container: Union[Sequence[ObjectTypes], ObjectTypes, None],
    ) -> MatchingService:
        """
        Building a word matching service.

        Args:
            matching_rule: Matching rule for intersection matcher.
            threshold: Threshold for intersection matcher.
            max_parent_only: Whether to use max parent only.
            parental_categories: Parent categories for matching.
            text_container: Text container categories.

        Returns:
            MatchingService: Word matching service instance.
        """
        matcher = IntersectionMatcher(
            matching_rule=matching_rule,
            threshold=threshold,
            max_parent_only=max_parent_only,
        )
        family_compounds = [
            FamilyCompound(
                parent_categories=parental_categories,
                child_categories=text_container,
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
        word_matching_service_kwargs = ServiceFactory._get_word_matching_service_kwargs_from_config(config)
        return ServiceFactory._build_word_matching_service(**word_matching_service_kwargs)

    @staticmethod
    def _get_layout_link_matching_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting layout link matching service kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "parental_categories": config.LAYOUT_LINK.PARENTAL_CATEGORIES,
            "child_categories": config.LAYOUT_LINK.CHILD_CATEGORIES,
        }

    @staticmethod
    def _build_layout_link_matching_service(
        parental_categories: Union[Sequence[ObjectTypes], ObjectTypes, None],
        child_categories: Union[Sequence[ObjectTypes], ObjectTypes, None],
    ) -> MatchingService:
        """
        Building a layout link matching service.

        Args:
            parental_categories: Parent categories for layout linking.
            child_categories: Child categories for layout linking.

        Returns:
            MatchingService: Layout link matching service instance.
        """
        neighbor_matcher = NeighbourMatcher()
        family_compounds = [
            FamilyCompound(
                parent_categories=parental_categories,
                child_categories=child_categories,
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
        layout_link_matching_service_kwargs = ServiceFactory._get_layout_link_matching_service_kwargs_from_config(
            config
        )
        return ServiceFactory._build_layout_link_matching_service(**layout_link_matching_service_kwargs)

    @staticmethod
    def _get_line_matching_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting line matching service kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "matching_rule": config.WORD_MATCHING.RULE,
            "threshold": config.WORD_MATCHING.THRESHOLD,
            "max_parent_only": config.WORD_MATCHING.MAX_PARENT_ONLY,
        }

    @staticmethod
    def _build_line_matching_service(
        matching_rule: Literal["iou", "ioa"], threshold: float, max_parent_only: bool
    ) -> MatchingService:
        """
        Building a line matching service.

        Args:
            matching_rule: Matching rule for intersection matcher.
            threshold: Threshold for intersection matcher.
            max_parent_only: Whether to use max parent only.

        Returns:
            MatchingService: Line matching service instance.
        """
        matcher = IntersectionMatcher(
            matching_rule=matching_rule,
            threshold=threshold,
            max_parent_only=max_parent_only,
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
        line_matching_service_kwargs = ServiceFactory._get_line_matching_service_kwargs_from_config(config)
        return ServiceFactory._build_line_matching_service(**line_matching_service_kwargs)

    @staticmethod
    def _get_text_order_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting text order service kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "text_container": config.TEXT_CONTAINER,
            "text_block_categories": config.TEXT_ORDERING.TEXT_BLOCK_CATEGORIES,
            "floating_text_block_categories": config.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            "include_residual_text_container": config.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
            "starting_point_tolerance": config.TEXT_ORDERING.STARTING_POINT_TOLERANCE,
            "broken_line_tolerance": config.TEXT_ORDERING.BROKEN_LINE_TOLERANCE,
            "height_tolerance": config.TEXT_ORDERING.HEIGHT_TOLERANCE,
            "paragraph_break": config.TEXT_ORDERING.PARAGRAPH_BREAK,
        }

    @staticmethod
    def _build_text_order_service(
        text_container: str,
        text_block_categories: Sequence[str],
        floating_text_block_categories: Sequence[str],
        include_residual_text_container: bool,
        starting_point_tolerance: float,
        broken_line_tolerance: float,
        height_tolerance: float,
        paragraph_break: float,
    ) -> TextOrderService:
        """
        Building a text order service.

        Args:
            text_container: Text container categories.
            text_block_categories: Text block categories for ordering.
            floating_text_block_categories: Floating text block categories.
            include_residual_text_container: Whether to include residual text container.
            starting_point_tolerance: Starting point tolerance for text ordering.
            broken_line_tolerance: Broken line tolerance for text ordering.
            height_tolerance: Height tolerance for text ordering.
            paragraph_break: Paragraph break threshold.

        Returns:
            TextOrderService: Text order service instance.
        """
        return TextOrderService(
            text_container=text_container,
            text_block_categories=text_block_categories,
            floating_text_block_categories=floating_text_block_categories,
            include_residual_text_container=include_residual_text_container,
            starting_point_tolerance=starting_point_tolerance,
            broken_line_tolerance=broken_line_tolerance,
            height_tolerance=height_tolerance,
            paragraph_break=paragraph_break,
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
        text_order_service_kwargs = ServiceFactory._get_text_order_service_kwargs_from_config(config)
        return ServiceFactory._build_text_order_service(**text_order_service_kwargs)

    @staticmethod
    def _get_language_detector_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting language detector kwargs from config.

        Args:
            config: Configuration object.
        """
        config_path = ModelCatalog.get_full_path_configs(config.LM_LANGUAGE_DETECT_CLASS.WEIGHTS)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(config.LM_LANGUAGE_DETECT_CLASS.WEIGHTS)
        profile = ModelCatalog.get_profile(config.LM_LANGUAGE_DETECT_CLASS.WEIGHTS)
        config_dir = ModelCatalog.get_full_path_configs_dir(config.LM_LANGUAGE_DETECT_CLASS.WEIGHTS)
        categories = profile.categories if profile.categories is not None else {}

        return {
            "config_path": config_path,
            "weights_path": weights_path,
            "categories": categories,
            "device": config.DEVICE,
            "model_wrapper": profile.model_wrapper,
            "tokenizer_config_dir": config_dir,
        }

    @staticmethod
    def _build_language_detector(
        config_path: str,
        weights_path: str,
        categories: Mapping[int, Union[ObjectTypes, str]],
        device: Literal["cuda", "cpu"],
        model_wrapper: str,
        tokenizer_config_dir: str,
    ) -> HFLmLanguageDetector:
        """
        Builds and returns a language detector instance.

        Args:
            config_path: Path to model configuration.
            weights_path: Path to model weights.
            categories: Model categories mapping.
            device: Device to run model on.
            model_wrapper: Model wrapper class name.

        Returns:
            A language detector instance.
        """
        if model_wrapper in ("HFLmLanguageDetector",):

            return HFLmLanguageDetector(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
                tokenizer_config_dir=tokenizer_config_dir,
            )

        raise ValueError(f"Unsupported language detector model wrapper: {model_wrapper}")

    @staticmethod
    def build_language_detector(config: AttrDict) -> HFLmLanguageDetector:
        """
        Builds and returns a language detector instance.

        Args:
            config: Configuration object that determines the type of language detector to construct.

        Returns:
            A language detector instance constructed according to the specified configuration.
        """
        language_detector_kwargs = ServiceFactory._get_language_detector_kwargs_from_config(config)
        return ServiceFactory._build_language_detector(**language_detector_kwargs)

    @staticmethod
    def _build_language_detection_service(language_detector: Any) -> LanguageDetectionService:
        """
        Building a language detection service.

        Args:
            language_detector: Language detector instance.

        Returns:
            LanguageDetectionService: Language detection service instance.
        """

        return LanguageDetectionService(language_detector=language_detector)

    @staticmethod
    def build_language_detection_service(config: AttrDict) -> LanguageDetectionService:
        """
        Building a language detection service.

        Args:
            config: Configuration object.

        Returns:
            LanguageDetectionService: Language detection service instance.
        """
        language_detector = ServiceFactory.build_language_detector(config)
        return ServiceFactory._build_language_detection_service(language_detector)

    @staticmethod
    def _get_sequence_classifier_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting sequence classifier kwargs from config.

        Args:
            config: Configuration object.
        """
        config_path = ModelCatalog.get_full_path_configs(config.LM_SEQUENCE_CLASS.WEIGHTS)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(config.LM_SEQUENCE_CLASS.WEIGHTS)
        profile = ModelCatalog.get_profile(config.LM_SEQUENCE_CLASS.WEIGHTS)
        categories = profile.categories if profile.categories is not None else {}

        return {
            "config_path": config_path,
            "weights_path": weights_path,
            "categories": categories,
            "device": config.DEVICE,
            "model_wrapper": profile.model_wrapper,
        }

    @staticmethod
    def _build_sequence_classifier(
        config_path: str,
        weights_path: str,
        categories: Mapping[int, Union[ObjectTypes, str]],
        device: Literal["cuda", "cpu"],
        model_wrapper: str,
    ) -> Union[LayoutSequenceModels, LmSequenceModels]:
        """
        Builds and returns a sequence classifier instance.

        Args:
            config_path: Path to model configuration.
            weights_path: Path to model weights.
            categories: Model categories mapping.
            device: Device to run model on.
            model_wrapper: Model wrapper class name.

        Returns:
            A sequence classifier instance constructed according to the specified configuration.
        """
        if model_wrapper in ("HFLayoutLmSequenceClassifier",):
            return HFLayoutLmSequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLayoutLmv2SequenceClassifier",):
            return HFLayoutLmv2SequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLayoutLmv3SequenceClassifier",):
            return HFLayoutLmv3SequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLiltSequenceClassifier",):
            return HFLiltSequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLmSequenceClassifier",):
            return HFLmSequenceClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        raise ValueError(f"Unsupported model wrapper: {model_wrapper}")

    @staticmethod
    def build_sequence_classifier(config: AttrDict) -> Union[LayoutSequenceModels, LmSequenceModels]:
        """
        Builds and returns a sequence classifier instance.

        Args:
            config: Configuration object that determines the type of sequence classifier to construct.

        Returns:
            A sequence classifier instance constructed according to the specified configuration.
        """
        sequence_classifier_kwargs = ServiceFactory._get_sequence_classifier_kwargs_from_config(config)
        return ServiceFactory._build_sequence_classifier(**sequence_classifier_kwargs)

    @staticmethod
    def _get_sequence_classifier_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting sequence classifier service kwargs from config.

        Args:
            config: Configuration object.
        """
        config_dir = ModelCatalog.get_full_path_configs_dir(config.LM_SEQUENCE_CLASS.WEIGHTS)
        tokenizer_fast = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config_dir)
        return {
            "tokenizer_fast": tokenizer_fast,
            "use_other_as_default_category": config.LM_SEQUENCE_CLASS.USE_OTHER_AS_DEFAULT_CATEGORY,
        }

    @staticmethod
    def _build_sequence_classifier_service(
        sequence_classifier: Union[LayoutSequenceModels, LmSequenceModels],
        tokenizer_fast: Any,
        use_other_as_default_category: bool,
    ) -> LMSequenceClassifierService:
        """
        Building a sequence classifier service.

        Args:
            sequence_classifier: Sequence classifier instance.
            tokenizer_fast: Fast Tokenizer instance.
            use_other_as_default_category: Whether to use other as default category.

        Returns:
            LMSequenceClassifierService: Text order service instance.
        """

        return LMSequenceClassifierService(
            tokenizer=tokenizer_fast,
            language_model=sequence_classifier,
            use_other_as_default_category=use_other_as_default_category,
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
        sequence_classifier_service_kwargs = ServiceFactory._get_sequence_classifier_service_kwargs_from_config(config)
        return ServiceFactory._build_sequence_classifier_service(
            sequence_classifier, **sequence_classifier_service_kwargs
        )

    @staticmethod
    def _get_token_classifier_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting token classifier kwargs from config.

        Args:
            config: Configuration object.
        """
        config_path = ModelCatalog.get_full_path_configs(config.LM_TOKEN_CLASS.WEIGHTS)
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs(config.LM_TOKEN_CLASS.WEIGHTS)
        profile = ModelCatalog.get_profile(config.LM_TOKEN_CLASS.WEIGHTS)
        categories = profile.categories if profile.categories is not None else {}

        return {
            "config_path": config_path,
            "weights_path": weights_path,
            "categories": categories,
            "device": config.DEVICE,
            "model_wrapper": profile.model_wrapper,
        }

    @staticmethod
    def _build_token_classifier(
        config_path: str,
        weights_path: str,
        categories: Mapping[int, Union[ObjectTypes, str]],
        device: Literal["cpu", "cuda"],
        model_wrapper: str,
    ) -> Union[LayoutTokenModels, LmTokenModels]:
        """
        Builds and returns a token classifier model.

        Args:
            config_path: Path to model configuration.
            weights_path: Path to model weights.
            categories: Model categories mapping.
            device: Device to run model on.
            model_wrapper: Model wrapper class name.

        Returns:
            The instantiated token classifier model.
        """
        if model_wrapper in ("HFLayoutLmTokenClassifier",):
            return HFLayoutLmTokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLayoutLmv2TokenClassifier",):
            return HFLayoutLmv2TokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLayoutLmv3TokenClassifier",):
            return HFLayoutLmv3TokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLiltTokenClassifier",):
            return HFLiltTokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
                device=device,
            )
        if model_wrapper in ("HFLmTokenClassifier",):
            return HFLmTokenClassifier(
                path_config_json=config_path,
                path_weights=weights_path,
                categories=categories,
            )
        raise ValueError(f"Unsupported model wrapper: {model_wrapper}")

    @staticmethod
    def build_token_classifier(config: AttrDict) -> Union[LayoutTokenModels, LmTokenModels]:
        """
        Builds and returns a token classifier model.

        Args:
            config: Configuration object.

        Returns:
            The instantiated token classifier model.
        """
        token_classifier_kwargs = ServiceFactory._get_token_classifier_kwargs_from_config(config)
        return ServiceFactory._build_token_classifier(**token_classifier_kwargs)

    @staticmethod
    def _get_token_classifier_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting token classifier service kwargs from config.

        Args:
            config: Configuration object.
        """
        config_dir = ModelCatalog.get_full_path_configs_dir(config.LM_TOKEN_CLASS.WEIGHTS)
        tokenizer_fast = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config_dir)
        return {
            "tokenizer_fast": tokenizer_fast,
            "use_other_as_default_category": config.LM_TOKEN_CLASS.USE_OTHER_AS_DEFAULT_CATEGORY,
            "segment_positions": config.LM_TOKEN_CLASS.SEGMENT_POSITIONS,
            "sliding_window_stride": config.LM_TOKEN_CLASS.SLIDING_WINDOW_STRIDE,
        }

    @staticmethod
    def _build_token_classifier_service(
        token_classifier: Union[LayoutTokenModels, LmTokenModels],
        tokenizer_fast: Any,
        use_other_as_default_category: bool,
        segment_positions: Union[LayoutType, Sequence[LayoutType], None],
        sliding_window_stride: int,
    ) -> LMTokenClassifierService:
        """
        Building a token classifier service.

        Args:
            token_classifier: Token classifier instance.
            use_other_as_default_category: Whether to use other as default category.
            segment_positions: Segment positions configuration.
            sliding_window_stride: Sliding window stride.

        Returns:
             A LMTokenClassifierService instance.
        """

        return LMTokenClassifierService(
            tokenizer=tokenizer_fast,
            language_model=token_classifier,
            use_other_as_default_category=use_other_as_default_category,
            segment_positions=segment_positions,
            sliding_window_stride=sliding_window_stride,
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
        token_classifier_service_kwargs = ServiceFactory._get_token_classifier_service_kwargs_from_config(config)
        return ServiceFactory._build_token_classifier_service(token_classifier, **token_classifier_service_kwargs)

    @staticmethod
    def _get_page_parsing_service_kwargs_from_config(config: AttrDict) -> dict[str, Any]:
        """
        Extracting page parsing service kwargs from config.

        Args:
            config: Configuration object.
        """
        return {
            "text_container": config.TEXT_CONTAINER,
            "floating_text_block_categories": config.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES,
            "include_residual_text_container": config.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER,
        }

    @staticmethod
    def _build_page_parsing_service(
        text_container: Union[ObjectTypes, str],
        floating_text_block_categories: Sequence[str],
        include_residual_text_container: bool,
    ) -> PageParsingService:
        """
        Building a page parsing service.

        Args:
            text_container: Text container categories.
            floating_text_block_categories: Floating text block categories.
            include_residual_text_container: Whether to include residual text container.

        Returns:
            PageParsingService: Page parsing service instance.
        """
        return PageParsingService(
            text_container=text_container,
            floating_text_block_categories=floating_text_block_categories,
            include_residual_text_container=include_residual_text_container,
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
        page_parsing_service_kwargs = ServiceFactory._get_page_parsing_service_kwargs_from_config(config)
        return ServiceFactory._build_page_parsing_service(**page_parsing_service_kwargs)

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
            rotation_detector = ServiceFactory.build_rotation_detector(config.ROTATOR.MODEL)
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

        if config.USE_LM_LANGUAGE_DETECTION:
            language_detection_service = ServiceFactory.build_language_detection_service(config)
            pipe_component_list.append(language_detection_service)

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
