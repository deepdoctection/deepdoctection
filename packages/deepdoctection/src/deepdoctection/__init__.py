# -*- coding: utf-8 -*-
# File: __init__.py

"""
Init file for deepdoctection package. This file is used to import all submodules and to set some environment variables
"""

import os
import sys
from typing import TYPE_CHECKING

# Import from dd_core (utils and datapoint are now external dependencies)
from dd_core.utils.env_info import collect_env_info
from dd_core.utils.file_utils import _LazyModule, pytorch_available
from dd_core.utils.logger import LoggingRecord, logger

__version__ = "1.0"

_IMPORT_STRUCTURE = {
    "analyzer": ["config_sanity_checks", "get_dd_analyzer", "ServiceFactory", "update_cfg_from_defaults"],
    "eval": [
        "AccuracyMetric",
        "ConfusionMetric",
        "PrecisionMetric",
        "RecallMetric",
        "F1Metric",
        "PrecisionMetricMicro",
        "RecallMetricMicro",
        "F1MetricMicro",
        "MetricBase",
        "CocoMetric",
        "Evaluator",
        "metric_registry",
        "get_metric",
        "TableTree",
        "CustomConfig",
        "TEDS",
        "TedsMetric",
    ],
    "extern": [
        "ModelCategories",
        "NerModelCategories",
        "PredictorBase",
        "DetectionResult",
        "ObjectDetector",
        "PdfMiner",
        "TextRecognizer",
        "TokenClassResult",
        "SequenceClassResult",
        "LMTokenClassifier",
        "LMSequenceClassifier",
        "LanguageDetector",
        "ImageTransformer",
        "DeterministicImageTransformer",
        "InferenceResize",
        "D2FrcnnDetector",
        "D2FrcnnTracingDetector",
        "Jdeskewer",
        "DoctrTextlineDetector",
        "DoctrTextRecognizer",
        "DocTrRotationTransformer",
        "HFDetrDerivedDetector",
        "get_tokenizer_from_architecture",
        "HFLayoutLmTokenClassifierBase",
        "HFLayoutLmTokenClassifier",
        "HFLayoutLmv2TokenClassifier",
        "HFLayoutLmv3TokenClassifier",
        "HFLayoutLmSequenceClassifier",
        "HFLayoutLmv2SequenceClassifier",
        "HFLayoutLmv3SequenceClassifier",
        "HFLiltTokenClassifier",
        "HFLiltSequenceClassifier",
        "HFLmTokenClassifier",
        "HFLmSequenceClassifier",
        "HFLmLanguageDetector",
        "ModelProfile",
        "ModelCatalog",
        "print_model_infos",
        "ModelDownloadManager",
        "PdfPlumberTextDetector",
        "Pdfmium2TextDetector",
        "TesseractOcrDetector",
        "TesseractRotationTransformer",
        "TextractOcrDetector",
    ],
    "pipe": [
        "DatapointManager",
        "PipelineComponent",
        "PredictorPipelineComponent",
        "LanguageModelPipelineComponent",
        "ImageTransformPipelineComponent",
        "Pipeline",
        "DetectResultGenerator",
        "SubImageLayoutService",
        "ImageCroppingService",
        "IntersectionMatcher",
        "NeighbourMatcher",
        "FamilyCompound",
        "MatchingService",
        "PageParsingService",
        "AnnotationNmsService",
        "MultiThreadPipelineComponent",
        "DoctectionPipe",
        "LanguageDetectionService",
        "skip_if_category_or_service_extracted",
        "ImageLayoutService",
        "LMTokenClassifierService",
        "LMSequenceClassifierService",
        "OrderGenerator",
        "TextLineGenerator",
        "TextLineService",
        "TextOrderService",
        "TableSegmentationRefinementService",
        "generate_html_string",
        "pipeline_component_registry",
        "TableSegmentationService",
        "PubtablesSegmentationService",
        "SegmentationResult",
        "TextExtractionService",
        "SimpleTransformService",
    ],
    "train": [
        "D2Trainer",
        "train_d2_faster_rcnn",
        "LayoutLMTrainer",
        "train_hf_layoutlm",
        "DetrDerivedTrainer",
        "train_hf_detr",
    ],
}

# Setting some environment variables so that standard functions can be invoked with available hardware
env_info = collect_env_info()
logger.debug(LoggingRecord(msg=env_info))


# Direct imports for type-checking
if TYPE_CHECKING:
    from .analyzer import *
    from .eval import *
    from .extern import *  # type: ignore
    from .pipe import *  # type: ignore
    from .train import *

else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _IMPORT_STRUCTURE,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
