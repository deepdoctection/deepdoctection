# -*- coding: utf-8 -*-
# File: __init__.py

"""
Init file for deepdoctection package. This file is used to import all submodules and to set some environment variables
"""

import sys
from typing import TYPE_CHECKING, Dict, List

from dd_core.utils.file_utils import _LazyModule
from dd_core.utils.env_info import collect_env_info
from dd_core.utils.logger import logger, LoggingRecord


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

# Build extra objects for the lazy module, starting with the version
_extra_objects: Dict[str, object] = {"__version__": __version__}

# Re-export all public attributes from dd_core under deepdoctection namespace
import dd_core
for _name in dir(dd_core):
    if _name.startswith("_"):
        continue
    # Optional: if dd_core defines __all__, you could respect it instead:
    # if hasattr(dd_core, "__all__") and _name not in dd_core.__all__:
    #     continue
    _extra_objects[_name] = getattr(dd_core, _name)

# Direct imports for type-checking
if TYPE_CHECKING:
    from .analyzer import *
    from .eval import *
    from .extern import *
    from .pipe import *
    from .train import *

    from dd_core import *
    from dd_datasets import *

else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _IMPORT_STRUCTURE,
        module_spec=globals().get("__spec__"),
        extra_objects=_extra_objects
    )
