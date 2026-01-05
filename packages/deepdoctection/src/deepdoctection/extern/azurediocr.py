# -*- coding: utf-8 -*-
# File: azurediocr.py

# Copyright 2026 Dr. Janis Meyer and Idan Hemed. All rights reserved.
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
Azure Document Intelligence OCR engine for text extraction
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

from lazy_imports import try_import

from dd_core.datapoint.convert import convert_np_array_to_b64_b
from dd_core.utils.file_utils import get_azure_di_requirement
from dd_core.utils.logger import LoggingRecord, logger
from dd_core.utils.object_types import LayoutType, ObjectTypes
from dd_core.utils.types import PixelValues, Requirement

from .base import DetectionResult, ModelCategories, ObjectDetector

with try_import() as import_guard:
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type:ignore
    from azure.core.credentials import AzureKeyCredential  # type:ignore


def _azure_di_to_detectresult(result: Any, width: int, height: int, text_lines: bool) -> list[DetectionResult]:
    """
    Convert Azure Document Intelligence API response to DetectionResult objects.

    Args:
        result: Azure Document Intelligence analyze result
        width: Image width in pixels
        height: Image height in pixels
        text_lines: If True, return DetectionResults for lines as well

    Returns:
        A list of DetectionResult objects
    """
    all_results: list[DetectionResult] = []

    if not hasattr(result, 'pages') or not result.pages:
        return all_results

    for page in result.pages:
        # Process words
        if hasattr(page, 'words') and page.words:
            for word in page.words:
                # Azure polygon: [x1,y1,x2,y2,x3,y3,x4,y4] in pixels (for images)
                if hasattr(word, 'polygon') and len(word.polygon) >= 8:
                    # robust bounding box calculation for rotated text
                    x_coords = word.polygon[0::2]
                    y_coords = word.polygon[1::2]
                    word_result = DetectionResult(
                        box=[
                            min(x_coords),  # x_min
                            min(y_coords),  # y_min
                            max(x_coords),  # x_max
                            max(y_coords),  # y_max
                        ],
                        score=word.confidence if hasattr(word, 'confidence') and word.confidence else 1.0,
                        text=word.content if hasattr(word, 'content') else "",
                        class_id=1,
                        class_name=LayoutType.WORD,
                        absolute_coords=True,
                    )
                    all_results.append(word_result)

        # Process lines if requested
        if text_lines and hasattr(page, 'lines') and page.lines:
            for line in page.lines:
                if hasattr(line, 'polygon') and len(line.polygon) >= 8:
                    x_coords = line.polygon[0::2]
                    y_coords = line.polygon[1::2]
                    line_result = DetectionResult(
                        box=[
                            min(x_coords),  # x_min
                            min(y_coords),  # y_min
                            max(x_coords),  # x_max
                            max(y_coords),  # y_max
                        ],
                        score=line.confidence if hasattr(line, 'confidence') and line.confidence else 1.0,
                        text=line.content if hasattr(line, 'content') else "",
                        class_id=2,
                        class_name=LayoutType.LINE,
                        absolute_coords=True,
                    )
                    all_results.append(line_result)

    return all_results


def predict_text(np_img: PixelValues, client: DocumentIntelligenceClient, text_lines: bool) -> list[DetectionResult]:
    """
    Calls Azure Document Intelligence client (prebuilt-read model) and returns plain OCR results.
    Azure account required.

    Args:
        np_img: Image in `np.array`.
        client: Azure DocumentIntelligenceClient
        text_lines: If `True`, it will return `DetectionResult`s of Text lines as well.

    Returns:
        A list of `DetectionResult`
    """

    width, height = np_img.shape[1], np_img.shape[0]
    b_img = convert_np_array_to_b64_b(np_img)

    try:
        # Azure uses async poller pattern
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=b_img,
            content_type="application/octet-stream"
        )
        result = poller.result()
    except:  # pylint: disable=W0702
        _, exc_val, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[0]
        error_type = type(exc_val).__name__
        error_msg = str(exc_val)
        log_dict = {
            "file_name": "NN",
            "error_type": error_type,
            "error_msg": error_msg,
            "orig_module": frame_summary.filename,
            "line": frame_summary.lineno,
        }
        logger.warning(LoggingRecord(f"Azure Document Intelligence Exception ({error_type}): {error_msg}", log_dict))  # type: ignore
        result = None

    all_results = _azure_di_to_detectresult(result, width, height, text_lines) if result else []
    return all_results


class AzureDocIntelOcrDetector(ObjectDetector):
    """
    Text object detector based on Azure Document Intelligence OCR engine. Note that an Azure account
    as well as the azure-ai-documentintelligence package are required.

    Note:
        The service is not free of charge. Additional information can be found at:
        <https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence> .

    The detector only calls the base prebuilt-read model and does not return additional
    Document Intelligence features like forms or tables.

    Example:

        Credentials can be passed directly:

        ```python
        azure_predictor = AzureDocIntelOcrDetector(
            endpoint="https://your-resource.cognitiveservices.azure.com/",
            api_key="your-api-key"
        )
        detection_result = azure_predictor.predict(bgr_image_as_np_array)
        ```

        Or via environment variables `AZURE_DI_ENDPOINT` and `AZURE_DI_KEY`:

        ```python
        import os
        os.environ["AZURE_DI_ENDPOINT"] = "https://your-resource.cognitiveservices.azure.com/"
        os.environ["AZURE_DI_KEY"] = "your-api-key"

        azure_predictor = AzureDocIntelOcrDetector()
        text_extract = TextExtractionService(azure_predictor)

        pipe = DoctectionPipe([text_extract])
        df = pipe.analyze(path="path/to/document.pdf")

        for dp in df:
            ...
        ```

    """

    def __init__(self, text_lines: bool = True, **credentials_kwargs: str) -> None:
        """
        Args:
            text_lines: If `True`, it will return `DetectionResult`s of Text lines as well.
            credentials_kwargs: `endpoint` and `api_key` for Azure Document Intelligence
        """
        self.name = "azure_document_intelligence"
        self.model_id = self.get_model_id()

        self.text_lines = text_lines

        credentials_kwargs = self._maybe_resolve_secret(**credentials_kwargs)

        endpoint = credentials_kwargs.get("endpoint")
        api_key = credentials_kwargs.get("api_key")
        
        if not endpoint or not api_key:
            raise ValueError(
                "Azure Document Intelligence requires 'endpoint' and 'api_key'. "
                "Pass them as keyword arguments: AzureDocIntelOcrDetector(endpoint='...', api_key='...')"
            )
        
        # Store for clone
        self._endpoint = endpoint
        self._api_key = api_key

        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )

        if self.text_lines:
            self.categories = ModelCategories(init_categories={1: LayoutType.WORD, 2: LayoutType.LINE})
        else:
            self.categories = ModelCategories(init_categories={1: LayoutType.WORD})

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Transfer of a `np.array` and call Azure Document Intelligence `client`. Return of the `DetectionResult`s.

        Args:
            np_img: image as `np.array`

        Returns:
            A list of `DetectionResult`s
        """

        return predict_text(np_img, self.client, self.text_lines)

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_azure_di_requirement()]

    def clone(self) -> AzureDocIntelOcrDetector:
        return self.__class__(text_lines=self.text_lines, endpoint=self._endpoint, api_key=self._api_key)

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)

    def _maybe_resolve_secret(self, **credentials_kwargs: str) -> dict[str, str]:
        for key, value in credentials_kwargs.items():
            if value is not None:
                if hasattr(value, "get_secret_value"):
                    credentials_kwargs[key] = value.get_secret_value()
        return credentials_kwargs
