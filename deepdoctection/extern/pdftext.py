# -*- coding: utf-8 -*-
# File: pdftext.py

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
PDFPlumber text extraction engine
"""

from typing import Optional

from lazy_imports import try_import

from ..utils.context import save_tmp_file
from ..utils.file_utils import get_pdfplumber_requirement, get_pypdfium2_requirement
from ..utils.settings import LayoutType, ObjectTypes
from ..utils.types import Requirement
from .base import DetectionResult, ModelCategories, PdfMiner

with try_import() as pdfplumber_import_guard:
    from pdfplumber.pdf import PDF, Page

with try_import() as pypdfmium_import_guard:
    import pypdfium2.raw as pypdfium_c
    from pypdfium2 import PdfDocument


def _to_detect_result(word: dict[str, str], class_name: ObjectTypes) -> DetectionResult:
    return DetectionResult(
        box=[float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"])],
        class_id=1,
        text=word["text"],
        class_name=class_name,
    )


class PdfPlumberTextDetector(PdfMiner):
    """
    Text miner based on the `pdfminer.six` engine. To convert `pdfminers` result, especially group character to get word
    level results we use `pdfplumber`.

    Example:
        ```python
        pdf_plumber = PdfPlumberTextDetector()
        df = SerializerPdfDoc.load("path/to/document.pdf")
        df.reset_state()

        for dp in df:
            detection_results = pdf_plumber.predict(dp["pdf_bytes"])
        ```

    To use it in a more integrated way:

    Example:
        ```python
        pdf_plumber = PdfPlumberTextDetector()
        text_extract = TextExtractionService(pdf_plumber)

        pipe = DoctectionPipe([text_extract])

        df = pipe.analyze(path="path/to/document.pdf")
        df.reset_state()

        for dp in df:
            ...
        ```
    """

    def __init__(self, x_tolerance: int = 3, y_tolerance: int = 3) -> None:
        self.name = "Pdfplumber"
        self.model_id = self.get_model_id()
        self.categories = ModelCategories(init_categories={1: LayoutType.WORD})
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self._page: Optional[Page] = None

    def predict(self, pdf_bytes: bytes) -> list[DetectionResult]:
        """
        Call `pdfminer.six` and returns detected text as `DetectionResult`

        Args:
            pdf_bytes: bytes of a single pdf page

        Returns:
            A list of `DetectionResult`
        """

        with save_tmp_file(pdf_bytes, "pdf_") as (tmp_name, _):
            with open(tmp_name, "rb") as fin:
                self._page = PDF(fin).pages[0]
                self._pdf_bytes = pdf_bytes
                words = self._page.extract_words(x_tolerance=self.x_tolerance, y_tolerance=self.y_tolerance)
        detect_results = [_to_detect_result(word, self.get_category_names()[0]) for word in words]
        return detect_results

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pdfplumber_requirement()]

    def get_width_height(self, pdf_bytes: bytes) -> tuple[float, float]:
        """
        Get the width and height of the full page

        Args:
            pdf_bytes: `pdf_bytes` generating the pdf

        Returns:
            `(width,height)`
        """

        if self._pdf_bytes == pdf_bytes and self._page is not None:
            return self._page.bbox[2], self._page.bbox[3]
        # if the pdf bytes is not equal to the cached pdf, will recalculate values
        with save_tmp_file(pdf_bytes, "pdf_") as (tmp_name, _):
            with open(tmp_name, "rb") as fin:
                _pdf = PDF(fin)
                self._page = _pdf.pages[0]
                self._pdf_bytes = pdf_bytes
        return self._page.bbox[2], self._page.bbox[3]

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)


class Pdfmium2TextDetector(PdfMiner):
    """
    Text miner based on the pypdfium2 engine. It will return text on text line level and not on word level

    Example:
        ```python
        pdfmium2 = Pdfmium2TextDetector()
        df = SerializerPdfDoc.load("path/to/document.pdf")
        df.reset_state()

        for dp in df:
            detection_results = pdfmium2.predict(dp["pdf_bytes"])
        ```

    To use it in a more integrated way:

    Example:
        ```python
        pdfmium2 = Pdfmium2TextDetector()
        text_extract = TextExtractionService(pdfmium2)

        pipe = DoctectionPipe([text_extract])

        df = pipe.analyze(path="path/to/document.pdf")
        df.reset_state()
        for dp in df:
            ...
        ```

    """

    def __init__(self) -> None:
        self.name = "Pdfmium"
        self.model_id = self.get_model_id()
        self.categories = ModelCategories(init_categories={1: LayoutType.LINE})
        self._page: Optional[Page] = None

    def predict(self, pdf_bytes: bytes) -> list[DetectionResult]:
        """
        Call pypdfium2 and returns detected text as detection results

        Args:
            pdf_bytes: bytes of a single pdf page

        Returns:
            A list of `DetectionResult`
        """

        pdf = PdfDocument(pdf_bytes)
        page = pdf.get_page(0)
        text = page.get_textpage()
        words = []
        height = page.get_height()
        for obj in page.get_objects((pypdfium_c.FPDF_PAGEOBJ_TEXT,)):
            box = obj.get_pos()
            if all(x > 0 for x in box):
                words.append(
                    {
                        "text": text.get_text_bounded(*box),
                        "x0": box[0],
                        "x1": box[2],
                        "top": height - box[3],
                        "bottom": height - box[1],
                    }
                )
        detect_results = [_to_detect_result(word, self.get_category_names()[0]) for word in words]
        return detect_results

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pypdfium2_requirement()]

    def get_width_height(self, pdf_bytes: bytes) -> tuple[float, float]:
        """
        Get the width and height of the full page

        Args:
            pdf_bytes: `pdf_bytes` generating the pdf

        Returns:
            `(width,height)`
        """

        if self._pdf_bytes == pdf_bytes and self._page is not None:
            return self._page.bbox[2], self._page.bbox[3]  # pylint: disable=E1101
        # if the pdf bytes is not equal to the cached pdf, will recalculate values
        pdf = PdfDocument(pdf_bytes)
        self._page = pdf.get_page(0)
        self._pdf_bytes = pdf_bytes
        if self._page is not None:
            return self._page.get_width(), self._page.get_height()  # type: ignore
        raise ValueError("Page not found")

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)
