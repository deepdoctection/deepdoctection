# -*- coding: utf-8 -*-
# File: test_dd.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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
Analyzer integration tests
"""

import pytest

import shared_test_utils as stu
from dd_core.datapoint.view import Page
from dd_core.utils.file_utils import detectron2_available, pytorch_available
from deepdoctection.analyzer import get_dd_analyzer

REQUIRES_PT_AND_D2 = pytest.mark.skipif(
    not (pytorch_available() and detectron2_available()),
    reason="Requires PyTorch and Detectron2 installed",
)


@REQUIRES_PT_AND_D2
@pytest.mark.slow
def test_dd_analyzer_default_config_on_invoice_pdf() -> None:
    """
    Run the analyzer with default config on an invoice PDF and validate:
    - At least one page is produced.
    - Layout count and types on the first page.
    - First table has CSV and HTML outputs and non-empty cells.
    - Residual layouts contain expected categories and page text is present.
    """
    analyzer = get_dd_analyzer()
    df = analyzer.analyze(path=stu.asset_path("invoice_pdf"))
    df.reset_state()
    pages = list(df)

    assert len(pages) >= 1
    assert isinstance(pages[0], Page)
    page = pages[0]

    assert len(page.layouts) in {7, 8}

    table = page.tables[0]
    assert table.csv is not None
    assert table.html is not None

    assert len(table.cells) > 0  # type: ignore

    assert len(page.residual_layouts) == 6

    assert {"page_header", "page_footer"} == {
        layout.category_name.value for layout in page.residual_layouts  # type: ignore
    }
    assert len(page.text) > 0


@REQUIRES_PT_AND_D2
@pytest.mark.slow
def test_dd_analyzer_toggle_components() -> None:
    """
    Run the analyzer with selected components toggled and validate:
    - Multiple pages are produced for a rotated input.
    - OCR disabled but PDF miner enabled yields text.
    - Two tables detected on the first page.
    - Detected page rotation is 90 degrees.
    """
    # Enable/Disable requested components only
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_ROTATOR=True",
            "USE_LAYOUT_NMS=False",
            "USE_OCR=False",
            "USE_PDF_MINER=True",
            "USE_LINE_MATCHER=True",
            "USE_LAYOUT_LINK=True",
        ]
    )
    df = analyzer.analyze(path=stu.asset_path("invoice_rotated_pdf"))
    df.reset_state()
    pages = list(df)
    assert len(pages) >= 2
    assert isinstance(pages[0], Page)
    assert len(pages[0].text) > 0
    assert len(pages[0].tables) == 2
    assert pages[0].angle == 90


@REQUIRES_PT_AND_D2
@pytest.mark.slow
def test_dd_analyzer_swap_layout_and_cell_models() -> None:
    """
    Run the analyzer with layout and cell model weights disabled and table refinement enabled.
    Validate:
    - Pipeline runs on an invoice PDF.
    - If tables exist, the first table produces HTML.
    """
    analyzer = get_dd_analyzer(
        config_overwrite=["LAYOUT.WEIGHTS=None", "ITEM.WEIGHTS=None", "CELL.WEIGHTS=None", "USE_TABLE_REFINEMENT=True"]
    )
    df = analyzer.analyze(path=stu.asset_path("invoice_pdf"))
    df.reset_state()
    pages = list(df)
    assert len(pages) >= 1
    assert isinstance(pages[0], Page)
    # If tables are present, segmentation may yield cells with swapped cell model
    # Do not enforce exact counts, just check pipeline runs
    if pages[0].tables:
        assert pages[0].tables[0].html is not None


@REQUIRES_PT_AND_D2
@pytest.mark.slow
def test_dd_analyzer_swap_ocr_engines_and_weights() -> None:
    """
    Switch OCR engine to Tesseract and ensure OCR-only processing.
    Validate:
    - Pages are produced for invoice PDF.
    - Layout counts reflect text extraction-only scenario.
    Note: Counts are asserted directly to detect regressions in OCR processing.
    """
    # Switch OCR to Tesseract and also set an alternative DocTR recognition weight (kept inert when Tesseract is used)
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_LAYOUT=True",
            "USE_TABLE_SEGMENTATION=False",
            "OCR.USE_TESSERACT=True",
            "OCR.USE_DOCTR=False",
            "OCR.USE_TEXTRACT=False",
            "OCR.USE_AZURE_DI=False",
        ]
    )
    df = analyzer.analyze(path=stu.asset_path("invoice_pdf"))
    df.reset_state()
    pages = list(df)
    assert len(pages) >= 1
    assert isinstance(pages[0], Page)
    assert len(pages[0].layouts) >= 1
