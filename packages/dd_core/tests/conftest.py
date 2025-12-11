# -*- coding: utf-8 -*-
"""
Test utilities for dd_datapoint tests
Minimal version to support standalone package tests
"""

from pathlib import Path
from dataclasses import dataclass, field
import pytest

import numpy as np


import shared_test_utils as stu

from dd_core.utils.object_types import ObjectTypes, object_types_registry, update_black_list
from dd_core.utils.viz import viz_handler

from .data import XFUND_SAMPLE


class ObjectTestType(ObjectTypes):
    """Object type members for testing purposes"""

    TEST_CAT_1 = "test_cat_1"
    TEST_CAT_2 = "test_cat_2"
    TEST_CAT_3 = "test_cat_3"
    TEST_CAT_4 = "test_cat_4"
    SUB_CAT_1 = "sub_cat_1"
    SUB_CAT_2 = "sub_cat_2"
    SUB_CAT_3 = "sub_cat_3"
    RELATIONSHIP_1 = "relationship_1"
    RELATIONSHIP_2 = "relationship_2"
    NON_EXISTENT = "non_existent"


@dataclass(frozen=True)
class TestPdfPage:
    """
    Container for a deterministic single-page PDF test asset.

    Attributes:
        pdf_bytes: Raw PDF file content as bytes
        loc: Logical location identifier
        file_name: Suggested filename for this PDF
        np_array_shape_default: Expected numpy array shape at default DPI (72)
        np_array_shape_300: Expected numpy array shape at 300 DPI
    """

    pdf_bytes: bytes
    loc: str =  "/testlocation/test"
    file_name: str  = "test_image_0.pdf"
    np_array_shape: tuple[int, int, int] = (3300, 2550, 3)
    np_array_shape_default: tuple[int, int, int] = (792, 612, 3)


@dataclass(frozen=True)
class WhiteImage:
    """Test fixture for a white image with deterministic properties"""

    image = np.ones([400, 600, 3], dtype=np.uint8)
    location = "/testlocation/test"
    file_name = "test_image.png"
    external_id = "1234"
    uuid = "90c05f37-0000-0000-0000-b84f9d14ff44"


@dataclass(frozen=True)
class XFundSample:
    """Deterministic XFund sample datapoint for testing."""
    data: dict = field(default_factory=lambda: XFUND_SAMPLE["documents"][0])
    np_array_shape: tuple[int, int, int] = (3508, 2480, 3)


@pytest.fixture(name="page_json_path")
def fixture_page_json_path() -> Path:
    """Provide path to a sample page json file."""
    return stu.asset_path("page_json")


@pytest.fixture(name="pdf_file_path_two_pages")
def fixture_pdf_file_path_two_pages() -> Path:
    return stu.asset_path("pdf_file_two_pages")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("ObjectTestType")(ObjectTestType)
    for item in ObjectTestType:
        update_black_list(item.value)
    viz_handler.refresh()
