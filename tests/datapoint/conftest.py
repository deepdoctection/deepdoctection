# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixtures for datapoint package testing
"""
from dataclasses import dataclass

import numpy as np
from pytest import fixture

from deepdoctection.datapoint import BoundingBox, convert_np_array_to_b64
from deepdoctection.utils import get_uuid
from deepdoctection.utils.detection_types import ImageType


@dataclass
class Box:
    """
    Coordinates for bounding box testing
    """

    absolute_coords = True
    ulx = 1.0
    uly = 2.0
    lrx = 2.5
    lry = 4.0
    image_width = 10.0
    image_height = 20

    @property
    def w(self) -> float:
        """
        width
        """
        return self.lrx - self.ulx

    @property
    def h(self) -> float:
        """
        height
        """
        return self.lry - self.uly

    @property
    def cx(self) -> float:
        """
        center x
        """
        return self.ulx + 0.5 * self.w

    @property
    def cy(self) -> float:
        """
        center y
        """
        return self.uly + 0.5 * self.h

    @property
    def area(self) -> float:
        """
        area
        """
        return self.w * self.h

    @property
    def ulx_relative(self) -> float:
        """
        ulx relative coordinate
        """
        return self.ulx / self.image_width

    @property
    def uly_relative(self) -> float:
        """
        uly relative coordinate
        """
        return self.uly / self.image_height

    @property
    def lrx_relative(self) -> float:
        """
        lry relative coordinate
        """
        return self.lrx / self.image_width

    @property
    def lry_relative(self) -> float:
        """
        lry relative coordinate
        """
        return self.lry / self.image_height

    @property
    def w_relative(self) -> float:
        """
        width relative coordinate
        """
        return self.w / self.image_width

    @property
    def h_relative(self) -> float:
        """
        height relative coordinate
        """
        return self.h / self.image_height


@fixture(name="box")
def fixture_box() -> Box:
    """
    Box fixture
    """
    return Box()


@dataclass
class WhiteImage:
    """
    np_array, dummy location and file name and ids for testing
    """

    img = np.ones([4, 6, 3], dtype=np.float32)
    loc = "/testlocation/test"
    file_name = "test_image.png"
    external_id = "1234"
    uuid = "90c05f37-0000-0000-0000-b84f9d14ff44"

    def get_bounding_box(self) -> BoundingBox:
        """
        BoundingBox
        """
        return BoundingBox(ulx=0.0, uly=0.0, width=self.img.shape[1], height=self.img.shape[0], absolute_coords=True)

    @classmethod
    def get_image_id(cls, type_id: str) -> str:
        """
        image_id

        :param type: Either "d" (default), "n" (no external uuid), "u" (external uuid)
        """
        if type_id == "d":
            return get_uuid(cls.loc + cls.file_name)
        if type_id == "n":
            return get_uuid(cls.external_id)
        return cls.uuid

    def get_image_as_b64_string(self) -> str:
        """
        b64_string image representation
        """
        return convert_np_array_to_b64(self.img)

    def get_image_as_np_array(self) -> ImageType:
        """
        np.array(dtype=np.float32) image representation
        """
        return self.img


@fixture(name="image")
def fixture_image() -> WhiteImage:
    """
    TestWhiteImage
    """
    return WhiteImage()


@dataclass
class CatAnn:
    """
    Category and ids for testing
    """

    category_name = "FOO"
    category_id = "1"
    external_id = "567"
    uuid = "00000000-0000-0000-0000-000000000000"

    @classmethod
    def get_annotation_id(cls, type_id: str) -> str:
        """
        annotation_id, if set externally
        :param type: "n" (no external uuid), "u" (external uuid)
        :return: Annotation as uuid
        """
        if type_id == "n":
            return get_uuid(cls.external_id)
        return cls.uuid


@fixture(name="category_ann")
def fixture_category_ann() -> CatAnn:
    """
    TestCatAnn
    """
    return CatAnn()


class TestPdfPage:  # pylint: disable=R0903
    """
    Pdf as bytes, dummy location and file name for testing
    """

    pdf_bytes = (
        b"%PDF-1.3\n1 0 obj\n<<\n/Type /Pages\n/Count 1\n/Kids [ 3 0 R ]\n>>\nendobj\n2 0 obj\n<<\n/Producer "
        b"(PyPDF2)\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 1 0 R\n/Resources <<\n/Font "
        b"<<\n/F1 5 0 R\n>>\n/ProcSet 6 0 R\n>>\n/MediaBox [ 0 0 612 792 ]\n/Contents 7 0 R\n>>\nendobj\n4 "
        b"0 obj\n<<\n/Type /Catalog\n/Pages 1 0 R\n>>\nendobj\n5 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/Name"
        b" /F1\n/BaseFont /Helvetica\n/Encoding /WinAnsiEncoding\n>>\nendobj\n6 0 obj\n[ /PDF /Text"
        b" ]\nendobj\n7 0 obj\n<<\n/Length 1074\n>>\nstream\n2 J\r\nBT\r\n0 0 0 rg\r\n/F1 0027 Tf\r\n57.3750"
        b" 722.2800 Td\r\n( A Simple PDF File ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 688.6080 Td\r\n("
        b" This is a small demonstration .pdf file - ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 664.7040"
        b" Td\r\n( just for use in the Virtual Mechanics tutorials. More text. And more ) Tj\r\nET\r\nBT\r\n/F1"
        b" 0010 Tf\r\n69.2500 652.7520 Td\r\n( text. And more text. And more text. And more text. )"
        b" Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 628.8480 Td\r\n( And more text. And more text."
        b" And more text. And more text. And more ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 616.8960"
        b" Td\r\n( text. And more text. Boring, zzzzz. And more text. And more text. And )"
        b" Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 604.9440 Td\r\n( more text. And more text. And more"
        b" text. And more text. And more text. ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 592.9920 Td\r\n("
        b" And more text. And more text. ) Tj\r\nET\r\nBT\r\n/F1 0010 Tf\r\n69.2500 569.0880 Td\r\n( And"
        b" more text. And more text. And more text. And more text. And more ) Tj\r\nET\r\nBT\r\n/F1 0010"
        b" Tf\r\n69.2500 557.1360 Td\r\n( text. And more text. And more text. Even more. Continued on page"
        b" 2 ...) Tj\r\nET\r\n\nendstream\nendobj\nxref\n0 8\n0000000000 65535 f \n0000000009 00000 n"
        b" \n0000000068 00000 n \n0000000108 00000 n \n0000000251 00000 n \n0000000300 00000 n \n0000000407"
        b" 00000 n \n0000000437 00000 n \ntrailer\n<<\n/Size 8\n/Root 4 0 R\n/Info 2 0"
        b" R\n>>\nstartxref\n1563\n%%EOF\n"
    )

    loc = "/testlocation/test"
    file_name = "test_image_0.pdf"
    np_array_shape = (3301, 2550, 3)
    np_array_shape_default = (792, 612, 3)

    def get_image_as_pdf_bytes(self) -> bytes:
        """
        pdf in bytes
        """
        return self.pdf_bytes


@fixture(name="pdf_page")
def fixture_pdf_page() -> TestPdfPage:
    """
    TestPdfPage
    """
    return TestPdfPage()
