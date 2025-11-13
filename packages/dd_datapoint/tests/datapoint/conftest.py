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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pytest import fixture

#from tests._shared.factory import build_test_pdf_page, TestPdfPage



"""
@dataclass
class WhiteImage:

    img = np.ones([4, 6, 3], dtype=np.uint8)
    loc = "/testlocation/test"
    file_name = "test_image.png"
    external_id = "1234"
    uuid = "90c05f37-0000-0000-0000-b84f9d14ff44"

    def get_bounding_box(self) -> BoundingBox:
        return BoundingBox(ulx=0.0, uly=0.0, width=self.img.shape[1], height=self.img.shape[0], absolute_coords=True)

    @classmethod
    def get_image_id(cls, type_id: str) -> str:
        if type_id == "d":
            return get_uuid(cls.loc + cls.file_name)
        if type_id == "n":
            return get_uuid(cls.external_id)
        return cls.uuid

    def get_image_as_b64_string(self) -> str:
        return convert_np_array_to_b64(self.img)

    def get_image_as_np_array(self) -> PixelValues:
        return self.img


@fixture(name="image")
def fixture_image() -> WhiteImage:
    return WhiteImage()


@dataclass
class CatAnn:

    category_name = "FOO"
    category_id = 1
    external_id = "567"
    uuid = "00000000-0000-0000-0000-000000000000"

    @classmethod
    def get_annotation_id(cls, type_id: str) -> str:
        if type_id == "n":
            return get_uuid(cls.external_id)
        return cls.uuid


@fixture(name="category_ann")
def fixture_category_ann() -> CatAnn:
    return CatAnn()
"""

#@fixture(name="pdf_page")
#def fixture_pdf_page() -> TestPdfPage:
#    """Provide a deterministic 1-page PDF for rendering tests."""
#    return build_test_pdf_page()

