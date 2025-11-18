# -*- coding: utf-8 -*-
# File: xxx.py

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


from pathlib import Path
from pytest import fixture

from dd_core.datapoint.image import Image


@fixture(name="image")
def fixture_page(page_json_path: Path) -> Image:
    """Provide a Page instance loaded from page_json fixture."""
    return Image.from_file(file_path=str(page_json_path))


@fixture(name="table_image")
def fixture_table_image(image: Image) -> Image:
    """An image from a table image annotation crop"""
    return image.get_annotation(category_names="table")[0].image
