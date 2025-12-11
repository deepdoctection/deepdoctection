# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixtures for view module testing
"""

from pathlib import Path

from pytest import fixture

from dd_core.datapoint.view import Page


@fixture(name="page")
def fixture_page(page_json_path: Path) -> Page:
    """Provide a Page instance loaded from page_json fixture."""
    return Page.from_file(file_path=str(page_json_path))
