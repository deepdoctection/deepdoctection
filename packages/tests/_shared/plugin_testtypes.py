# -*- coding: utf-8 -*-
# File: plugin_testtypes.py

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

import pytest

from dd_datapoint.utils.object_types import  object_types_registry, update_black_list
from dd_datapoint.utils.viz import viz_handler

from .test_types import TestType

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("TestType")(TestType)
    for item in TestType:
        update_black_list(item.value)
    viz_handler.refresh()