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


import pytest

from dd_core.utils.object_types import ObjectTypes, object_types_registry, update_black_list
from dd_core.utils.viz import viz_handler



class ObjectTestType(ObjectTypes):
    """Object type members for testing purposes"""

    REPORT_DATE = "report_date"
    UMBRELLA = "umbrella"
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


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("ObjectTestType")(ObjectTestType)
    for item in ObjectTestType:
        update_black_list(item.value)
    viz_handler.refresh()