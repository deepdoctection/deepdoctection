# -*- coding: utf-8 -*-
# File: test_tedsmetric.py

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
Testing module eval.tedsmetric
"""

from pytest import mark

from deepdoctection.utils.file_utils import apted_available

if apted_available():
    from deepdoctection.eval.tedsmetric import teds_metric


@mark.basic
def test_teds_metric_returns_correct_distance() -> None:
    """
    teds returns score of 1.0 when comparing identical html strings
    """

    html_str = "<table><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>"
    results, number_results = teds_metric([html_str], [html_str], False)

    assert number_results == 1
    assert results == 1.0
