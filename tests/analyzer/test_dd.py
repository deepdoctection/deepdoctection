# -*- coding: utf-8 -*-
# File: test_dd.py

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
Testing module analyzer.dd. This test case requires a GPU and should be considered as integration test
"""
from pytest import mark

from deepdoctection.analyzer import get_dd_analyzer
from deepdoctection.datapoint import Page

from ..test_utils import collect_datapoint_from_dataflow, get_integration_test_path


@mark.integration
@mark.requires_pt
@mark.requires_tf
def test_dd_analyzer_builds_and_process_image_layout_correctly() -> None:
    """
    Analyzer integration test with setting tables = False and ocr = False
    """

    # Arrange
    analyzer = get_dd_analyzer(tables=False, ocr=False)

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    assert len(page.items) == 10
    assert len(page.tables) == 1
    assert page.height == 2339
    assert page.width == 1654


@mark.integration
@mark.requires_pt
@mark.requires_tf
def test_dd_analyzer_builds_and_process_image_layout_and_tables_correctly() -> None:
    """
    Analyzer integration test with setting tables = True and ocr = False
    """

    # Arrange
    analyzer = get_dd_analyzer(tables=True, ocr=False)

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    assert len(page.items) == 10
    assert len(page.tables) == 1
    assert len(page.tables[0].cells) == 16
    assert len(page.tables[0].items) == 10
    assert (
        page.tables[0].html == "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>"
    )
    assert page.height == 2339
    assert page.width == 1654


@mark.integration
@mark.requires_pt
@mark.requires_tf
def test_dd_analyzer_builds_and_process_image_correctly() -> None:
    """
    Analyzer integration test with setting tables = True and ocr = True
    """

    # Arrange
    analyzer = get_dd_analyzer(tables=True, ocr=True)

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    assert len(page.items) == 10
    assert len(page.tables) == 1
    assert len(page.tables[0].cells) == 16
    assert len(page.tables[0].items) == 10
    assert (
        page.tables[0].html == "<table><tr><td>Jahresdurchschnitt der Mitarbeiterzahl</td><td>139</td></tr><tr>"
        "<td>Gesamtvergiitung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe Vergiitung</td>"
        "<td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR 2.164.096</td>"
        "</tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung"
        " fiir Senior Management °</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung"
        " fiir sonstige Risikotrager</td><td>EUR 324.229</td></tr><tr><td>Gesamtvergiitung"
        " fir Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>"
    )
    assert page.height == 2339
    assert page.width == 1654
    assert len(page.get_text()) == 4291
