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
def test_dd_analyzer_builds_and_process_image_layout_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = False and USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(config_overwrite=["USE_TABLE_SEGMENTATION=False",
                                                 "USE_OCR=False",
                                                 "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
                                                 "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
                                                 "PT.ITEM.FILTER=None",
                                                 "PT.LAYOUT.PAD.TOP=60",
                                                 "PT.LAYOUT.PAD.RIGHT=60",
                                                 "PT.LAYOUT.PAD.BOTTOM=60",
                                                 "PT.LAYOUT.PAD.LEFT=60",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
                                                 "WORD_MATCHING.THRESHOLD=0.6",
                                                 "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
                                                 "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
                                                 "USE_LAYOUT_LINK=False",
                                                 "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
                                                 "LAYOUT_LINK.CHILD_CATEGORIES=[]",
                                                 "OCR.USE_DOCTR=False",
                                                 "OCR.USE_TESSERACT=True",
                                                 "USE_LAYOUT_NMS=False",
                                                 "USE_TABLE_REFINEMENT=True",
                                                 "USE_LINE_MATCHER=False",
                                                 "PT.ENFORCE_WEIGHTS.LAYOUT=True",
                                                 "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
                                                 "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
                                                 "LAYOUT_NMS_PAIRS.PRIORITY=None",
                                                 "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
                                                 ])

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 13, 16}
    assert len(page.tables) == 1
    assert page.height == 2339
    assert page.width == 1654


@mark.tf_integration
def test_dd_tf_analyzer_builds_and_process_image_layout_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = False and USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(config_overwrite=["USE_TABLE_SEGMENTATION=False",
                                                 "USE_OCR=False",
                                                 "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
                                                 "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
                                                 "PT.ITEM.FILTER=None",
                                                 "PT.LAYOUT.PAD.TOP=60",
                                                 "PT.LAYOUT.PAD.RIGHT=60",
                                                 "PT.LAYOUT.PAD.BOTTOM=60",
                                                 "PT.LAYOUT.PAD.LEFT=60",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
                                                 "WORD_MATCHING.THRESHOLD=0.6",
                                                 "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
                                                 "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
                                                 "USE_LAYOUT_LINK=False",
                                                 "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
                                                 "LAYOUT_LINK.CHILD_CATEGORIES=[]",
                                                 "OCR.USE_DOCTR=False",
                                                 "OCR.USE_TESSERACT=True",
                                                 "USE_LAYOUT_NMS=False",
                                                 "USE_TABLE_REFINEMENT=True",
                                                 "USE_LINE_MATCHER=False",
                                                 "PT.ENFORCE_WEIGHTS.LAYOUT=True",
                                                 "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
                                                 "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
                                                 "LAYOUT_NMS_PAIRS.PRIORITY=None",
                                                 "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
    ])

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 16}
    assert len(page.tables) == 1
    assert page.height == 2339
    assert page.width == 1654


@mark.integration
def test_dd_analyzer_builds_and_process_image_layout_and_tables_correctly() -> None:
    """
    Analyzer integration test with setting USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(config_overwrite=["USE_OCR=False",
                                                 "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
                                                 "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
                                                 "PT.ITEM.FILTER=None",
                                                 "PT.LAYOUT.PAD.TOP=60",
                                                 "PT.LAYOUT.PAD.RIGHT=60",
                                                 "PT.LAYOUT.PAD.BOTTOM=60",
                                                 "PT.LAYOUT.PAD.LEFT=60",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
                                                 "WORD_MATCHING.THRESHOLD=0.6",
                                                 "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
                                                 "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
                                                 "USE_LAYOUT_LINK=False",
                                                 "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
                                                 "LAYOUT_LINK.CHILD_CATEGORIES=[]",
                                                 "OCR.USE_DOCTR=False",
                                                 "OCR.USE_TESSERACT=True",
                                                 "USE_LAYOUT_NMS=False",
                                                 "USE_TABLE_REFINEMENT=True",
                                                 "USE_LINE_MATCHER=False",
                                                 "PT.ENFORCE_WEIGHTS.LAYOUT=True",
                                                 "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
                                                 "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
                                                 "LAYOUT_NMS_PAIRS.PRIORITY=None",
                                                 "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
    ])

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 13, 16}
    assert len(page.tables) in {1, 2}
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {14, 15, 16}  # type: ignore
    # first html for tp model, second for d2 model
    assert page.tables[0].html in {
        "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>",
        "<table><tr><td></td><td rowspan=2></td></tr><tr><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>"
        "</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td>"
        "<td></td></tr></table>",
        "<table><tr><td></td><td></td></tr><tr><td rowspan=2></td><td></td></tr><tr><td></td></tr><tr><td></td><td>"
        "</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>"
        "</td></tr></table>",
    }
    assert page.height == 2339
    assert page.width == 1654


@mark.tf_integration
def test_dd_tf_analyzer_builds_and_process_image_layout_and_tables_correctly() -> None:
    """
    Analyzer integration test with setting USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(config_overwrite=["USE_OCR=False",
                                                 "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
                                                 "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
                                                 "PT.ITEM.FILTER=None",
                                                 "PT.LAYOUT.PAD.TOP=60",
                                                 "PT.LAYOUT.PAD.RIGHT=60",
                                                 "PT.LAYOUT.PAD.BOTTOM=60",
                                                 "PT.LAYOUT.PAD.LEFT=60",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
                                                 "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
                                                 "WORD_MATCHING.THRESHOLD=0.6",
                                                 "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
                                                 "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
                                                 "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
                                                 "USE_LAYOUT_LINK=False",
                                                 "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
                                                 "LAYOUT_LINK.CHILD_CATEGORIES=[]",
                                                 "OCR.USE_DOCTR=False",
                                                 "OCR.USE_TESSERACT=True",
                                                 "USE_LAYOUT_NMS=False",
                                                 "USE_TABLE_REFINEMENT=True",
                                                 "USE_LINE_MATCHER=False",
                                                 "PT.ENFORCE_WEIGHTS.LAYOUT=True",
                                                 "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
                                                 "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
                                                 "LAYOUT_NMS_PAIRS.PRIORITY=None",
                                                 "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
                                                 ])

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 16}
    assert len(page.tables) == 1
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {15, 16}  # type: ignore
    # first html for tp model, second for d2 model
    print(page.tables[0].html)
    assert page.tables[0].html in {
        "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>",
        "<table><tr><td></td><td rowspan=2></td></tr><tr><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>"
        "</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td>"
        "<td></td></tr></table>",
    }
    assert page.height == 2339
    assert page.width == 1654


@mark.integration
def test_dd_analyzer_builds_and_process_image_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = True and USE_OCR = True
    """

    # Arrange
    analyzer = get_dd_analyzer()

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 13, 16}
    assert len(page.tables) == 1
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {13, 15, 16}  # type: ignore
    assert page.height == 2339
    assert page.width == 1654
    # first number for tp model, second for pt model
    print(page.text)
    assert len(page.text) in {4343,4345,4346,5042}
    text_ = page.text_
    assert text_["text"] == page._make_text(line_break=False)  # pylint: disable=W0212
    assert len(text_["words"]) in {555,631}
    assert len(text_["ann_ids"]) in {555,631}


@mark.tf_integration
def test_dd_tf_analyzer_builds_and_process_image_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = True and USE_OCR = True
    """

    # Arrange
    analyzer = get_dd_analyzer()

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 16}
    assert len(page.tables) == 1
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {15, 16}  # type: ignore
    # first html for tp model, second for d2 model
    assert page.tables[0].html in {
        "<table><tr><td>Jahresdurchschnitt der Mitarbeiterzahl</td><td>139</td></tr><tr>"
        "<td>Gesamtvergiitung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe Vergiitung</td>"
        "<td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR 2.164.096</td>"
        "</tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung"
        " fiir Senior Management °</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung"
        " fiir sonstige Risikotrager</td><td>EUR 324.229</td></tr><tr><td>Gesamtvergiitung"
        " fir Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>",
        "<table><tr><td></td><td rowspan=2>EUR 15.315.952</td></tr><tr><td>Gesamtvergiitung ?</td></tr><tr><td>Fixe"
        " Vergiitung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR 2.164.096</td></tr><tr>"
        "<td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung fiir Senior Management °</td><td>"
        "EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung fuir sonstige Risikotrager</td><td>EUR 324.229</td></tr><tr>"
        "<td>Gesamtvergiitung fir Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>",
        "<table><tr><td></td><td>139</td></tr><tr><td>Gesamtvergiitung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe "
        "Vergiitung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR "
        "2.164.096</td></tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung fiir "
        "Senior Management °</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung fiir sonstige Risikotrager"
        "</td><td>EUR 324.229</td></tr><tr><td></td><td></td></tr></table>",
    }
    assert page.height == 2339
    assert page.width == 1654
    # first number for tp model, second for pt model
    assert len(page.text) in {4290, 5042, 5043, 5044, 5045, 5153}
    text_ = page.text_
    assert text_["text"] == page._make_text(line_break=False)  # pylint: disable=W0212
    assert len(text_["words"]) in {529}
    assert len(text_["ann_ids"]) in {529}


@mark.integration_additional
def test_dd_analyzer_with_tatr() -> None:
    """
    Analyzer integration test with setting USE_OCR=False and table transformer for table detection and table recognition
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_OCR=False",
            "PT.LAYOUT.WEIGHTS=microsoft/table-transformer-detection/pytorch_model.bin",
            "PT.ITEM.WEIGHTS=microsoft/table-transformer-structure-recognition/pytorch_model.bin",
            "PT.ITEM.FILTER=['table']",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "OCR.USE_DOCTR=False",
            "OCR.USE_TESSERACT=True",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "PT.ENFORCE_WEIGHTS.LAYOUT=True",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert not page.layouts
    assert len(page.tables) == 1
    assert len(page.tables[0].cells) in {12,14}  # type: ignore


@mark.integration_additional
def test_dd_analyzer_with_doctr() -> None:
    """
    Analyzer integration test with setting USE_LAYOUT=False and USE_TABLE_SEGMENTATION=False and OCR.USE_DOCTR=True
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_LAYOUT=False",
            "USE_TABLE_SEGMENTATION=False",
            "OCR.USE_TESSERACT=False",
            "OCR.USE_DOCTR=True",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=True",
            "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
            "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
            "PT.ITEM.FILTER=None",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "PT.ENFORCE_WEIGHTS.LAYOUT=True",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    assert len(page.layouts) in {53, 55, 63}
    print(page.text_no_line_break)
    assert page.text_no_line_break in (
        """Festlegung der VV und angemessene Risikoadjustierung Die W-Pools der DWS Gruppe werden einer angemessenen Anpassung der Risiken unterzogen, die die Adjustierung ex ante als auch ex post umfasst. Die angewandte robuste Method d soll sicherstellen, dass bei der Festlegung der W sowohl der risikoadjustierten Leistung als auch der Kapital- und Liguiditatsaustattung der DWS Gruppe Rechnung getragen wird. Die mittlung des Gesamtbetrags der W orientiert sich primàr an 0 der Tragfâhigkeit fur die DWS Gruppe (das heiBt, was kann" die DWS Gruppe langfristig an) Wi im Einklang mit regulatorisch Anforderungen gewâhren) und ( der Leistung (das heilt, was sollte" die DWS Gruppe an W gewâhren, um fur eine angemessene leistungsbezogene Vergutung zu sorgen und gleichzeit den langfristigen Erfolg des Unternehmens zu sichern). Die DWS Gruppe hat fûr die Festlegung der W auf Ebene der individuellen Mitarbeiter die Grundsâtze for die Festlegung der variablen Vergutung" eingefuhrt. Diese enthalten Informatione Uber die Faktoren und MessgroBen, die bei Entscheidungen: zur W berlcksichtigt werden mussen. Dazu zâhlen beispielsweise Investmentperiomance. Kundenbindung, Erwàgungen Unternehmenskutur sowie Zielvereinbarungen und Leistungsbeurteilung im Rahmen des Ganzheitliche Leistung" -Ansatzes. Zudem werden Hinweise der Kontrollfunktionen und Diszipli narmalSnahmen sowie deren Einfluss auf die W einbezogen. Bei per Ermessensentscherdiung erfolgenden Sub-PoolZuteilungen verwendet das DWS DCC die internen (finanziellen und nichtfinanziellen) Balanced Scorecard-Kennzahlen zur Erstellung differenzierter und leistungsbezogener W-Pools. Vergutung fur das Jahr 2018 Nach der hervorragenden Entwicklung im Jahr 2017 hatte die globale easealpibande 2018 mit einigen Schwierigkeiten zu kâmpfen. Grûnde waren unguinstige Marktbedin- gungen, stârkere geopolitische Spannungen und die negative Stimmung unter den Anlegern, vor allem am europàischen Retail-Markt. Auch die DWS Gruppe blieb von dieser Entwicklung nicht verschont. Vor diesem Hintergrund hat das DCC die Tragfahigkeit der W fur das Jahr 2018 kontrolliert und festgestellt, dass die Kapital- und Liguditaitsausstattung der DWS Gruppe unter Berlcksichti gung des Ergebnisses vor und nach Steuern klar Uber den regulatorisch vorgeschriebenen Mindestanforderungen und dem internen Schwellenwert fur die Risikotoleranz liegt Als Teil der im Màrz 2019 fur das Performance-Jahr 2018 gewâhrten W wurde die Gruppenkomponente allen berechtigten Mitarbeitern auf Basis der Bewertung der vier festgelegten Leistungs kennzahlen gewàhrt. Der Vorstand der Deutsche Bank AG hat fûr 2018 unter Berlcksichtigung der betràchtlichen Leistungen der Mitarbeiter und in seinem Ermessen einen Zielerreichungsgrad von 70 % festgelegt. Identifizierung von Risikotràger GemaBs Gesetz vom 17 Dezember 2010 Uber die Organismen fûr gemeinsame Anlagen (in seiner jeweils gultigen Fassung) sowie den ESMA-Leitlinien unter Berlcksichtigung der OGAW- Richtlinie hat die Gesellschaft Mitarbeiter mit wesentlichem Einfluss auf das Risikoprofil der Gesellschaft ermittelt GRisikotrâger"). Das dentifiaerungsveriahren basiert auf der Bewertung des Einflusses folgender Kategorien von Mitarbeitern auf das Risikoprofil der Gesellschaft oder einen von ihr ver walteten Fonds: (a) GeschatstihungSenior Management, (b) Portfolio-1 Investmentmanager. (c) Kontrolfunktionen, (d) Mitarbeiter mit Leitungsfunktionen in Verwaltung, Marketing und Human Resources, (e) sonstige Mitarbeiter (Risikotràger) mit wesentlichem Einfluss, () sonstige Mitarbeiter in der gleichen Vergitungsstufe wie sonstige Risikotrâger. Mindestens 40 % der wi fur Risikotràger werden aufgeschoben vergeben. Des Weiteren werden fur wichtige Anlageexperten mindestens 50 % sowohl des direkt ausgezahlten als auch des aufgeschobenen Teils in Form von aktienbasierten oder fondsbasierten Instrumenten der DWS Gruppe gewàhrt. Alle aufgeschobenen Komponenten sind bestimmten Leistungs- und Verfalbedingungen unterworfen, um eine angemessene nachtràgliche Risikoadjustierung zu gewàhr leisten. Bei einem W-Betrag von weniger als EUR 50.000 erhalten Risikotrâger ihre gesamte w in bar und ohne Aufschub. Zusammenfassung der Informationen zur Vergutung fur die Gesellschaft fur 20181 Jahresdurchschnitt der Mitarbeiterzahl 139 EUR 15.315. .952 Gesamtvergutung? Fixe Vergutung EUR 13.151.856 Variable Vergutung EUR 2.164.096 davon: Carried Interest EURO Gesamtvergutung fur Senior Management EUR 1.468.434 Gesamtvergutung fûr sonstige Risikotrâger EUR 324.229 Gesamtvergutung fur Mitarbeiter mit Kontrollfunktionen EUR 554.046 Vergûtungsdaten fur Delegierte, an die die Gesellschaft Portfolio- oder Rskomangementaupaben Ubertragen hat, sind nicht in der Tabelle erfasst. Unter Berlcksichtigung diverser Vergdtungsbestandteile entsprechend den Definitionen in den ESMA-Leitinien, die Geldzahlungen oder -leistungen (wie Bargeld, Anteile, Optionsscheir Rentenbeitràge) oder Nicht-Idirekte) Geldleistungen (wie Gehaltsnebenleistungen oder Sondervergitungen fur Fahrzeuge, Mobiltelefone, usw.) umfassen. Senior Management" umfasst nur den Vorstand der Gesellschaft. Der Vorstand erfillt die Definition als Fuhrungskrâfte der Gesellschaft. Ober den Vorstand hinaus wurden keine weitere Fuhrungskrâfte identifiziert. 22""",  # pylint: disable=C0301
    )


@mark.tf_integration
def test_dd_tf_analyzer_with_doctr() -> None:
    """
    Analyzer integration test with setting USE_LAYOUT=False and USE_TABLE_SEGMENTATION=False and OCR.USE_DOCTR=True
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_LAYOUT=False",
            "USE_TABLE_SEGMENTATION=False",
            "OCR.USE_TESSERACT=False",
            "OCR.USE_DOCTR=True",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=True",
            "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
            "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
            "PT.ITEM.FILTER=None",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "PT.ENFORCE_WEIGHTS.LAYOUT=True",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    assert len(page.layouts) in {53, 55, 63}
    print(page.text_no_line_break)
    assert page.text_no_line_break in (
        """Festlegung der VV und angemessene Risikoadjustierung Die VV-Pools der DWS Gruppe werden einer angemessenen Anpassung der Risiken unterzogen, die die Adjustierung ex ante als auch ex post umfasst. Die angewandte robuste Methode soll sicherstellen, dass bei der Festlegung der W sowohl der risikoadjustierten Leistung als auch der Kapital- und Liquiditatsausstattung der DWS Gruppe Rechnung getragen wird. Die Er mittlung des Gesamtbetrags der W orientiert sich primar an ( der Tragfâhigkeit fur die DWS Gruppe (das heilt, was kann" die DWS Gruppe langfristig an W im Einklang mit regulatorischen Anforderungen gewahren) und (i) der Leistung (das heiBst, was wsollte" die DWS Gruppe an W gewahren, um fur eine angemessene leistungsbezogene Vergutung zus sorgen und gleichzeitig den langfristigen Erfolg des Unternehmens zu sichern). Die DWS Gruppe hat fur die Festlegung der W auf Ebene der individueller Mitarbeiter die Grundsâtze fur die Festlegung der variablen Vergutung" eingefuhrt. Diese ent tha en Informationen uber die Faktoren und MessgroBen, die bei Entscheidungen zur IVV berucksichtigt werden mussen. Dazu zâhlen beispielsweise Investmentperformance, Kundenbindung, Erwagungen zur Unternehmenskultur sowie Zielvereinbarungen und Leistungsbeurtellung im Rahmen des Ganzheitliche Leistung" -Ansatzes. Zudem werden Hinweise der Kontrollfunktionen und Diszipli- narmaBnahmen sowie deren Einfluss auf die W einbezogen. Bei per Ermessensentscheidung erfolgenden Sub-Pool-Zuteilungen verwendet das DWS DCC die internen (finanziellen und nichtfinanziellen) Balanced Scorecard-Kennzahlen zur Erstellung differenzierter und leistungsbezogener W-Pools. Vergutung fur das Jahr 2018 Nach der hervorragenden Entwicklung im Jahr 2017 hatte die globale Vermogenmervastungabande 2018 mit einigen Schwierigkeiten zu kâmpfen. Grunde waren ungunstige Marktbedin- gungen, starkere geopolitische Spannungen und die negative Stimmung unter den Anlegern, vor allem am europaischen Retail-Markt. Auch die DWS Gruppe blieb von dieser Entwicklung nicht verschont. Vor diesem Hintergrund hat das DCC die Tragfahigkeit der W fur das Jahr 2018 kontrolliert und festgestellt, dass die Kapital- und Liquiditatsausstatung der DWS Gruppe unter Berucksichti- gung des Ergebnisses vor und nach Steuern klar uber den regulatorisch vorgeschriebenen Mindestanforderungen und dem internen Schwellenwert fur die Risikotoleranz liegt. Als Teil der im Marz 2019 fur das Performance-Jahr 2018 gewâhrten W wurde die Gruppenkomponente allen berechtigten Mitarbeitern auf Basis der Bewertung der vier festgelegten Leistungs- kennzahlen gewahrt. Der Vorstand der Deutsche Bank AG hat fur 2018 unter Berucksichtigung der betrâchtlichen Leistungen der Mitarbeiter und in seinem Ermessen einen Zielerreichungsgrad von 70 % festgelegt. Identifizierung von Risikotragern Gemals Gesetz vom 17. Dezember 2010 uber die Organismen fur gemeinsame Anlagen (in seiner jeweils gultigen Fassung) sowie den ESMA-Leitlinien unter Berucksichtigung der OGAW- Richtlinie hat die Gesellschaft Mitarbeiter mit vesentlichem Einfluss auf das Risikoprofil der Gesellschaft ermittelt (Risikotrager"). Das identifizierungsverfahven basiert auf der Bewertung des Einflusses folgender Kategorien von Mitarbeitern auf das Risikoprofil der Gesellschaft oder einen von ihr ver walteten Fonds: (a) Geschaftsfuhungy/Senior Management (b) Portfolio-/ Investmentmanager; (c) Kontrollfunktionen, (d) Mitarbeite er mit Leitungsfunktionen in Verwaltung, Marketing und Human Resources, (e) sonstige Mitarbeiter (Risikotrager) mit wesentlichem Einfluss, (f) sonstige Mitarbeiter in der gleichen Vergutungsstufe wie sonstige Risikotrager. Mindestens 40 % der W fur Risikotrager werden aufgeschoben vergeben. Des Weiteren werden fur wichtige Anlageexperten mindestens 50 % sowohl des direkt ausgezahlten als auch des aufgeschobenen Teils in Form von aktienbasie et erten oder fondsbasierten Instrur imenten der DWS Gruppe gewahrt. Alle aufgeschobenen Komponenten sind bestimmten Leistungs- und Verfallbedingungen unterworfen, um eine angemessene nachtragliche Risikoadjustierung zu gewahr leisten. Bei einem W-Betrag von weniger als EUR 50.000 erhalten Risikotrager ihre gesamte W in bar und ohne Aufschub. Zusammenfassung der Informationen zur Vergutung fur die Gesellschaft fur 2018 1 Jahresdurchschnitt der Mitarbeiterzahl 139 EUR 15.315.952 Gesamtvergutung Fixe Vergutung EUR 13.151.856 Variable Vergutung EUR 2.164. .096 davon: Carried Interest EUR 0 Gesamtvergutung fur Senior Management 3 EUR 1.468.434 Gesamtvergutung fur sonstige Risikotrager EUR 324.229 Gesamtvergutung fur Mitarbeiter mit Kontrolfunktionen EUR 554.046 Vergutungsdaten fur Delegierte, an die die Gesellschaft Portfolio- oder Rasliomanagementaufoaben ubertragen hat, sind nicht in der Tabelle erfasst. Unter Berucksichtigung diverser Vergutungsbestandteile entsprechend den Definitionen in den ESMA-Leitlinien, die Geldzahlungen oder leistungen (wie Bargeld, Anteile, Optionsscheine, Rentenbeitrage) oder Nicht-(direkte) Geldleistungen (wie Gehatsnebenleistungen oder Sondervergdtungen fur Fahrzeuge, Mobiltelefone, usw.) umfassen. Senior Management" umfasst nur den Vorstand der Gesellschaft. Der Vorstand erfullt die Definition als Fuhrungskrafte der Gesellschaft. Uber den Vorstand hinaus wurden keine weiteren Fuhrungskrafte identifiziert. 22"""  # pylint: disable=C0301  # pylint: disable=C0301
    )
