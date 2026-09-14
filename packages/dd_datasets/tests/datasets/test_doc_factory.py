# -*- coding: utf-8 -*-
# File: test_doc_factory.py

# Copyright 2026 Dr. Janis Meyer. All rights reserved.
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
Testing module doc_factory

Uses the `eval_doc` fixture document (a two-page bank statement `doc.Document`, the same one used for
testing `deepdoctection.eval.record_align`) to exercise `DocumentDatasetFactory.make` end to end: argument
validation, the `mode="doc"`/`mode="image"` dataflow branches, and category filtering for
`DatasetKind.OBJECT_DETECTION`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import shared_test_utils as stu
from dd_core.utils.object_types import CellLabel, DatasetKind, LayoutLabel
from dd_datasets.base import CustomDataset
from dd_datasets.doc_factory import DocumentDatasetFactory

ALL_CATEGORIES = [
    LayoutLabel.TABLE,
    LayoutLabel.TEXT,
    LayoutLabel.TITLE,
    LayoutLabel.KEY_VALUE_AREA,
    LayoutLabel.FIGURE,
    LayoutLabel.PAGE_FOOTER,
    LayoutLabel.PAGE_NUMBER,
    LayoutLabel.ROW,
    LayoutLabel.COLUMN,
    CellLabel.COLUMN_HEADER,
    LayoutLabel.CELL,
    LayoutLabel.WORD,
    LayoutLabel.LINE,
]


@pytest.fixture
def eval_doc_workdir(monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Absolute path to the `eval_doc` fixture directory. Passing it as `location` makes
    `SETTINGS.DATASET_DIR / location` resolve to that absolute path regardless of `SETTINGS.DATASET_DIR`, so
    nothing needs to be copied or monkeypatched into place.
    """
    monkeypatch.setenv("ENABLE_DYNAMIC_OBJECT_TYPES", "True")
    return str(stu.asset_path("eval_doc").parent)


class TestDocumentDatasetFactoryMakeValidation:
    """
    Test argument validation of `DocumentDatasetFactory.make`
    """

    @staticmethod
    def test_make_warns_when_documents_subfolder_missing(
        eval_doc_workdir: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        A dataset directory without a `documents` sub folder is still built, but logs a warning
        """

        # Act
        caplog.clear()
        dataset = DocumentDatasetFactory.make(
            name="eval_doc",
            location=eval_doc_workdir,
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=ALL_CATEGORIES,
        )

        # Assert
        assert isinstance(dataset, CustomDataset)
        assert any("no 'documents' sub folder" in record.getMessage() for record in caplog.records)


class TestDocumentDataFlowBuilderBuild:
    """
    Test `DocumentDataFlowBuilder.build` (the builder `DocumentDatasetFactory.make` generates)
    """

    @staticmethod
    def test_build_doc_mode_with_load_image_raises(eval_doc_workdir: str) -> None:
        """
        `mode="doc"` combined with `load_image=True` raises `ValueError`
        """

        # Arrange
        dataset = DocumentDatasetFactory.make(
            name="eval_doc",
            location=eval_doc_workdir,
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=ALL_CATEGORIES,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.dataflow_builder.build(mode="doc", load_image=True)

    @staticmethod
    def test_build_mode_doc_returns_single_document_with_rebased_location(eval_doc_workdir: str) -> None:
        """
        `mode="doc"` yields exactly one `doc.Document` (the fixture holds a single document with two pages),
        and `rebase_document_location` points both the document and its pages at the local `documents`
        folder
        """

        # Arrange
        dataset = DocumentDatasetFactory.make(
            name="eval_doc",
            location=eval_doc_workdir,
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=ALL_CATEGORIES,
        )
        df = dataset.dataflow_builder.build(mode="doc")
        df.reset_state()

        # Act
        documents = list(df)

        # Assert
        assert len(documents) == 1
        document = documents[0]
        assert document.document_id == "136c24d6948a751375baebf4eb6b8a95"
        assert len(document._images) == 2  # pylint: disable=W0212
        expected_location = Path(eval_doc_workdir) / "documents" / document.file_name
        assert Path(document.location) == expected_location
        for image in document._images.values():  # pylint: disable=W0212
            assert Path(image.location) == expected_location

    @staticmethod
    def test_build_mode_image_flattens_pages_in_page_order(eval_doc_workdir: str) -> None:
        """
        `mode="image"` (the default) flattens the document into one `Image` datapoint per page, in page
        order
        """

        # Arrange
        dataset = DocumentDatasetFactory.make(
            name="eval_doc",
            location=eval_doc_workdir,
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=ALL_CATEGORIES,
        )
        df = dataset.dataflow_builder.build(mode="image")
        df.reset_state()

        # Act
        images = list(df)

        # Assert
        assert len(images) == 2
        assert [image.page_number for image in images] == [1, 2]
        assert len({image.image_id for image in images}) == 2

    @staticmethod
    def test_build_object_detection_filters_categories_and_reassigns_ids(eval_doc_workdir: str) -> None:
        """
        For `DatasetKind.OBJECT_DETECTION`, filtering categories to a subset removes every
        `ImageAnnotation` outside that subset and re-assigns `category_id`s according to the filtered
        order. Documents generated by a pipeline carry many more annotations (e.g. words, cells) than the
        three categories kept here.
        """

        # Arrange
        dataset = DocumentDatasetFactory.make(
            name="eval_doc",
            location=eval_doc_workdir,
            dataset_type=DatasetKind.OBJECT_DETECTION,
            init_categories=ALL_CATEGORIES,
        )
        dataset.dataflow.categories.filter_categories([LayoutLabel.TABLE, LayoutLabel.TEXT, LayoutLabel.TITLE])
        df = dataset.dataflow_builder.build(mode="image")
        df.reset_state()

        # Act
        images = list(df)

        # Assert
        assert len(images) == 2
        for image in images:
            annotations = image.get_annotation()
            assert annotations  # every page keeps at least its table/text/title annotations
            category_names_and_ids = {(ann.category_name, ann.category_id) for ann in annotations}
            assert {name for name, _ in category_names_and_ids} <= {
                LayoutLabel.TABLE,
                LayoutLabel.TEXT,
                LayoutLabel.TITLE,
            }
            assert {category_id for _, category_id in category_names_and_ids} <= {1, 2, 3}
