# -*- coding: utf-8 -*-
# File: doc.py

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

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .dataflow.base import DataFlow
from .dataflow.common import MapData
from .dataflow.custom_serialize import SerializerFiles, SerializerPdfDoc
from .dataflow.serialize import DataFromList
from .datapoint.annotation import CategoryAnnotation
from .datapoint.image import Image
from .datapoint.view import ImageAnnotationBaseView, Page
from .mapper.maputils import curry
from .utils import get_uuid_from_str
from .utils.file_utils import mkdir_p
from .utils.identifier import is_uuid_like
from .utils.object_types import ObjectTypes, SummaryKey, get_type
from .utils.types import PathLikeOrStr
from .utils.viz import viz_handler

from .utils.object_types import DocumentFileLabel

"""
Document model and utilities for multi-page document handling.

This module provides lightweight references and a full-featured Document class
used for representing and processing multi-page documents (PDFs and image
collections). Core responsibilities include lazy page and pixel loading,
annotation resolution, structured-output traversal, JSON serialization/
deserialization, and visualization helpers.
"""


@dataclass(frozen=True)
class AnnPair:
    """Lightweight pair referencing an image annotation."""
    image_id: str
    annotation_id: str

@dataclass(frozen=True)
class PageReference:
    """
    Lightweight reference to a page.
    """

    source_path: str
    page_number: int | None = None
    image_id: str | None = None

@dataclass(frozen=True)
class PipelineSession:
    """
    Tracks pipeline processing sessions for documents.
    """

    pipeline_name: str
    session_id: str
    pipeline_info: dict[str, str]


def _maybe_to_annpair(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "image_id" in obj and "annotation_id" in obj:
            return AnnPair(image_id=obj["image_id"], annotation_id=obj["annotation_id"])
    return obj


def _walk(node: Any, path: list[str], ann_to_paths: defaultdict[str, set[str]]) -> None:
    if node is None:
        return

    if isinstance(node, Mapping):
        for k, v in node.items():
            _walk(v, path + [str(k)], ann_to_paths)
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            item = _maybe_to_annpair(item)
            if isinstance(item, AnnPair):
                ann_to_paths[item.annotation_id].add(".".join(path))
            else:
                _walk(item, path, ann_to_paths)
        return

    node = _maybe_to_annpair(node)
    if isinstance(node, AnnPair):
        ann_to_paths[node.annotation_id].add(".".join(path))
    return

def flatten_entity_dict_to_ann_index(
    data: Mapping[str, Any],
) -> dict[str, set[str]]:
    """
    Flattens a nested structure (mapping / list / tuple) of entities into
    a mapping from annotation id to the set of key-paths where the annotation
    occurs.

    Args:
        data (Mapping[str, Any]): Arbitrary nested mapping/list structure that may
            contain annotation reference objects or dictionaries representing
            annotation pairs.

    Returns:
        dict[str, set[str]]: Mapping where keys are annotation UUID strings and
        values are sets of dot-separated key paths pointing to occurrences.
    """

    ann_to_paths: defaultdict[str, set[str]] = defaultdict(set)



    _walk(data, [], ann_to_paths)
    return dict(ann_to_paths)


def build_viz_labels_from_nested_entities(
    entities: Mapping[str, Any],
) -> dict[str, str]:
    """
    Create human-readable labels for annotations found inside a nested entities
    structure. Each annotation id is mapped to a single label composed of
    sorted leaf key paths joined by \"|\".

    Args:
        entities (Mapping[str, Any]): Nested mapping/list structure containing
            annotation references.

    Returns:
        dict[str, str]: Mapping from annotation UUID to label string.
    """
    ann_index = flatten_entity_dict_to_ann_index(entities)
    return {ann_id: "|".join(sorted(paths)) for ann_id, paths in ann_index.items()}



@dataclass
class Document:
    """
    Document class for managing multi-page documents.

    Supports PDF documents and image collections with:
    - Lazy loading for memory efficiency
    - Page-level and document-level annotations
    - JSON serialization/deserialization
    - Pipeline metadata tracking

    Attributes:
        file_name: Name of the document file
        location: Path to the document or directory containing images
        document_type: Type of document (PDF or image collection)
        external_id: Optional external identifier
        document_id: UUID-like identifier used to identify the document instance.
        compute_metadata: Whether to compute page references during initialization.
        pipeline_sessions: Stored pipeline session metadata.

    Example:
        ```python
        # Create from PDF
        doc = Document(
            file_name="report.pdf",
            location="/path/to/report.pdf",
            document_type=DocumentType.PDF
        )

        # Iterate pages (lazy loaded)
        for page in doc.pages:
            print(page.text)

        # Save processed document
        doc.save("/output/dir")

        # Load from saved JSON
        doc = Document.from_directory("/output/dir")
        ```
    """

    file_name: str = ""
    location: Path = field(default_factory=Path)
    document_type: Optional[DocumentFileLabel] = None
    external_id: Optional[str] = None
    document_id: str = ""
    compute_metadata: bool = True

    pipeline_sessions: dict[str, PipelineSession] = field(default_factory=dict)

    _page_references: dict[int, PageReference] = field(default_factory=dict, init=False, repr=False)
    _images: dict[str, Image] = field(default_factory=dict, init=False, repr=False)
    _summary: Optional[CategoryAnnotation] = field(default=None, init=False, repr=False)
    _pdf_bytes: Optional[bytes] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.location is None or self.location == "":
            self.location = Path()
        elif not isinstance(self.location, Path):
            self.location = Path(self.location)

        self._resolve_document_type()

        if not self.document_id:
            self.document_id = get_uuid_from_str(str(self.location))

        if self.compute_metadata:
            self._initialize_page_references()

    def _resolve_document_type(self) -> None:
        """
        Infer `document_type` from `location`.

        Rules:
        - If `location` is a directory: `DocumentFileType.IMAGE_COLLECTION`
        - If `location` is a file with suffix `.pdf`: `DocumentFileType.PDF`

        Leaves `document_type` unchanged if it is already set or cannot be resolved.
        """
        if self.document_type is not None:
            return
        if not self.location:
            return

        if self.location.exists() and self.location.is_dir():
            self.document_type = DocumentFileLabel.VAR
            return

        if self.location.exists() and self.location.is_file() and self.location.suffix.lower() == ".pdf":
            self.document_type = DocumentFileLabel.PDF
            self.file_name = self.location.name
            return

    def _initialize_page_references(self) -> None:
        """Initialize page references without loading page data."""
        if not self.location:
            return

        if self.document_type == DocumentFileLabel.PDF:
            if not self.location.exists():
                raise FileNotFoundError(f"Document location does not exist: {self.location}")
            self._load_pdf_metadata()
            return

        if self.document_type == DocumentFileLabel.VAR:
            if not self.location.exists():
                raise FileNotFoundError(f"Document location does not exist: {self.location}")
            self._load_image_metadata()
            return

    @property
    def number_of_pages(self) -> int:
        """Number of pages in the document."""
        if self._page_references:
            return len(self._page_references)

        if self.document_type == DocumentFileLabel.PDF:
            self._load_pdf_metadata()

        return len(self._page_references)

    def _load_image_metadata(self) -> None:

        df = SerializerFiles.load(self.location, file_type=(".png", ".jpg", ".jpeg"))
        df.reset_state()

        refs: dict[int, PageReference] = {}
        loaded: dict[str, Image] = {}

        for page_number, dp in enumerate(df):
            location = Path(dp)
            image = Image(
                file_name=location.name,
                location=dp,
                page_number=page_number,
            )
            loaded[image.image_id] = image

            refs[page_number] = PageReference(source_path=dp, image_id=image.image_id)

        self._page_references = refs
        self._images = loaded

    def _load_pdf_metadata(self) -> None:
        location_path = Path(self.location)
        if not location_path.exists():
            return

        df = SerializerPdfDoc.load(path=location_path)
        df.reset_state()

        refs: dict[int, PageReference] = {}
        loaded: dict[int, Image] = {}

        for dp in df:
            image = Image(
                file_name=dp["file_name"],
                location=os.fspath(location_path),
                external_id=f"{dp['document_id']}_{dp['page_number']}",
                document_id=dp["document_id"],
                page_number=dp["page_number"],
            )
            image.pdf_bytes = dp["pdf_bytes"]

            refs[dp["page_number"]] = PageReference(
                page_number=dp["page_number"],
                source_path=os.fspath(location_path),
                image_id=image.image_id,
            )
            loaded[dp["page_number"]] = image

        self._page_references = refs

    def resolve_in_segments(self, obj: Any) -> Any:
        """
        Resolve UUID pairs into annotations.

        Supported leaves:
        - [image_id, annotation_id] -> ImageAnnotationBaseView (via self.get_annotation(...)[0])
        - [[image_id, annotation_id], ...] -> list[ImageAnnotationBaseView]

        Other values pass through unchanged (e.g. None).
        """

        def is_uuid_pair(x: Any) -> bool:
            return isinstance(x, list) and len(x) == 2 and is_uuid_like(x[0]) and is_uuid_like(x[1])

        if is_uuid_pair(obj):
            image_id, ann_id = obj
            return self.get_annotation(image_id, annotation_ids=ann_id)[0]

        if isinstance(obj, list) and obj and all(is_uuid_pair(item) for item in obj):
            resolved: list[tuple[str, ImageAnnotationBaseView]] = []
            for image_id, ann_id in obj:
                resolved.append((image_id, self.get_annotation(image_id, annotation_ids=ann_id)[0]))
            return resolved

        return obj

    def _walk_structured_output(self, obj: Any) -> Any:
        """
        Recursively walk a nested dict/list structure and replace UUID pairs with annotations.
        Preserves the original container shape.
        """
        if isinstance(obj, dict):
            return {k: self._walk_structured_output(v) for k, v in obj.items()}

        if isinstance(obj, list):
            resolved = self.resolve_in_segments(obj)
            if resolved is not obj:
                return resolved
            return [self._walk_structured_output(item) for item in obj]

        return self.resolve_in_segments(obj)

    @property
    def structured_output(self) -> dict[str, Any]:
        """structured output"""
        if "key_values" in self.summary.sub_categories:
            dict_value = self.summary.get_sub_category(get_type("key_values")).value  # type: ignore
            return self._walk_structured_output(dict_value)
        return {}

    @property
    def summary(self) -> CategoryAnnotation:
        """summary"""
        if self._summary is None:
            self._summary = CategoryAnnotation(
                category_name=SummaryKey.SUMMARY, external_id=self.document_id + "summary"
            )
        return self._summary

    @summary.setter
    def summary(self, summary_annotation: CategoryAnnotation) -> None:
        """summary setter"""
        if self._summary is not None:
            raise ValueError("Document.summary already defined and cannot be reset")
        self._summary = summary_annotation


    def get_image(
        self,
        page_number: Optional[int] = None,
        image_id: Optional[str] = None,
        load_pixels: bool = False,
    ) -> Image:
        """
        Get an `Image` instance for a page.

        Resolution order:
        1) fetch by `image_id` from `_images`
        2) ensure `_page_references` initialized, then fetch by `page_number`
        3) load/clear pixel payload based on `load_pixels` and `document_type`

        Args:
            page_number: 0-based page index to fetch.
            image_id: Image UUID to fetch directly.
            load_pixels: If True, ensure the image pixels/pdf bytes are loaded.

        Returns:
            Image: The resolved Image object with pixel payload present or cleared
            according to `load_pixels`.
        """
        if page_number is None and image_id is None:
            raise ValueError("Page number or image_id must be provided")

        def _load_pdf_page_bytes(target_page_number: Optional[int]) -> Optional[bytes]:
            df = SerializerPdfDoc.load(path=self.location)
            df.reset_state()
            for dp in enumerate(df):
                if dp[1]["page_number"] == target_page_number:
                    return dp[1]["pdf_bytes"]
            return None

        def _apply_pixel_policy(img: Image, target_page_number: Optional[int]) -> Image:
            if load_pixels and img._image is not None:
                return img

            if not load_pixels and img._image is not None:
                img.clear_image()
                return img

            if load_pixels and self.document_type == DocumentFileLabel.VAR:
                img.image = viz_handler.read_image(img.location)
                return img

            if load_pixels and self.document_type == DocumentFileLabel.PDF:
                img.image = _load_pdf_page_bytes(target_page_number)
                return img

            return img

        if image_id is not None:
            img = self._images[image_id]
            return _apply_pixel_policy(img, page_number)

        if not self._page_references:
            self._initialize_page_references()

        if page_number is None:
            page_number = 0

        if page_number < 0 or page_number > self.number_of_pages:
            raise IndexError(f"Page number {page_number} out of range (1-{self.number_of_pages})")

        ref = self._page_references.get(page_number)
        if ref is not None and ref.image_id and ref.image_id in self._images:
            img = self._images[ref.image_id]
            return _apply_pixel_policy(img, page_number)

        raise ValueError(f"Image for page {page_number} could not be found.")

    def get_page(
        self, page_number: Optional[int] = None, image_id: Optional[str] = None, load_image: bool = False
    ) -> Page:
        """
        Return a Page wrapper for a document page.

        This is a thin wrapper around `get_image` and converts the returned Image to a Page.

        Args:
            page_number (Optional[int]): 0-based page index.
            image_id (Optional[str]): Image UUID to fetch directly.
            load_image (bool): Whether to load pixel data for this page.

        Returns:
            Page: Page view created from the underlying Image.
        """
        image = self.get_image(page_number=page_number, image_id=image_id, load_pixels=load_image)
        return Page.from_image(image)

    def get_image_dataflow(self, load_pixels: bool = False) -> DataFlow:
        """
        Return a DataFlow that yields Image objects for all pages.

        This constructs a DataFromList over page indices and maps each index to the
        corresponding Image via `get_image`.

        Args:
            load_pixels (bool): Whether returned Image objects should include pixel data.

        Returns:
            DataFlow: DataFlow producing Image instances for each page.

        """

        @curry
        def get_image(page_number: int, load_pixels: bool) -> Image:
            return self.get_image(page_number=page_number, load_pixels=load_pixels)

        df = DataFromList(lst=list(range(self.number_of_pages)), shuffle=False)
        return MapData(df, get_image(load_pixels=load_pixels))

    def set_image(self, image: Image, page_number: int) -> None:
        """
        Store a processed `Image` for a page and update the corresponding `PageReference.image_id`.
        """
        maybe_reference = self._page_references.get(page_number)
        if maybe_reference:
            if (
                maybe_reference.image_id != image.image_id
                or maybe_reference.page_number != image.page_number
                or maybe_reference.source_path != image.location
            ):
                self._page_references[image.page_number] = PageReference(
                    source_path=image.location, page_number=image.page_number, image_id=image.image_id
                )
        else:
            self._page_references[page_number] = PageReference(
                image_id=image.image_id, page_number=page_number, source_path=image.location
            )
        self._images[image.image_id] = image

    def get_annotation(
        self,
        image_id: str,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        service_ids: Optional[Union[str, Sequence[str]]] = None,
        model_id: Optional[Union[str, Sequence[str]]] = None,
        session_ids: Optional[Union[str, Sequence[str]]] = None,
        ignore_inactive: bool = True,
    ) -> list[ImageAnnotationBaseView]:
        """
        Retrieve annotations from a specific image by various filters.

        Delegates to Page.get_annotation after wrapping the stored Image into a Page.

        Args:
            image_id: UUID of the image to query.
            category_names: Filter by category names or types.
            annotation_ids: Filter by annotation UUID(s).
            service_ids: Filter by service id(s).
            model_id: Filter by model id(s).
            session_ids: Filter by pipeline session id(s).
            ignore_inactive: If True, exclude inactive annotations.

        Returns:
            list[ImageAnnotationBaseView]: List of matching annotation views.
        """
        img = self._images[image_id]
        page = Page.from_image(img)
        return page.get_annotation(
            category_names=category_names,
            annotation_ids=annotation_ids,
            service_ids=service_ids,
            model_id=model_id,
            session_ids=session_ids,
            ignore_inactive=ignore_inactive,
        )

    def __repr__(self) -> str:
        return (
            f"Document(file_name={self.file_name!r}, "
            f"document_type={self.document_type.value if self.document_type else ''}, "
            f"pages={self.number_of_pages}, "
        )

    def remove_page(self, page_number: int) -> None:
        """
        Unload a page from memory.

        Useful for memory management when processing large documents.

        Args:
            page_number: 1-indexed page number
        """
        if page_number in self._page_references:
            image_id = self._page_references[page_number].image_id
            del self._images[image_id]  # type: ignore
            del self._page_references[page_number]

    def unload_all_pages(self) -> None:
        """Unload all pages from memory."""
        self._images.clear()
        self._page_references.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return document metadata as dict."""
        return {
            "file_name": self.file_name,
            "location": os.fspath(self.location) if self.location is not None else None,
            "document_type": self.document_type,
            "external_id": self.external_id,
            "document_id": self.document_id,
            "compute_metadata": self.compute_metadata,
            "pipeline_sessions": {k: asdict(v) for k, v in self.pipeline_sessions.items()},
            "_summary": self._summary.as_dict() if self._summary is not None else None,
            "_images": {key: val.as_dict() for key, val in self._images.items()},
            "_page_references": {key: asdict(value) for key, value in self._page_references.items()},
        }

    def as_json(self) -> str:
        """Return document metadata as JSON string."""
        return json.dumps(self.as_dict(), indent=4, default=str)

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:
        return self.number_of_pages

    def __getitem__(self, index: int) -> Page:
        """Get page by 0-indexed position."""
        return self.get_page(index)

    def save(
        self,
        image_to_json: bool = True,
        image_to_dir: bool = False,
        highest_hierarchy_only: bool = False,
        path: Optional[PathLikeOrStr] = None,
        dry: bool = False,
    ) -> Optional[Union[dict[str, Any], str]]:
        """
        Save the document instance to a JSON file (or return a dict when `dry=True`).

        Args:
            image_to_json: If `True` keeps image payloads when present; if `False` strips `_image` keys recursively.
            image_to_dir: If True, export pixel payloads as image files in a directory.
            highest_hierarchy_only: If `True` removes heavier/low level image data more aggressively.
            path: Path to save the `.json` file to. If `None`, uses `self.location`.
                  If a directory is provided, saves as `<dir>/<document_id>.json`.
            dry: If `True`, do not write files; return the export dict.

        Returns:
            A dict if `dry=True`, otherwise the JSON file path as string.
        """

        def set_image_keys_to_none(d: Any) -> None:
            if isinstance(d, dict):
                for key, value in d.items():
                    if key == "_image":
                        d[key] = None
                    else:
                        set_image_keys_to_none(value)
            elif isinstance(d, list):
                for item in d:
                    set_image_keys_to_none(item)

        if image_to_dir:
            image_to_json = False

        if image_to_dir:
            loc = Path(self.location) if self.location else Path()
            base_dir = loc if loc.is_dir() else loc.parent
            export_dir = base_dir / self.document_id
            mkdir_p(export_dir)

            for img in self._images.values():
                pixel_values = getattr(img, "image", None)
                if pixel_values is None:
                    continue
                png_path = export_dir / f"{img.image_id}.png"
                viz_handler.write_image(path=png_path, image=pixel_values)

        if path is None:
            path = Path(self.location)
        path = Path(path)

        if path.is_dir():
            path = path / self.document_id

        suffix = path.suffix
        if suffix:
            path_json = os.fspath(path).replace(suffix, ".json")
        else:
            path_json = os.fspath(path) + ".json"

        for img in self._images.values():
            if highest_hierarchy_only:
                img.remove_image_from_lower_hierarchy()
            else:
                img.remove_image_from_lower_hierarchy(pixel_values_only=True)

        export_dict = self.as_dict()
        if "location" in export_dict and export_dict["location"] is not None:
            export_dict["location"] = os.fspath(export_dict["location"])

        if not image_to_json:
            set_image_keys_to_none(export_dict)

        if dry:
            return export_dict

        with open(path_json, "w", encoding="UTF-8") as file:
            json.dump(export_dict, file, indent=2)

        return path_json

    @classmethod
    def from_json(cls, file_path: PathLikeOrStr) -> Document:
        """
        Create `Document` instance from `.json` file.

        Restores private attrs (e.g. `_images`, `_page_references`, `_summary`, `_processing_state`)
        that are not populated automatically by pydantic from input data.
        """
        with open(file_path, "r", encoding="UTF-8") as f:
            raw: dict[str, Any] = json.load(f)

        summary_raw = raw.pop("_summary", None)
        images_raw = raw.pop("_images", None)
        page_refs_raw = raw.pop("_page_references", None)
        pipeline_sessions = raw.pop("pipeline_sessions", {})

        if "_processing_state" in raw:
            raw.pop("_processing_state")

        raw["compute_metadata"] = False

        doc = cls(**raw)

        if pipeline_sessions:
            doc.pipeline_sessions = {key: PipelineSession(**val) for key, val in pipeline_sessions.items()}

        if doc.location:
            doc.location = Path(doc.location)

        if summary_raw is not None:
            doc._summary = CategoryAnnotation.from_dict(**summary_raw)

        if images_raw is not None:
            restored_images: dict[str, Image] = {}
            for image_id, img in images_raw.items():
                restored_images[image_id] = img if isinstance(img, Image) else Image(**img)
                if img["_image"]:
                    restored_images[image_id].image = img["_image"]
            doc._images = restored_images

        if page_refs_raw is not None:
            restored_refs: dict[int, PageReference] = {}
            for page_number, v in page_refs_raw.items():
                restored_refs[int(page_number)] = v if isinstance(v, PageReference) else PageReference(**v)
            doc._page_references = restored_refs

        return doc

    def viz_entities(  # type
        self,
        scaled_width: int = 900,
        show_tables: bool = True,
        show_layouts: bool = True,
        show_figures: bool = False,
        show_residual_layouts: bool = False,
        show_cells: bool = True,
        show_table_structure: bool = True,
    ) -> None:
        """
        Visualize all entity annotations found in `self.structured_output` on their corresponding pages.

        Args:
            scaled_width: Width to scale visualization to.
            show_tables: Whether to show detected tables.
            show_layouts: Whether to show detected page layouts.
            show_figures: Whether to show figures.
            show_residual_layouts: Whether to show residual layout annotations.
            show_cells: Whether to show table cell boxes.
            show_table_structure: Whether to visualize inferred table structure.

        """
        entities = self.structured_output

        ann_id_to_label = build_viz_labels_from_nested_entities(entities)

        image_to_ann_ids: defaultdict[str, set[str]] = defaultdict(set)

        _walk(entities, [], image_to_ann_ids)

        for image_id, ann_ids in image_to_ann_ids.items():
            seen: set[str] = set()
            ann_ids_unique = [a for a in ann_ids if not (a in seen or seen.add(a))]  # type:ignore

            page = self.get_page(image_id=image_id, load_image=True)

            page_labels = {ann_id: ann_id_to_label[ann_id] for ann_id in ann_ids_unique if ann_id in ann_id_to_label}

            page.viz(
                show_tables=show_tables,
                show_layouts=show_layouts,
                show_figures=show_figures,
                show_residual_layouts=show_residual_layouts,
                show_cells=show_cells,
                show_table_structure=show_table_structure,
                interactive=True,
                scaled_width=scaled_width,
                annotation_id_labels=page_labels,
                annotation_ids=ann_ids_unique,
            )