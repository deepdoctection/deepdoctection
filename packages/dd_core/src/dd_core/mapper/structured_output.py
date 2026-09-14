# -*- coding: utf-8 -*-
# File: structured_output.py

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
Mapping function for turning an `Image` or `doc.Document`'s summary sub categories into the nested-dict
"structured output" shape that `RecordAlignMetric`'s `field_counts` compares.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from ..datapoint.annotation import ContainerAnnotation, ReferencePayload
from ..datapoint.image import Image
from ..datapoint.view import Page
from ..utils.object_types import TypeOrStr, get_type
from .maputils import curry

if TYPE_CHECKING:
    from ..doc import Document

__all__ = ["image_or_doc_to_structured_output"]


@curry
def image_or_doc_to_structured_output(
    dp: Union[Image, "Document"],
    summary_sub_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
) -> tuple[dict[TypeOrStr, list[Any]], str]:
    """
    Extracts one or more summary sub categories, resolving any `ReferencePayload` value, into a
    defaultdict of lists. Used by `RecordAlignMetric` when `_summary_sub_cats` (e.g. `structured_output`)
    is configured, for both `Image` (`mode="image"`) and `doc.Document` (`mode="doc"`) datapoints.

    Unlike `image_to_cat_id`, this mapper always resolves `ReferencePayload` values explicitly rather
    than relying on `Document.structured_output`/`Page.structured_output`, so it works for arbitrary
    summary sub category names, not only `"structured_output"`, and behaves the same whether the
    underlying value is a genuine `ReferencePayload` or an already resolved plain value - both can occur
    at either the `Document` level or the `Image`/`Page` level.

    Example:
        ```python
        image_or_doc_to_structured_output(summary_sub_category_names="structured_output")(dp)
        # Output: ({SummaryKey.STRUCTURED_OUTPUT: [{"amount": ["1,00-"], ...}]}, image_id_or_document_id)
        ```

    Args:
        dp: Image or Document
        summary_sub_category_names: A single summary sub category name or a sequence of them. Required -
            there is no default.

    Returns:
        A defaultdict of lists keyed by the (unresolved `ObjectTypes`) summary sub category name, and the
        `image_id` (or `document_id` for a `Document`).

    Raises:
        ValueError: If `summary_sub_category_names` is not given, or a requested summary sub category is
                    not a `ContainerAnnotation`.
    """
    # local import: dd_core.doc imports dd_core.mapper.maputils, so importing Document at module level
    # here would create a circular import
    from ..doc import Document  # pylint: disable=C0415

    if isinstance(summary_sub_category_names, str):
        summary_sub_category_names = [summary_sub_category_names]
    if not summary_sub_category_names:
        raise ValueError("summary_sub_category_names must be given (e.g. 'structured_output'); there is no default.")

    if isinstance(dp, Document):
        target: Union[Document, Page] = dp
        id_ = dp.document_id
    else:
        target = Page.from_image(dp)
        id_ = dp.image_id

    cat_container: dict[TypeOrStr, list[Any]] = defaultdict(list)
    for name in summary_sub_category_names:
        sub_cat = target.summary.get_sub_category(get_type(name))
        if not isinstance(sub_cat, ContainerAnnotation):
            raise ValueError(
                f"summary sub category {name} does not have a ContainerAnnotation. "
                f"image_or_doc_to_structured_output only supports ContainerAnnotation values."
            )
        value = sub_cat.value
        resolved = target.resolve_reference_payload(value) if isinstance(value, ReferencePayload) else value
        cat_container[name].append(resolved)

    return cat_container, id_
