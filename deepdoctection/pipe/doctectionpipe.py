# -*- coding: utf-8 -*-
# File: doctectionpipe.py

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
Module for document processing pipeline
"""

import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

from ..dataflow import CustomDataFromIterable, DataFlow, DataFromList, MapData
from ..dataflow.custom_serialize import SerializerFiles, SerializerPdfDoc
from ..datapoint.image import Image
from ..datapoint.view import IMAGE_DEFAULTS
from ..mapper.maputils import curry
from ..mapper.misc import to_image
from ..utils.fs import maybe_path_or_pdf
from ..utils.identifier import get_uuid_from_str
from ..utils.logger import LoggingRecord, logger
from ..utils.pdf_utils import PDFStreamer
from ..utils.types import PathLikeOrStr
from ..utils.utils import is_file_extension
from .base import Pipeline, PipelineComponent
from .common import PageParsingService


def _collect_from_kwargs(
    **kwargs: Union[Optional[str], bytes, DataFlow, bool, int, PathLikeOrStr, Union[str, List[str]]]
) -> Tuple[Optional[str], Union[str, Sequence[str]], bool, int, str, DataFlow, Optional[bytes]]:
    """
    Collects and validates keyword arguments for dataflow construction.

    Args:
        **kwargs: Keyword arguments that may include `path`, `bytes`, `dataset_dataflow`, `shuffle`, `file_type`, and
            `max_datapoints`.

    Returns:
        Tuple containing `path`, `file_type`, `shuffle`, `max_datapoints`, `doc_path`, `dataset_dataflow`, and
        `b_bytes`.

    Raises:
        ValueError: If neither `path` nor `dataset_dataflow` is provided, or if required arguments are missing.
        TypeError: If argument types are incorrect.
    """
    b_bytes = kwargs.get("bytes")
    dataset_dataflow = kwargs.get("dataset_dataflow")
    path = kwargs.get("path")
    if path is None and dataset_dataflow is None:
        raise ValueError("Pass either path or dataset_dataflow as argument")
    if path is None and b_bytes:
        raise ValueError("When passing bytes, a path to the source document must be provided")

    shuffle = kwargs.get("shuffle", False)
    if not isinstance(shuffle, bool):
        raise TypeError(f"shuffle must be of type bool but is of type {type(shuffle)}")

    file_type = None
    doc_path = None
    if path:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be of type PathOrStr")
        path_type = maybe_path_or_pdf(path)
        if path_type == 2:
            doc_path = path
            path = None
            file_type = ".pdf"
        elif path_type == 3:
            if is_file_extension(path, ".jpg"):
                file_type = ".jpg"
            if is_file_extension(path, ".png"):
                file_type = ".png"
            if is_file_extension(path, ".jpeg"):
                file_type = ".jpeg"
            if not b_bytes:
                raise ValueError("When passing a path to a single image, bytes of the image must be passed")
        elif not path_type:
            raise ValueError("Pass only a path to a directory or to a pdf file")

    file_type = kwargs.get(
        "file_type", [".jpg", ".png", ".jpeg", ".tif"] if file_type is None else file_type  # type: ignore
    )

    max_datapoints = kwargs.get("max_datapoints")
    if not isinstance(max_datapoints, (int, type(None))):
        raise TypeError(f"max_datapoints must be of type int, but is of type {type(max_datapoints)}")
    return path, file_type, shuffle, max_datapoints, doc_path, dataset_dataflow, b_bytes  # type: ignore


@curry
def _proto_process(
    dp: Union[str, Mapping[str, str]], path: Optional[PathLikeOrStr], doc_path: Optional[PathLikeOrStr]
) -> Union[str, Mapping[str, str]]:
    if isinstance(dp, str):
        file_name = Path(dp).name
    elif isinstance(dp, Image):
        file_name = dp.file_name
    else:
        file_name = dp["file_name"]
    if path is None:
        path_tmp = doc_path or ""
    else:
        path_tmp = path
    logger.info(
        LoggingRecord(
            f"Processing {file_name}", {"path": os.fspath(path_tmp), "df": os.fspath(path_tmp), "file_name": file_name}
        )
    )
    return dp


@curry
def _to_image(
    dp: Union[str, Mapping[str, Union[str, bytes]]],
    dpi: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Image]:
    """
    Converts a data point to an `Image` object.

    Args:
        dp: The data point, which can be a string or a mapping.
        dpi: Dots per inch for the image.
        width: Width of the image.
        height: Height of the image.

    Returns:
        An `Image` object or None.
    """
    return to_image(dp, dpi, width, height)


def _doc_to_dataflow(path: PathLikeOrStr, max_datapoints: Optional[int] = None) -> DataFlow:
    """
    Creates a dataflow from a PDF document.

    Args:
        path: Path to the PDF document.
        max_datapoints: Maximum number of data points to consider.

    Returns:
        A `DataFlow` object.

    Raises:
        FileExistsError: If the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileExistsError(f"{path} not a file")

    df = SerializerPdfDoc.load(path, max_datapoints=max_datapoints)

    return df


class DoctectionPipe(Pipeline):
    """
    Prototype for a document layout pipeline.

    Contains implementation for loading document types (images in directory, single PDF document, dataflow from
     datasets), conversions in dataflows, and building a pipeline.

    See `deepdoctection.analyzer.dd` for a concrete implementation.

    See also the explanations in `base.Pipeline`.

    By default, `DoctectionPipe` will instantiate a default `PageParsingService`:

    Example:
        ```python
        pipe = DoctectionPipe([comp_1, com_2], PageParsingService(text_container= my_custom_setting))
        ```

    Note:
        You can overwrite the current setting by providing a custom `PageParsingService`.
    """

    def __init__(
        self,
        pipeline_component_list: List[PipelineComponent],
        page_parsing_service: Optional[PageParsingService] = None,
    ):
        """
        Initializes the `DoctectionPipe`.

        Args:
            pipeline_component_list: List of pipeline components.
            page_parsing_service: Optional custom `PageParsingService`.
        """
        self.page_parser = (
            PageParsingService(
                text_container=IMAGE_DEFAULTS.TEXT_CONTAINER,
            )
            if page_parsing_service is None
            else page_parsing_service
        )

        super().__init__(pipeline_component_list)

    def _entry(
        self, **kwargs: Union[str, bytes, DataFlow, bool, int, PathLikeOrStr, Union[str, List[str]]]
    ) -> DataFlow:
        path, file_type, shuffle, max_datapoints, doc_path, dataset_dataflow, b_bytes = _collect_from_kwargs(**kwargs)

        df: DataFlow

        if isinstance(b_bytes, bytes):
            df = DoctectionPipe.bytes_to_dataflow(
                path=doc_path if path is None else path, b_bytes=b_bytes, file_type=file_type
            )

        elif isinstance(path, (str, Path)):
            if not isinstance(file_type, (str, list)):
                raise TypeError(f"file_type must be of type string or list, but is of type {type(file_type)}")
            df = DoctectionPipe.path_to_dataflow(path=path, file_type=file_type, shuffle=shuffle)
        elif isinstance(doc_path, (str, Path)):
            df = DoctectionPipe.doc_to_dataflow(
                path=doc_path, max_datapoints=int(max_datapoints) if max_datapoints is not None else None
            )
        elif dataset_dataflow is not None and isinstance(dataset_dataflow, DataFlow):
            df = dataset_dataflow
        else:
            raise BrokenPipeError("Cannot build Dataflow")

        df = MapData(df, _proto_process(path, doc_path))
        if dataset_dataflow is None:
            if dpi := os.environ["DPI"]:
                df = MapData(df, _to_image(dpi=int(dpi)))  # pylint: disable=E1120
            else:
                width, height = kwargs.get("width", ""), kwargs.get("height", "")
                if not width or not height:
                    width = os.environ["IMAGE_WIDTH"]
                    height = os.environ["IMAGE_HEIGHT"]
                    if not width or not height:
                        raise ValueError(
                            "DPI, IMAGE_WIDTH and IMAGE_HEIGHT are all None, but "
                            "either DPI or IMAGE_WIDTH and IMAGE_HEIGHT must be set"
                        )
                df = MapData(df, _to_image(width=int(width), height=int(height)))  # pylint: disable=E1120
        return df

    @staticmethod
    def path_to_dataflow(
        path: PathLikeOrStr,
        file_type: Union[str, Sequence[str]],
        max_datapoints: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataFlow:
        """
        Processing method for directories.

        Args:
            path: Path to directory.
            file_type: File type to consider (single string or list of strings).
            max_datapoints: Maximum number of data points to consider.
            shuffle: Whether to shuffle file names for random streaming.

        Returns:
            A `DataFlow` object.

        Raises:
            NotADirectoryError: If the path is not a directory.
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{os.fspath(path)} not a directory")
        df = SerializerFiles.load(path, file_type, max_datapoints, shuffle)
        return df

    @staticmethod
    def doc_to_dataflow(path: PathLikeOrStr, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Processing method for documents.

        Args:
            path: Path to the document.
            max_datapoints: Maximum number of data points to consider.

        Returns:
            A `DataFlow` object.
        """
        return _doc_to_dataflow(path, max_datapoints)

    @staticmethod
    def bytes_to_dataflow(
        path: str, b_bytes: bytes, file_type: Union[str, Sequence[str]], max_datapoints: Optional[int] = None
    ) -> DataFlow:
        """
        Converts a bytes object to a dataflow.

        Args:
            path: Path to directory or an image file.
            b_bytes: Bytes object.
            file_type: File type, e.g., `.pdf`, `.jpg`, or a list of image file types.
            max_datapoints: Maximum number of data points to consider.

        Returns:
            A `DataFlow` object.

        Raises:
            ValueError: If the combination of arguments is not supported.
        """

        file_name = os.path.split(path)[1]
        if isinstance(file_type, str):
            if file_type == ".pdf":
                prefix, suffix = os.path.splitext(file_name)
                df: DataFlow
                df = CustomDataFromIterable(PDFStreamer(path_or_bytes=b_bytes), max_datapoints=max_datapoints)
                df = MapData(
                    df,
                    lambda dp: {
                        "path": path,
                        "file_name": prefix + f"_{dp[1]}" + suffix,
                        "pdf_bytes": dp[0],
                        "page_number": dp[1],
                        "document_id": get_uuid_from_str(prefix),
                    },
                )
            else:
                df = DataFromList(lst=[{"path": path, "file_name": file_name, "image_bytes": b_bytes}])
            return df
        raise ValueError(
            f"pass: {path}, b_bytes: {b_bytes!r}, file_type: {file_type} and max_datapoints: {max_datapoints} "
            f"not supported"
        )

    def dataflow_to_page(self, df: DataFlow) -> DataFlow:
        """
        Converts a dataflow of images to a dataflow of pages.

        Args:
            df: Dataflow.

        Returns:
            A dataflow of pages.
        """
        return self.page_parser.predict_dataflow(df)

    def analyze(
        self, **kwargs: Union[str, bytes, DataFlow, bool, int, PathLikeOrStr, Union[str, List[str]]]
    ) -> DataFlow:
        """
        Args:
            `kwargs:
                 dataset_dataflow (Dataflow):` Transfer a dataflow of a dataset via its dataflow builder
                 path (TypeOrStr):` A path to a directory in which either image documents or pdf files are located. It
                               is assumed that the pdf documents consist of only one page. If there are multiple pages,
                               only the first page is processed through the pipeline.
                               Alternatively, a path to a pdf document with multiple pages.
                 bytes:` A bytes object of an image
                 file_type:` Selection of the file type, if: args:`file_type` is passed
                 max_datapoints:` Stops processing as soon as max_datapoints images have been processed

        :return: dataflow
        """

        output = kwargs.get("output", "page")
        session_id = kwargs.get("session_id")
        assert output in ("page", "image", "dict"), "output must be either page image or dict"
        df = self._entry(**kwargs)
        df = self._build_pipe(df, session_id=session_id)  # type: ignore
        if output == "page":
            df = self.dataflow_to_page(df)
        elif output == "dict":
            df = self.dataflow_to_page(df)
            df = MapData(df, lambda dp: dp.as_dict())
        return df
