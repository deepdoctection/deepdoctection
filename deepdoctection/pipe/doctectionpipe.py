# -*- coding: utf-8 -*-
# File: tpdoctectionpipe.py

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
Module for pipeline with Tensorpack predictors
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..dataflow import DataFlow, MapData
from ..dataflow.custom_serialize import SerializerFiles, SerializerPdfDoc
from ..datapoint.image import Image
from ..mapper.maputils import curry
from ..mapper.misc import to_image
from ..utils.detection_types import Pathlike
from ..utils.fs import maybe_path_or_pdf
from ..utils.logger import logger
from ..utils.settings import LayoutType
from .base import Pipeline, PipelineComponent, PredictorPipelineComponent
from .common import PageParsingService


class DoctectionPipe(Pipeline):
    """
    Prototype for a document layout pipeline. Contains implementation for loading document types (images in directory,
    single PDF document, dataflow from datasets), conversions in dataflows and building a pipeline.

    See `deepdoctection.analyzer.dd` for a concrete implementation.

    See also the explanations in :class:`base.Pipeline`.

    By default, :class:`DoctectionPipe` will instantiate a default `PageParsingService`

    .. code-block:: python

        PageParsingService(text_container=LayoutType.word,
                           text_block_names=[LayoutType.title,
                                             LayoutType.text,
                                             LayoutType.list,
                                             LayoutType.table])

    but you can overwrite the current setting:

    **Example:**

        .. code-block:: python

            pipe = DoctectionPipe([comp_1, com_2])
            pipe.page_parser =  PageParsingService(text_container= my_custom_setting)


    """

    def __init__(self, pipeline_component_list: List[Union[PipelineComponent]]):
        self.page_parser = PageParsingService(
            text_container=LayoutType.word,
            text_block_names=[LayoutType.title, LayoutType.text, LayoutType.list, LayoutType.table],
        )
        assert all(
            isinstance(element, (PipelineComponent, PredictorPipelineComponent)) for element in pipeline_component_list
        )
        super().__init__(pipeline_component_list)

    def _entry(self, **kwargs: Union[str, DataFlow, bool, int, Pathlike, Union[str, List[str]]]) -> DataFlow:
        dataset_dataflow = kwargs.get("dataset_dataflow")
        path = kwargs.get("path")
        if path is None and dataset_dataflow is None:
            raise ValueError("Pass either path or dataset_dataflow as argument")

        shuffle = kwargs.get("shuffle", False)
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be of type bool but is of type {type(shuffle)}")

        doc_path = None
        if path:
            if not isinstance(path, (str, Path)):
                raise TypeError("path must be of type PathOrStr")
            path_type = maybe_path_or_pdf(path)
            if path_type == 2:
                doc_path = path
                path = None
            elif not path_type:
                raise ValueError("Pass only a path to a directory or to a pdf file")

        file_type = kwargs.get("file_type", [".jpg", ".png", ".tif"])

        max_datapoints = kwargs.get("max_datapoints")
        if not isinstance(max_datapoints, (int, type(None))):
            raise TypeError(f"max_datapoints must be of type int, but is of type {type(max_datapoints)}")
        df: DataFlow

        if isinstance(path, (str, Path)):
            if not isinstance(file_type, (str, list)):
                raise TypeError(f"file_type must be of type string or list, but is of type {type(file_type)}")
            df = DoctectionPipe.path_to_dataflow(path, file_type, shuffle=shuffle)
        if isinstance(doc_path, (str, Path)):
            df = DoctectionPipe.doc_to_dataflow(
                path=doc_path, max_datapoints=int(max_datapoints) if max_datapoints is not None else None
            )
        if dataset_dataflow is not None and isinstance(dataset_dataflow, DataFlow):
            df = dataset_dataflow

        def _proto_process(dp: Image) -> Image:
            if path is None:
                path_tmp = doc_path
            else:
                path_tmp = path  # type: ignore
            logger.info("Processing %s", dp.file_name, {"path": path_tmp, "df": path_tmp, "file_name": dp.file_name})
            return dp

        df = MapData(df, _proto_process)

        return df

    @staticmethod
    def path_to_dataflow(
        path: Pathlike,
        file_type: Union[str, Sequence[str]],
        max_datapoints: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataFlow:
        """
        Processing method for directories

        :param path: path to directory
        :param file_type: file type to consider (single str or list of strings)
        :param max_datapoints: max number of datapoints to consider
        :param shuffle: Shuffle file names in order to stream them randomly
        :return: dataflow
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} not a directory")
        df = SerializerFiles.load(path, file_type, max_datapoints, shuffle)

        def _to_image(dp: str) -> Optional[Image]:
            _, file_name = os.path.split(dp)
            dp_dict = {"file_name": file_name, "location": dp}
            return to_image(dp_dict)

        df = MapData(df, _to_image)
        return df

    @staticmethod
    def doc_to_dataflow(path: Pathlike, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Processing method for documents

        :param path: path to directory
        :param max_datapoints: max number of datapoints to consider
        :return: dataflow
        """
        if not os.path.isfile(path):
            raise FileExistsError(f"{path} not a file")

        df = SerializerPdfDoc.load(path, max_datapoints=max_datapoints)

        @curry
        def _to_image(dp: Union[str, Dict[str, Union[str, bytes]]], dpi: Optional[int] = None) -> Optional[Image]:
            return to_image(dp, dpi)

        df = MapData(df, _to_image(dpi=300))  # pylint: disable=E1120
        return df

    def dataflow_to_page(self, df: DataFlow) -> DataFlow:
        """
        Converts a dataflow of images to a dataflow of pages

        :param df: Dataflow
        :return: Dataflow
        """
        return self.page_parser.predict_dataflow(df)

    def analyze(self, **kwargs: Union[str, DataFlow, bool, int, Pathlike, Union[str, List[str]]]) -> DataFlow:
        """
        :param kwargs key dataset_dataflow: Transfer a dataflow of a dataset via its dataflow builder
        :param kwargs key path: A path to a directory in which either image documents or pdf files are located. It is
                                assumed that the pdf documents consist of only one page. If there are multiple pages,
                                only the first page is processed through the pipeline.
                                Alternatively, a path to a pdf document with multiple pages.
        :param kwargs key file_type: Selection of the file type, if: args:`file_type` is passed
        :param kwargs key max_datapoints: Stops processing as soon as max_datapoints images have been processed
        :return: dataflow
        """

        output = kwargs.get("output", "page")
        assert output in ("page", "image", "dict"), "output must be either page image or dict"
        df = self._entry(**kwargs)
        df = self._build_pipe(df)
        if output == "page":
            df = self.dataflow_to_page(df)
        elif output == "dict":
            df = self.dataflow_to_page(df)
            df = MapData(df, lambda dp: dp.as_dict())
        return df
