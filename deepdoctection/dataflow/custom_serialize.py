# -*- coding: utf-8 -*-
# File: custom_serialize.py

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
Adding some methods that convert incoming data to dataflows.
"""

import os
from typing import List, Optional, Union

from dataflow.dataflow import DataFlow, JoinData, MapData  # type: ignore
from jsonlines import Reader, Writer  # type: ignore
from pycocotools.coco import COCO

from ..utils.fs import is_file_extension
from ..utils.logger import logger
from ..utils.pdf_utils import PDFStreamer
from ..utils.timer import timed_operation
from ..utils.tqdm import get_tqdm
from .common import FlattenData
from .custom import CacheData, CustomDataFromIterable, CustomDataFromList

__all__ = ["SerializerJsonlines", "SerializerFiles", "SerializerCoco", "SerializerPdfDoc"]


def _reset_df_and_get_length(df: DataFlow) -> int:
    df.reset_state()
    try:
        length = len(df)
    except NotImplementedError:
        length = 0
    return length


class SerializerJsonlines:
    """
    Serialize a dataflow from a jsonlines file. Alternatively, save a dataflow of JSON objects to a .jsonl file.
    """

    @staticmethod
    def load(path: str, max_datapoints: Optional[int] = None) -> CustomDataFromIterable:
        """
        :param path: a path to a .jsonl file.
        :param max_datapoints: Will stop the iteration once max_datapoints have been streamed

        :return: dataflow to iterate from
        """
        file = open(path, "r")  # pylint: disable=W1514,R1732
        iterator = Reader(file)
        return CustomDataFromIterable(iterator, max_datapoints=max_datapoints)

    @staticmethod
    def save(df: DataFlow, path: str, file_name: str, max_datapoints: Optional[int] = None) -> None:
        """
        Writes a dataflow iteratively to a .jsonl file. Every datapoint must be a dict where all items are serializable.
        As the length of the dataflow cannot be determined in every case max_datapoint prevents generating an
        unexpectedly large file

        :param df: The dataflow to write from.
        :param path: The path, the .jsonl file to write to.
        :param file_name: name of the target file.
        :param max_datapoints: maximum number of datapoint to consider writing to a file.
        """

        assert os.path.isdir(path), f"not a dir {path}"
        assert is_file_extension(file_name, ".jsonl")

        with open(os.path.join(path, file_name), "w") as file:  # pylint: disable=W1514
            writer = Writer(file)
            length = _reset_df_and_get_length(df)
            if length == 0:
                logger.info("cannot estimate length of dataflow")
            if max_datapoints is not None:
                if max_datapoints < length:
                    logger.info("dataflow larger than max_datapoints")
            for k, dp in enumerate(df):
                if max_datapoints is None:
                    writer.write(dp)
                elif k < max_datapoints:
                    writer.write(dp)
                else:
                    break


class SerializerFiles:
    """
    Serialize files from a directory and all subdirectories. Only one file type can be serialized. Once specified, all
    other types will be filtered out.
    """

    @staticmethod
    def load(
        path: str,
        file_type: Union[str, List[str]],
        max_datapoints: Optional[int] = None,
        shuffle: Optional[bool] = False,
        sort: Optional[bool] = True,
    ) -> DataFlow:
        """
        Generates a dataflow where a datapoint consists of a string of names of files with respect to some file type.
        If you want to load the files you need to do this in a following step by yourself.

        :param path: A path to some base directory. Will inspect all subdirectories, as well
        :param file_type: A file type (suffix) to look out for (single str or list of stings)
        :param max_datapoints: Stop iteration after passing max_datapoints
        :param shuffle: Shuffle the files, so that the order of appearance in dataflow is random.
        :param sort: If set to "True" it will sort all selected files by its string
        :return: dataflow to iterate from
        """
        if shuffle:
            sort = False
        it1 = os.walk(path, topdown=False)
        it2 = os.walk(path, topdown=False)
        df1 = CustomDataFromIterable(it1)
        df2 = CustomDataFromIterable(it2)
        df1 = MapData(df1, lambda dp: None if len(dp[2]) == 0 else dp)
        df2 = MapData(df2, lambda dp: None if len(dp[2]) == 0 else dp)
        df1 = MapData(df1, lambda dp: [dp[0]] * len(dp[2]))
        df2 = MapData(df2, lambda dp: dp[2])
        df1 = FlattenData(df1)
        df2 = FlattenData(df2)
        df3 = JoinData(df_lists=[df1, df2])
        df3 = MapData(df3, lambda dp: os.path.join(dp[0], dp[1]))
        df = MapData(df3, lambda dp: dp if is_file_extension(dp, file_type) else None)
        if max_datapoints is not None or sort:
            df_list = CacheData(df).get_cache()
            if sort:
                df_list.sort()
            df = CustomDataFromList(df_list, max_datapoints=max_datapoints, shuffle=False)
        elif shuffle:
            df_list = CacheData(df).get_cache()
            df = CustomDataFromList(df_list, shuffle=shuffle)

        return df

    @staticmethod
    def save() -> None:
        """
        Not implemented
        """
        raise NotImplementedError


class SerializerCoco:
    """
    Class for serializing annotation files in Coco format. Coco comes in JSON format which is a priori not
    serialized. This class implements only the very basic methods to generate a dataflow. It wraps the coco class
    from pycocotools and assembles annotations that belong to the image. Note, that the conversion into the core
    :class:`Image` has to be done by yourself.
    """

    @staticmethod
    def load(path: str, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Loads a .json file and generates a dataflow.

        **Example:**

            {'images':[img1,img2,...], 'annotations':[ann1,ann2,...],...}

            it will generate a dataflow  with datapoints

            {'image':{'id',...},'annotations':[{'id':…,'bbox':...}]}

            for each single image id.

        :param max_datapoints: Will stop the iteration once max_datapoints have been streamed.
        :param path: a path to a .json file.
        :return: dataflow to iterate from
        """
        assert os.path.isfile(path), path
        file = os.path.split(path)[1]
        assert is_file_extension(file, ".json"), file

        with timed_operation("Start loading .json file and serializing"):
            coco = COCO(path)
            img_ids = coco.getImgIds()
            imgs = coco.loadImgs(img_ids)

            with get_tqdm(total=len(imgs)) as status_bar:
                for img in imgs:
                    img["annotations"] = coco.imgToAnns[img["id"]]
                    status_bar.update()

        df = CustomDataFromList(imgs, max_datapoints=max_datapoints)
        return df

    @staticmethod
    def save() -> None:
        """
        Not implemented
        """
        raise NotImplementedError


class SerializerPdfDoc:
    """
    Serialize a pdf document with an arbitrary number of pages.
    """

    @staticmethod
    def load(path: str, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Loads the document page wise and returns a dataflow accordingly.

        :param path: Path to the pdf document.
        :param max_datapoints: The maximum number of pages to stream.
        :return: A dict with structure {"path":... ,"file_name": ..., "pdf_bytes": ...}. The file name is a
                 concatenation of the physical file name and the current page number.
        """

        file_name = os.path.split(path)[1]
        prefix, suffix = os.path.splitext(file_name)
        df = CustomDataFromIterable(PDFStreamer(path=path), max_datapoints=max_datapoints)
        df = MapData(df, lambda dp: {"path": path, "file_name": prefix + f"_{dp[1]}" + suffix, "pdf_bytes": dp[0]})
        return df

    @staticmethod
    def save(path: str) -> None:
        """
        Not implemented
        """
        raise NotImplementedError

    @staticmethod
    def split(path: str, path_target: Optional[str] = None, max_datapoint: Optional[int] = None) -> None:
        """
        Split a Document into single pages.
        """
        if path_target is None:
            path_target, _ = os.path.split(path)
        assert os.path.isdir(path_target), f"not a dir {path_target}"
        df = SerializerPdfDoc.load(path, max_datapoint)
        for dp in df:
            with open(os.path.join(path_target, dp["file_name"]), "wb") as page:
                page.write(dp["pdf_bytes"])
