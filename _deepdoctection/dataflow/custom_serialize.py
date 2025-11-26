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
Classes to load data and produce dataflows
"""

from __future__ import annotations

import itertools
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Sequence, TextIO, Union

from jsonlines import Reader, Writer
from tabulate import tabulate
from termcolor import colored

from ..utils.context import timed_operation
from ..utils.error import FileExtensionError
from ..utils.identifier import get_uuid_from_str
from ..utils.pdf_utils import PDFStreamer
from ..utils.tqdm import get_tqdm
from ..utils.types import JsonDict, PathLikeOrStr
from ..utils.utils import is_file_extension
from .base import DataFlow
from .common import FlattenData, JoinData, MapData
from .custom import CacheData, CustomDataFromIterable, CustomDataFromList

__all__ = ["SerializerJsonlines", "SerializerFiles", "SerializerCoco", "SerializerPdfDoc", "SerializerTabsepFiles"]


def _reset_df_and_get_length(df: DataFlow) -> int:
    df.reset_state()
    try:
        length = len(df)
    except NotImplementedError:
        length = 0
    return length


class FileClosingIterator:
    """
    A custom iterator that closes the file object once the iteration is complete.

    This iterator is used to ensure that the file object is properly closed after
    reading the data from it. It is used in the context of reading data from a file
    in a streaming manner, where the data is not loaded into memory all at once.

    Example:
        ```python
        file = open(path, "r")
        iterator = Reader(file)
        closing_iterator = FileClosingIterator(file, iter(iterator))

        df = CustomDataFromIterable(closing_iterator, max_datapoints=max_datapoints)
        ```

    """

    def __init__(self, file_obj: TextIO, iterator: Iterator[Any]):
        """
        Initializes the FileClosingIterator with a file object and its iterator.

        Args:
            file_obj: The file object to read data from.
            iterator: The actual iterator of the file object.
        """
        self.file_obj = file_obj
        self.iterator = iterator

    def __iter__(self) -> FileClosingIterator:
        """
        Returns the iterator object itself.

        Returns:
            FileClosingIterator: The instance of the class itself.
        """
        return self

    def __next__(self) -> Any:
        """
        Returns the next item from the file object's iterator.
        Closes the file object if the iteration is finished.

        Returns:
            The next item from the file object's iterator.

        Raises:
            StopIteration: If there are no more items to return.
        """
        try:
            return next(self.iterator)
        except StopIteration as exc:
            self.file_obj.close()
            raise StopIteration from exc


class SerializerJsonlines:
    """
    Serialize a dataflow from a jsonlines file. Alternatively, save a dataflow of `JSON` objects to a `.jsonl` file.

    Example:
        ```python
          df = SerializerJsonlines.load("path/to/file.jsonl")
          df.reset_state()

          for dp in df:
              ... # is a dict
        ```
    """

    @staticmethod
    def load(path: PathLikeOrStr, max_datapoints: Optional[int] = None) -> CustomDataFromIterable:
        """
        Args:
            path: a path to a .jsonl file.
            max_datapoints: Will stop the iteration once max_datapoints have been streamed

        Returns:
            Dataflow to iterate from
        """
        file = open(path, "r")  # pylint: disable=W1514,R1732
        iterator = Reader(file)
        closing_iterator = FileClosingIterator(file, iter(iterator))
        return CustomDataFromIterable(closing_iterator, max_datapoints=max_datapoints)

    @staticmethod
    def save(df: DataFlow, path: PathLikeOrStr, file_name: str, max_datapoints: Optional[int] = None) -> None:
        """
        Writes a dataflow iteratively to a `.jsonl` file. Every datapoint must be a dict where all items are
        serializable. As the length of the dataflow cannot be determined in every case max_datapoint prevents
        generating an unexpectedly large file

        Args:
            df: The dataflow to write from.
            path: The path, the .jsonl file to write to.
            file_name: name of the target file.
            max_datapoints: maximum number of datapoint to consider writing to a file.
        """

        if not os.path.isdir(path):
            raise NotADirectoryError(path)
        if not is_file_extension(file_name, ".jsonl"):
            raise FileExtensionError(f"Expected .jsonl file got {path}")

        df.reset_state()
        with open(os.path.join(path, file_name), "w") as file:  # pylint: disable=W1514
            writer = Writer(file)
            for k, dp in enumerate(df):
                if max_datapoints is None:
                    writer.write(dp)
                elif k < max_datapoints:
                    writer.write(dp)
                else:
                    break


class SerializerTabsepFiles:
    """
    Serialize a dataflow from a tab separated text file. Alternatively, save a dataflow of plain text
    to a `.txt` file.

    Example:
        ```python
        df = SerializerTabsepFiles.load("path/to/file.txt")

        will yield each text line of the file.
        ```
    """

    @staticmethod
    def load(path: PathLikeOrStr, max_datapoints: Optional[int] = None) -> CustomDataFromList:
        """
        Args:
            path: a path to a .txt file.
            max_datapoints: Will stop the iteration once max_datapoints have been streamed

        Returns:
            Dataflow to iterate from
        """

        with open(path, "r", encoding="UTF-8") as file:
            file_list = file.readlines()
        return CustomDataFromList(file_list, max_datapoints=max_datapoints)

    @staticmethod
    def save(df: DataFlow, path: PathLikeOrStr, file_name: str, max_datapoints: Optional[int] = None) -> None:
        """
        Writes a dataflow iteratively to a .txt file. Every datapoint must be a string.
        As the length of the dataflow cannot be determined in every case max_datapoint prevents generating an
        unexpectedly large file

        Args:
            df: The dataflow to write from.
            path: The path, the .txt file to write to.
            file_name: Name of the target file.
            max_datapoints: Maximum number of datapoint to consider writing to a file.
        """

        if not os.path.isdir(path):
            raise NotADirectoryError(path)
        if not is_file_extension(file_name, ".jsonl"):
            raise FileExtensionError(f"Expected .txt file got {path}")

        with open(os.path.join(path, file_name), "w", encoding="UTF-8") as file:
            for k, dp in enumerate(df):
                if max_datapoints is None:
                    file.write(dp)
                elif k < max_datapoints:
                    file.write(dp)
                else:
                    break


class SerializerFiles:
    """
    Serialize files from a directory and all subdirectories. Only one file type can be serialized. Once specified, all
    other types will be filtered out.

    Example:
        ```python
        df = SerializerFiles.load("path/to/dir",file_type=".pdf")

        will yield absolute paths to all `.pdf` files in the directory and all subdirectories.
        ```
    """

    @staticmethod
    def load(
        path: PathLikeOrStr,
        file_type: Union[str, Sequence[str]],
        max_datapoints: Optional[int] = None,
        shuffle: Optional[bool] = False,
        sort: Optional[bool] = True,
    ) -> DataFlow:
        """
        Generates a dataflow where a datapoint consists of a string of names of files with respect to some file type.
        If you want to load the files you need to do this in a following step by yourself.

        Args:
            path: A path to some base directory. Will inspect all subdirectories, as well
            file_type: A file type (suffix) to look out for (single str or list of stings)
            max_datapoints: Stop iteration after passing max_datapoints
            shuffle: Shuffle the files, so that the order of appearance in dataflow is random.
            sort: If set to `True` it will sort all selected files by its string

        Returns:
            Dataflow to iterate from
        """
        df: DataFlow
        df1: DataFlow
        df2: DataFlow
        df3: DataFlow

        path = Path(path)
        if not path.exists():
            raise NotADirectoryError(f"The path {path} to the directory or file does not exist")

        if shuffle:
            sort = False
        it1 = os.walk(os.fspath(path), topdown=False)
        it2 = os.walk(os.fspath(path), topdown=False)
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
        raise NotImplementedError()


class CocoParser:
    """
    A simplified version of the COCO helper class for reading  annotations. It currently supports only
    bounding box annotations

    Args:
        annotation_file: Location of annotation file
    """

    def __init__(self, annotation_file: Optional[PathLikeOrStr] = None) -> None:
        self.dataset: JsonDict = {}
        self.anns: Dict[int, JsonDict] = {}
        self.cats: Dict[int, JsonDict] = {}
        self.imgs: Dict[int, JsonDict] = {}

        self.img_to_anns: DefaultDict[int, List[int]] = defaultdict(list)
        self.cat_to_imgs: DefaultDict[int, List[int]] = defaultdict(list)

        if annotation_file is not None:
            with timed_operation(message="Loading annotations to memory"):
                with open(annotation_file, "r", encoding="UTF-8") as file:
                    dataset = json.load(file)
                if not isinstance(dataset, dict):
                    raise TypeError(f"Annotation file format {type(dataset)} for {annotation_file} not supported")
            self.dataset = dataset
            self._create_index()

    def _create_index(self) -> None:
        with timed_operation(message="creating index"):
            anns, cats, imgs = {}, {}, {}
            img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
            if "annotations" in self.dataset:
                for ann in self.dataset["annotations"]:
                    img_to_anns[ann["image_id"]].append(ann)
                    anns[ann["id"]] = ann

            if "images" in self.dataset:
                for img in self.dataset["images"]:
                    imgs[img["id"]] = img

            if "categories" in self.dataset:
                for cat in self.dataset["categories"]:
                    cats[cat["id"]] = cat

            if "annotations" in self.dataset and "categories" in self.dataset:
                for ann in self.dataset["annotations"]:
                    cat_to_imgs[ann["category_id"]].append(ann["image_id"])

            self.anns = anns
            self.img_to_anns = img_to_anns
            self.cat_to_imgs = cat_to_imgs
            self.imgs = imgs
            self.cats = cats

    def info(self) -> None:
        """
        Print information about the annotation file.
        """
        rows = []
        for key, value in self.dataset["info"].items():
            row = [key, value]
            rows.append(row)

        header = ["key", "value"]
        table = tabulate(rows, headers=header, tablefmt="fancy_grid", stralign="left", numalign="left")
        print(colored(table, "cyan"))

    def get_ann_ids(
        self,
        img_ids: Optional[Union[int, Sequence[int]]] = None,
        cat_ids: Optional[Union[int, Sequence[int]]] = None,
        area_range: Optional[Sequence[int]] = None,
        is_crowd: Optional[bool] = None,
    ) -> Sequence[int]:
        """
        Get annotation ids that satisfy given filter conditions. default skips that filter

        Args:
            img_ids: get anns for given imgs
            cat_ids: get anns for given cats
            area_range: get anns for given area range (e.g. [0 inf])
            is_crowd: get anns for given crowd label (False or True)

        Returns:
            ids: integer array of ann ids
        """

        if img_ids is None:
            img_ids = []
        if cat_ids is None:
            cat_ids = []
        if area_range is None:
            area_range = []

        img_ids = [img_ids] if isinstance(img_ids, int) else img_ids
        cat_ids = [cat_ids] if isinstance(cat_ids, int) else cat_ids

        if len(img_ids) == len(cat_ids) == len(area_range) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(img_ids) == 0:
                lists = [self.img_to_anns[img_id] for img_id in img_ids if img_id in self.img_to_anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(cat_ids) == 0 else [ann for ann in anns if ann["category_id"] in cat_ids]
            anns = (
                anns if len(area_range) == 0 else [ann for ann in anns if area_range[0] < ann["area"] < area_range[1]]
            )
        if is_crowd is not None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == is_crowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def get_cat_ids(
        self,
        category_names: Optional[Union[str, Sequence[str]]] = None,
        super_category_names: Optional[Union[str, Sequence[str]]] = None,
        category_ids: Optional[Union[int, Sequence[int]]] = None,
    ) -> Sequence[int]:
        """
        Filtering parameters. Default does not filter anything.

        Args:
            category_names: get cats for given cat names
            super_category_names: get cats for given super category names
            category_ids: get cats for given cat ids

        Returns:
            ids: integer array of cat ids
        """

        if category_names is None:
            category_names = []
        if super_category_names is None:
            super_category_names = []
        if category_ids is None:
            category_ids = []

        category_names = [category_names] if isinstance(category_names, str) else category_names
        super_category_names = [super_category_names] if isinstance(super_category_names, str) else super_category_names
        category_ids = [category_ids] if isinstance(category_ids, int) else category_ids

        if len(category_names) == len(super_category_names) == len(category_ids) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]
            cats = cats if len(category_names) == 0 else [cat for cat in cats if cat["name"] in category_names]
            cats = (
                cats
                if len(super_category_names) == 0
                else [cat for cat in cats if cat["supercategory"] in super_category_names]
            )
            cats = cats if len(category_ids) == 0 else [cat for cat in cats if cat["id"] in category_ids]
        ids = [cat["id"] for cat in cats]
        return ids

    def get_image_ids(
        self, img_ids: Optional[Union[int, Sequence[int]]] = None, cat_ids: Optional[Union[int, Sequence[int]]] = None
    ) -> Sequence[int]:
        """
        Get image ids that satisfy given filter conditions.

        Args:
            img_ids: get imgs for given ids
            cat_ids: get imgs with all given cats

        Returns:
            ids: integer array of img ids
        """

        if img_ids is None:
            img_ids = []
        if cat_ids is None:
            cat_ids = []

        img_ids = [img_ids] if isinstance(img_ids, int) else img_ids
        cat_ids = [cat_ids] if isinstance(cat_ids, int) else cat_ids

        if len(img_ids) == len(cat_ids) == 0:
            ids = set(self.imgs.keys())
        else:
            ids = set(img_ids)
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_imgs[cat_id])
                else:
                    ids &= set(self.cat_to_imgs[cat_id])
        return list(ids)

    def load_anns(self, ids: Optional[Union[int, Sequence[int]]] = None) -> List[JsonDict]:
        """
        Load anns with the specified ids.

        Args:
            ids: integer ids specifying anns

        Returns:
            anns: loaded ann objects
        """
        if ids is None:
            ids = []
        ids = [ids] if isinstance(ids, int) else ids

        return [self.anns[id] for id in ids]

    def load_cats(self, ids: Optional[Union[int, Sequence[int]]] = None) -> List[JsonDict]:
        """
        Load cats with the specified ids.

        Args:
            ids: integer ids specifying cats

        Returns:
            cats: loaded cat objects
        """
        if ids is None:
            ids = []
        ids = [ids] if isinstance(ids, int) else ids

        return [self.cats[idx] for idx in ids]

    def load_imgs(self, ids: Optional[Union[int, Sequence[int]]] = None) -> List[JsonDict]:
        """
        Load anns with the specified ids.

        Args:
            ids: integer ids specifying img

        Returns:
            imgs: loaded img objects
        """
        if ids is None:
            ids = []
        ids = [ids] if isinstance(ids, int) else ids

        return [self.imgs[idx] for idx in ids]


class SerializerCoco:
    """
    Class for serializing annotation files in COCO format. COCO comes in `JSON` format which is a priori not
    serialized. This class implements only the very basic methods to generate a dataflow. It wraps the coco class
    from `pycocotools` and assembles annotations that belong to the image.

    Note:
        Conversion into the core `Image` has to be done by yourself.

    Example:
        ```python
        df = SerializerCoco.load("path/to/annotations.json")
        df.reset_state()
        for dp in df:
            # {'image':{'id',...},'annotations':[{'id':…,'bbox':...}]}
        ```

    """

    @staticmethod
    def load(path: PathLikeOrStr, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Loads a `.json` file and generates a dataflow.

        Args:
            max_datapoints: Will stop the iteration once max_datapoints have been streamed.
            path: a path to a .json file.

        Returns:
            dataflow to iterate from
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        file = os.path.split(path)[1]
        if not is_file_extension(file, ".json"):
            raise FileExtensionError(f"Expected .json file got {path}")

        with timed_operation("Start loading .json file and serializing"):
            coco = CocoParser(path)
            img_ids = coco.get_image_ids()
            imgs = coco.load_imgs(img_ids)

            with get_tqdm(total=len(imgs)) as status_bar:
                for img in imgs:
                    img["annotations"] = coco.img_to_anns[img["id"]]
                    status_bar.update()

        df = CustomDataFromList(imgs, max_datapoints=max_datapoints)
        return df

    @staticmethod
    def save() -> None:
        """
        Not implemented
        """
        raise NotImplementedError()


class SerializerPdfDoc:
    """
    Serialize a pdf document with an arbitrary number of pages.

    Example:
        ```python
        df = SerializerPdfDoc.load("path/to/document.pdf")

        will yield datapoints:

        {"path": "path/to/document.pdf", "file_name" document_page_1.pdf, "pdf_bytes": b"some-bytes"}
        ```

    """

    @staticmethod
    def load(path: PathLikeOrStr, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Loads the document page wise and returns a dataflow accordingly.

        Args:
            path: Path to the pdf document.
            max_datapoints: The maximum number of pages to stream.

        Returns:
            A dict with structure `{"path":... ,"file_name": ..., "pdf_bytes": ...}`. The file name is a
            concatenation of the physical file name and the current page number.
        """

        file_name = os.path.split(path)[1]
        prefix, suffix = os.path.splitext(file_name)
        df: DataFlow
        df = CustomDataFromIterable(PDFStreamer(path_or_bytes=path), max_datapoints=max_datapoints)
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
        return df

    @staticmethod
    def save(path: PathLikeOrStr) -> None:
        """
        Not implemented
        """
        raise NotImplementedError()

    @staticmethod
    def split(
        path: PathLikeOrStr, path_target: Optional[PathLikeOrStr] = None, max_datapoint: Optional[int] = None
    ) -> None:
        """
        Split a document into single pages.
        """
        if path_target is None:
            path_target, _ = os.path.split(path)
        if not os.path.isdir(path_target):
            raise NotADirectoryError(path)
        df = SerializerPdfDoc.load(path, max_datapoint)
        for dp in df:
            with open(os.path.join(path_target, dp["file_name"]), "wb") as page:
                page.write(dp["pdf_bytes"])
