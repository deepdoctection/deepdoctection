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

import itertools
import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Union

from jsonlines import Reader, Writer

from ..utils.context import timed_operation
from ..utils.detection_types import JsonDict, Pathlike
from ..utils.pdf_utils import PDFStreamer
from ..utils.tqdm import get_tqdm
from ..utils.utils import FileExtensionError, is_file_extension
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


class SerializerJsonlines:
    """
    Serialize a dataflow from a jsonlines file. Alternatively, save a dataflow of JSON objects to a .jsonl file.

    **Example:**

        .. code-block:: python

            df = SerializerJsonlines.load("path/to/file.jsonl")
            df.reset_state()
            for dp in df:
               ... # is a dict
    """

    @staticmethod
    def load(path: Pathlike, max_datapoints: Optional[int] = None) -> CustomDataFromIterable:
        """
        :param path: a path to a .jsonl file.
        :param max_datapoints: Will stop the iteration once max_datapoints have been streamed

        :return: dataflow to iterate from
        """
        file = open(path, "r")  # pylint: disable=W1514,R1732
        iterator = Reader(file)
        return CustomDataFromIterable(iterator, max_datapoints=max_datapoints)

    @staticmethod
    def save(df: DataFlow, path: Pathlike, file_name: str, max_datapoints: Optional[int] = None) -> None:
        """
        Writes a dataflow iteratively to a .jsonl file. Every datapoint must be a dict where all items are serializable.
        As the length of the dataflow cannot be determined in every case max_datapoint prevents generating an
        unexpectedly large file

        :param df: The dataflow to write from.
        :param path: The path, the .jsonl file to write to.
        :param file_name: name of the target file.
        :param max_datapoints: maximum number of datapoint to consider writing to a file.
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
    to a .txt file.

    **Example**:

        .. code-block:: python

            df = SerializerTabsepFiles.load("path/to/file.txt")

        will yield each text line of the file.
    """

    @staticmethod
    def load(path: Pathlike, max_datapoins: Optional[int] = None) -> CustomDataFromList:
        """
        :param path: a path to a .txt file.
        :param max_datapoins: Will stop the iteration once max_datapoints have been streamed

        :return: dataflow to iterate from
        """

        with open(path, "r", encoding="UTF-8") as file:
            file_list = file.readlines()
        return CustomDataFromList(file_list, max_datapoints=max_datapoins)

    @staticmethod
    def save(df: DataFlow, path: Pathlike, file_name: str, max_datapoints: Optional[int] = None) -> None:
        """
        Writes a dataflow iteratively to a .txt file. Every datapoint must be a string.
        As the length of the dataflow cannot be determined in every case max_datapoint prevents generating an
        unexpectedly large file

        :param df: The dataflow to write from.
        :param path: The path, the .txt file to write to.
        :param file_name: name of the target file.
        :param max_datapoints: maximum number of datapoint to consider writing to a file.
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
    """

    @staticmethod
    def load(
        path: Pathlike,
        file_type: Union[str, Sequence[str]],
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
        df: DataFlow
        df1: DataFlow
        df2: DataFlow
        df3: DataFlow

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


class CocoParser:
    """
    A simplified version of the Microsoft COCO helper class for reading  annotations. It currently supports only
    bounding box annotations

    :param annotation_file (str): location of annotation file
    :param image_folder (str): location to the folder that hosts images.
    """

    def __init__(self, annotation_file: Optional[Pathlike] = None) -> None:

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
        for key, value in self.dataset["info"].items():
            print(f"{key}: {value}")

    def get_ann_ids(
        self,
        img_ids: Optional[Union[int, Sequence[int]]] = None,
        cat_ids: Optional[Union[int, Sequence[int]]] = None,
        area_range: Optional[Sequence[int]] = None,
        is_crowd: Optional[bool] = None,
    ) -> Sequence[int]:
        """
        Get ann ids that satisfy given filter conditions. default skips that filter

        :param img_ids: get anns for given imgs
        :param cat_ids: get anns for given cats
        :param area_range: get anns for given area range (e.g. [0 inf])
        :param is_crowd: get anns for given crowd label (False or True)

        :return: ids: integer array of ann ids
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
        Filtering parameters. default skips that filter.

        :param category_names: get cats for given cat names
        :param super_category_names: get cats for given super category names
        :param category_ids: get cats for given cat ids

        :return: ids: integer array of cat ids
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
        Get img ids that satisfy given filter conditions.

        :param img_ids: get imgs for given ids
        :param cat_ids: get imgs with all given cats

        :return: ids: integer array of img ids
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

        :param ids: integer ids specifying anns

        :return: anns: loaded ann objects
        """
        if ids is None:
            ids = []
        ids = [ids] if isinstance(ids, int) else ids

        return [self.anns[id] for id in ids]

    def load_cats(self, ids: Optional[Union[int, Sequence[int]]] = None) -> List[JsonDict]:
        """
        Load cats with the specified ids.

        :param ids: integer ids specifying cats

        :return: cats: loaded cat objects
        """
        if ids is None:
            ids = []
        ids = [ids] if isinstance(ids, int) else ids

        return [self.cats[idx] for idx in ids]

    def load_imgs(self, ids: Optional[Union[int, Sequence[int]]] = None) -> List[JsonDict]:
        """
        Load anns with the specified ids.

        :param ids: integer ids specifying img

        :return: imgs: loaded img objects
        """
        if ids is None:
            ids = []
        ids = [ids] if isinstance(ids, int) else ids

        return [self.imgs[idx] for idx in ids]


class SerializerCoco:
    """
    Class for serializing annotation files in Coco format. Coco comes in JSON format which is a priori not
    serialized. This class implements only the very basic methods to generate a dataflow. It wraps the coco class
    from pycocotools and assembles annotations that belong to the image. Note, that the conversion into the core
    :class:`Image` has to be done by yourself.
    """

    @staticmethod
    def load(path: Pathlike, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Loads a .json file and generates a dataflow.

        **Example:**

            .. code-block:: python

                {'images':[img1,img2,...], 'annotations':[ann1,ann2,...],...}

            it will generate a dataflow with datapoints


            .. code-block:: python

                {'image':{'id',...},'annotations':[{'id':…,'bbox':...}]}

            for each single image id.

        :param max_datapoints: Will stop the iteration once max_datapoints have been streamed.
        :param path: a path to a .json file.
        :return: dataflow to iterate from
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
        raise NotImplementedError


class SerializerPdfDoc:
    """
    Serialize a pdf document with an arbitrary number of pages.

    **Example:**

        .. code-block:: python

            df = SerializerPdfDoc.load("path/to/document.pdf")

        will yield datapoints:

        .. code-block:: python

            {"path": "path/to/document.pdf", "file_name" document_page_1.pdf, "pdf_bytes": b"some-bytes"}
    """

    @staticmethod
    def load(path: Pathlike, max_datapoints: Optional[int] = None) -> DataFlow:
        """
        Loads the document page wise and returns a dataflow accordingly.

        :param path: Path to the pdf document.
        :param max_datapoints: The maximum number of pages to stream.
        :return: A dict with structure {"path":... ,"file_name": ..., "pdf_bytes": ...}. The file name is a
                 concatenation of the physical file name and the current page number.
        """

        file_name = os.path.split(path)[1]
        prefix, suffix = os.path.splitext(file_name)
        df: DataFlow
        df = CustomDataFromIterable(PDFStreamer(path=path), max_datapoints=max_datapoints)
        df = MapData(df, lambda dp: {"path": path, "file_name": prefix + f"_{dp[1]}" + suffix, "pdf_bytes": dp[0]})
        return df

    @staticmethod
    def save(path: Pathlike) -> None:
        """
        Not implemented
        """
        raise NotImplementedError

    @staticmethod
    def split(path: Pathlike, path_target: Optional[Pathlike] = None, max_datapoint: Optional[int] = None) -> None:
        """
        Split a Document into single pages.
        """
        if path_target is None:
            path_target, _ = os.path.split(path)
        if not os.path.isdir(path_target):
            raise NotADirectoryError(path)
        df = SerializerPdfDoc.load(path, max_datapoint)
        for dp in df:
            with open(os.path.join(path_target, dp["file_name"]), "wb") as page:
                page.write(dp["pdf_bytes"])
