# -*- coding: utf-8 -*-
# File: recipes.py

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
Recipes for the annotation tool prodigy by Explosion.
"""

import sys
import itertools

from typing import Optional, Union, List, Dict, Any

from deep_doctection.utils.fs import maybe_path_or_pdf
from deep_doctection.utils.settings import names
from deep_doctection.utils.utils import to_bool
from deep_doctection.datasets.registry import DatasetRegistry
from deep_doctection.datasets.info import DatasetCategories
from deep_doctection.utils import string_to_dict, split_string
from deep_doctection.mapper.cats import filter_cat
from deep_doctection.mapper.cats import cat_to_sub_cat as cat_to_sub_cat_func
from deep_doctection.mapper.prodigystruct import image_to_prodigy
from deep_doctection.dataflow import MapData  # type: ignore
from deep_doctection.analyzer.dd import get_dd_analyzer

try:
    from prodigy import recipe  # type: ignore
except ImportError:
    print("Prodigy package is licensed: https://prodi.gy/buy . Please install separately.")
    sys.exit(1)


def _no_progress_bar(ctrl, update_return_value) -> None:  # type: ignore  # pylint: disable=W0613
    return None  # will otherwise raise an error for dataflows without a length


@recipe(
    "view_dataset",
    dataset_name=("A dataset with images and image annotations", "positional", None, str),
    split=("The split of a dataset. In most cases, if any: train, val or test", "option", "s", str),
    build_mode=(
        "Datasets can be displayed in different build modes. Annotated objects can be cropped or different "
        "annotation stages can be displayed. ",
        "option",
        "b",
        str,
    ),
    dump_name=("The dataset name in Prodigy-db. Will default to the original name", "option", "d", str),
    filter_categories=(
        "The categories to display. Will filter out everything else. Will default to display all cats.",
        "option",
        "c",
        split_string,
    ),
    label_categories=(
        "Label new categories that are not part of the dataset. This will display these "
        "categories on top of the available categories",
        "option",
        "n",
        split_string,
    ),
    cat_to_sub_cat=("Display categories with sub category label. E.G 'CAT=SUB_CAT'", "option", "r", string_to_dict),
)
def dataset_detection_recipe(
    dataset_name: str,
    split: str = "val",
    build_mode: str = "",
    dump_name: Optional[str] = None,
    filter_categories: Optional[Union[str, List[str]]] = None,
    label_categories: Optional[Union[str, List[str]]] = None,
    cat_to_sub_cat: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    The recipe for displaying annotation of various datasets. Check https://prodi.gy/docs/custom-recipes for further
    info.

    :param dataset_name: Dataset with images and image annotations
    :param split: Split(s) of a dataset. In most cases, if any: train, val or test
    :param build_mode: As datasets can be in different build modes one can optionally add
                       a specific build mode.
    :param dump_name: The name of the dataset the annotation results are saved in Prodigy-db. Will default to the
                      original dataset. If there is no such bucket in db it will create one.
    :param filter_categories: Categories that should be displayed. Will filter out everything else. If None, everything
                              will be displayed.
    :param label_categories: Label new categories that are currently not part of the dataset.
    :param cat_to_sub_cat: Display categories with sub category label. E.G 'cat=sub_cat'

    :return: Dict of prodigy inputs
    """

    if not dump_name:
        dump_name = dataset_name

    dataset = DatasetRegistry.get_dataset(dataset_name)

    if isinstance(dataset.dataflow.categories, DatasetCategories):
        if cat_to_sub_cat:
            dataset.dataflow.categories.set_cat_to_sub_cat(cat_to_sub_cat)

        if filter_categories:
            dataset.dataflow.categories.filter_categories(filter_categories)

        category_list = dataset.dataflow.categories.get_categories(as_dict=False, name_as_key=True, filtered=True)

    else:
        category_list = []

    if label_categories:
        if isinstance(label_categories, str):
            label_categories = [label_categories]
        category_list.extend(label_categories)  # type: ignore

    df = dataset.dataflow.build(split=split, build_mode=build_mode, load_image=True)
    cat_to_sub_cat_mapper = cat_to_sub_cat_func(  # pylint: disable=E1120  # 259  # type: ignore
        dataset.dataflow.categories
    )
    df = MapData(df, cat_to_sub_cat_mapper)
    cats = dataset.dataflow.categories

    df = MapData(
        df,
        filter_cat(  # pylint: disable=E1120
            cats.get_categories(as_dict=False, filtered=True),  # type: ignore
            cats.get_categories(as_dict=False, filtered=False),  # type: ignore
        ),
    )

    df = MapData(df, image_to_prodigy)

    df.reset_state()

    prodigy_inputs = {"dataset": dump_name, "stream": df, "view_id": "image_manual", "progress": _no_progress_bar}

    config = {"labels": category_list, "feed_overlap": False, "image_manual_stroke_width": 2}
    prodigy_inputs["config"] = config

    return prodigy_inputs


@recipe(
    "view_analyzer",
    path=("A path to a pdf document or an image", "positional", None, str),
    dump_name=("A name to dump the results to", "option", "d", str),
    tables=("If True will enable table recognition ", "option", "t", bool),
    words=("If True will enable ocr", "option", "w", bool),
    filter_categories=(
        "The categories to display. Will filter out everything else. Will default to display all cats.",
        "option",
        "c",
        split_string,
    ),
)
def analyzer_detection_recipe(
    path: str,
    dump_name: str = "trash",
    tables: Union[str, bool] = False,
    words: Union[str, bool] = False,
    filter_categories: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    The recipe for displaying annotation of analyzer results. Check https://prodi.gy/docs/custom-recipes for further
    info.

    :param path: Path to a pdf document or an image.
    :param dump_name: The name of the dataset the annotation results are saved in Prodigy-db. Will default to
                      "trash". If there is no such bucket in db it will create one.
    :param  filter_categories: Select category names you want to view and filters out everything else. By default, it
                               will filter nothing.
    :param tables: If True will enable table recognition
    :param words: If True will enable ocr

    :return: Dict of prodigy inputs
    """

    tables = to_bool(tables)
    words = to_bool(words)
    path_type = maybe_path_or_pdf(path)

    if path_type == 1:
        kwargs = {"path": path, "output": "image"}
    elif path_type == 2:
        kwargs = {"doc_path": path, "output": "image"}
    else:
        raise ValueError("not a valid path or a pdf document")

    all_categories = list(
        itertools.chain(*[attr[1].to_dict().values() for attr in vars(names).items() if attr[0] == "C"])
    )

    if filter_categories is None:
        filter_categories = all_categories

    analyzer = get_dd_analyzer(tables=tables, ocr=words, table_refinement=False)
    df = analyzer.analyze(**kwargs)
    df = MapData(df, filter_cat(filter_categories, all_categories))  # type: ignore # pylint: disable=E1120
    df = MapData(df, image_to_prodigy)

    prodigy_inputs = {"dataset": dump_name, "stream": df, "view_id": "image_manual", "progress": _no_progress_bar}
    config = {"labels": filter_categories, "feed_overlap": True, "image_manual_stroke_width": 2}
    prodigy_inputs["config"] = config

    return prodigy_inputs
