# -*- coding: utf-8 -*-
# File: model.py

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
Module for ModelCatalog and ModelDownloadManager
"""

import os
from copy import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

from huggingface_hub import cached_download, hf_hub_url  # type: ignore
from tabulate import tabulate
from termcolor import colored

from ..utils.fs import download
from ..utils.logger import log_once, logger
from ..utils.settings import Languages, LayoutType, ObjectTypes
from ..utils.systools import get_configs_dir_path, get_weights_dir_path

__all__ = ["ModelCatalog", "ModelDownloadManager", "print_model_infos", "ModelProfile"]


@dataclass
class ModelProfile:
    """
    Class for model profile. Add for each model one ModelProfile to the ModelCatalog
    """

    name: str
    description: str

    size: List[int]
    tp_model: bool = field(default=False)
    config: Optional[str] = field(default=None)
    hf_repo_id: Optional[str] = field(default=None)
    hf_model_name: Optional[str] = field(default=None)
    hf_config_file: Optional[List[str]] = field(default=None)
    urls: Optional[List[str]] = field(default=None)
    categories: Optional[Dict[str, ObjectTypes]] = field(default=None)
    model_wrapper: Optional[str] = field(default=None)

    def as_dict(self) -> Dict[str, Any]:
        """
        returns a dict of the dataclass
        """
        return asdict(self)


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well.

    To get an overview of all registered models

    .. code-block:: python

        print(ModelCatalog.get_model_list())

    To get a model card for some specific model:

    .. code-block:: python

        profile = ModelCatalog.get_profile("layout/model-800000_inf_only.data-00000-of-00001")
        print(profile.description)

    Some models will have their weights and configs stored in the cache. To instantiate predictors one will sometimes
    need their path. Use

    .. code-block:: python

        path_weights = ModelCatalog.get_full_path_configs("layout/model-800000_inf_only.data-00000-of-00001")
        path_configs = ModelCatalog.get_full_path_weights("layout/model-800000_inf_only.data-00000-of-00001")

    To register a new model

    .. code-block:: python

        ModelCatalog.get_full_path_configs("my_new_model")

    """

    CATALOG: Dict[str, ModelProfile] = {
        "layout/model-800000_inf_only.data-00000-of-00001": ModelProfile(
            name="layout/model-800000_inf_only.data-00000-of-00001",
            description="Tensorpack layout model for inference purposes trained on Publaynet",
            config="dd/tp/conf_frcnn_layout.yaml",
            size=[274552244, 7907],
            tp_model=True,
            hf_repo_id="deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            hf_model_name="model-800000_inf_only",
            hf_config_file=["conf_frcnn_layout.yaml"],
            categories={
                "1": LayoutType.text,
                "2": LayoutType.title,
                "3": LayoutType.list,
                "4": LayoutType.table,
                "5": LayoutType.figure,
            },
            model_wrapper="TPFrcnnDetector",
        ),
        "cell/model-1800000_inf_only.data-00000-of-00001": ModelProfile(
            name="cell/model-1800000_inf_only.data-00000-of-00001",
            description="Tensorpack cell detection model for inference purposes trained on Pubtabnet",
            config="dd/tp/conf_frcnn_cell.yaml",
            size=[274503056, 8056],
            tp_model=True,
            hf_repo_id="deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            hf_model_name="model-1800000_inf_only",
            hf_config_file=["conf_frcnn_cell.yaml"],
            categories={"1": LayoutType.cell},
            model_wrapper="TPFrcnnDetector",
        ),
        "item/model-1620000_inf_only.data-00000-of-00001": ModelProfile(
            name="item/model-1620000_inf_only.data-00000-of-00001",
            description="Tensorpack row/column detection model for inference purposes trained on Pubtabnet",
            config="dd/tp/conf_frcnn_rows.yaml",
            size=[274515344, 7904],
            tp_model=True,
            hf_repo_id="deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            hf_model_name="model-1620000_inf_only",
            hf_config_file=["conf_frcnn_rows.yaml"],
            categories={"1": LayoutType.row, "2": LayoutType.column},
            model_wrapper="TPFrcnnDetector",
        ),
        "item/model-1620000.data-00000-of-00001": ModelProfile(
            name="item/model-1620000.data-00000-of-00001",
            description="Tensorpack row/column detection model trained on Pubtabnet",
            config="dd/tp/conf_frcnn_rows.yaml",
            size=[823546048, 25787],
            tp_model=True,
            hf_repo_id="deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc",
            hf_model_name="model-1620000",
            hf_config_file=["conf_frcnn_rows.yaml"],
            categories={"1": LayoutType.row, "2": LayoutType.column},
            model_wrapper="TPFrcnnDetector",
        ),
        "layout/model-800000.data-00000-of-00001": ModelProfile(
            name="layout/model-800000.data-00000-of-00001",
            description="Tensorpack layout detection model trained on Publaynet",
            config="dd/tp/conf_frcnn_layout.yaml",
            size=[823656748, 25796],
            tp_model=True,
            hf_repo_id="deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet",
            hf_model_name="model-800000",
            hf_config_file=["conf_frcnn_layout.yaml"],
            categories={
                "1": LayoutType.text,
                "2": LayoutType.title,
                "3": LayoutType.list,
                "4": LayoutType.table,
                "5": LayoutType.figure,
            },
            model_wrapper="TPFrcnnDetector",
        ),
        "cell/model-1800000.data-00000-of-00001": ModelProfile(
            name="cell/model-1800000.data-00000-of-00001",
            description="Tensorpack cell detection model trained on Pubtabnet",
            config="dd/tp/conf_frcnn_cell.yaml",
            size=[823509160, 25905],
            tp_model=True,
            hf_repo_id="deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c",
            hf_model_name="model-1800000",
            hf_config_file=["conf_frcnn_cell.yaml"],
            categories={"1": LayoutType.cell},
            model_wrapper="TPFrcnnDetector",
        ),
        "layout/d2_model-800000-layout.pkl": ModelProfile(
            name="layout/d2_model-800000-layout.pkl",
            description="Detectron2 layout detection model trained on Publaynet",
            config="dd/d2/layout/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[274568239],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            hf_model_name="d2_model-800000-layout.pkl",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={
                "1": LayoutType.text,
                "2": LayoutType.title,
                "3": LayoutType.list,
                "4": LayoutType.table,
                "5": LayoutType.figure,
            },
            model_wrapper="D2FrcnnDetector",
        ),
        "layout/d2_model_0829999_layout_inf_only.pt": ModelProfile(
            name="layout/d2_model_0829999_layout_inf_only.pt",
            description="Detectron2 layout detection model trained on Publaynet",
            config="dd/d2/layout/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[274632215],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            hf_model_name="d2_model_0829999_layout_inf_only.pt",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={
                "1": LayoutType.text,
                "2": LayoutType.title,
                "3": LayoutType.list,
                "4": LayoutType.table,
                "5": LayoutType.figure,
            },
            model_wrapper="D2FrcnnDetector",
        ),
        "layout/d2_model_0829999_layout.pth": ModelProfile(
            name="layout/d2_model_0829999_layout.pth",
            description="Detectron2 layout detection model trained on Publaynet. Checkpoint for resuming training",
            config="dd/d2/layout/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[548377327],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            hf_model_name="d2_model_0829999_layout.pth",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={
                "1": LayoutType.text,
                "2": LayoutType.title,
                "3": LayoutType.list,
                "4": LayoutType.table,
                "5": LayoutType.figure,
            },
            model_wrapper="D2FrcnnDetector",
        ),
        "cell/d2_model-1800000-cell.pkl": ModelProfile(
            name="cell/d2_model-1800000-cell.pkl",
            description="Detectron2 cell detection inference only model trained on Pubtabnet",
            config="dd/d2/cell/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[274519039],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            hf_model_name="d2_model-1800000-cell.pkl",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={"1": LayoutType.cell},
            model_wrapper="D2FrcnnDetector",
        ),
        "cell/d2_model_1849999_cell_inf_only.pt": ModelProfile(
            name="cell/d2_model_1849999_cell_inf_only.pt",
            description="Detectron2 cell detection inference only model trained on Pubtabnet",
            config="dd/d2/cell/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[274583063],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            hf_model_name="d2_model_1849999_cell_inf_only.pt",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={"1": LayoutType.cell},
            model_wrapper="D2FrcnnDetector",
        ),
        "cell/d2_model_1849999_cell.pth": ModelProfile(
            name="cell/d2_model_1849999_cell.pth",
            description="Detectron2 cell detection inference only model trained on Pubtabnet",
            config="dd/d2/cell/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[548279023],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            hf_model_name="cell/d2_model_1849999_cell.pth",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={"1": LayoutType.cell},
            model_wrapper="D2FrcnnDetector",
        ),
        "item/d2_model-1620000-item.pkl": ModelProfile(
            name="item/d2_model-1620000-item.pkl",
            description="Detectron2 item detection inference only model trained on Pubtabnet",
            config="dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[274531339],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            hf_model_name="d2_model-1620000-item.pkl",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={"1": LayoutType.row, "2": LayoutType.column},
            model_wrapper="D2FrcnnDetector",
        ),
        "item/d2_model_1639999_item.pth": ModelProfile(
            name="item/d2_model_1639999_item.pth",
            description="Detectron2 item detection model trained on Pubtabnet",
            config="dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[548303599],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            hf_model_name="d2_model_1639999_item.pth",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={"1": LayoutType.row, "2": LayoutType.column},
            model_wrapper="D2FrcnnDetector",
        ),
        "item/d2_model_1639999_item_inf_only.pt": ModelProfile(
            name="item/d2_model_1639999_item_inf_only.pt",
            description="Detectron2 item detection model inference only trained on Pubtabnet",
            config="dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml",
            size=[274595351],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            hf_model_name="d2_model_1639999_item_inf_only.pt",
            hf_config_file=["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            categories={"1": LayoutType.row, "2": LayoutType.column},
            model_wrapper="D2FrcnnDetector",
        ),
        "microsoft/layoutlm-base-uncased/pytorch_model.bin": ModelProfile(
            name="microsoft/layoutlm-base-uncased/pytorch_model.bin",
            description="LayoutLM is a simple but effective pre-training method of text and layout for document image"
            " understanding and information extraction tasks, such as form understanding and receipt"
            " understanding. LayoutLM archived the SOTA results on multiple datasets. This model does not"
            "contain any head and has to be fine tuned on a downstream task. This is model has been trained "
            "on 11M documents for 2 epochs.  Configuration: 12-layer, 768-hidden, 12-heads, 113M parameters",
            size=[453093832],
            tp_model=False,
            config="microsoft/layoutlm-base-uncased/config.json",
            hf_repo_id="microsoft/layoutlm-base-uncased",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
        ),
        "microsoft/layoutlm-large-uncased/pytorch_model.bin": ModelProfile(
            name="microsoft/layoutlm-large-uncased/pytorch_model.bin",
            description="LayoutLM is a simple but effective pre-training method of text and layout for document image"
            " understanding and information extraction tasks, such as form understanding and receipt"
            " understanding. LayoutLM archived the SOTA results on multiple datasets. This model does not"
            "contain any head and has to be fine tuned on a downstream task. This is model has been trained"
            " on 11M documents for 2 epochs.  Configuration: 24-layer, 1024-hidden, 16-heads, 343M parameters",
            size=[1361845448],
            tp_model=False,
            config="microsoft/layoutlm-large-uncased/config.json",
            hf_repo_id="microsoft/layoutlm-large-uncased",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
        ),
        "fasttext/lid.176.bin": ModelProfile(
            name="fasttext/lid.176.bin",
            description="Fasttext language detection model",
            size=[131266198],
            urls=["https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"],
            categories={
                "__label__en": Languages.english,
                "__label__ru": Languages.russian,
                "__label__de": Languages.german,
                "__label__fr": Languages.french,
                "__label__it": Languages.italian,
                "__label__ja": Languages.japanese,
                "__label__es": Languages.spanish,
                "__label__ceb": Languages.cebuano,
                "__label__tr": Languages.turkish,
                "__label__pt": Languages.portuguese,
                "__label__uk": Languages.ukrainian,
                "__label__eo": Languages.esperanto,
                "__label__pl": Languages.polish,
                "__label__sv": Languages.swedish,
                "__label__nl": Languages.dutch,
                "__label__he": Languages.hebrew,
                "__label__zh": Languages.chinese,
                "__label__hu": Languages.hungarian,
                "__label__ar": Languages.arabic,
                "__label__ca": Languages.catalan,
                "__label__fi": Languages.finnish,
                "__label__cs": Languages.czech,
                "__label__fa": Languages.persian,
                "__label__sr": Languages.serbian,
                "__label__el": Languages.greek,
                "__label__vi": Languages.vietnamese,
                "__label__bg": Languages.bulgarian,
                "__label__ko": Languages.korean,
                "__label__no": Languages.norwegian,
                "__label__mk": Languages.macedonian,
                "__label__ro": Languages.romanian,
                "__label__id": Languages.indonesian,
                "__label__th": Languages.thai,
                "__label__hy": Languages.armenian,
                "__label__da": Languages.danish,
                "__label__ta": Languages.tamil,
                "__label__hi": Languages.hindi,
                "__label__hr": Languages.croatian,
                "__label__sh": Languages.not_defined,
                "__label__be": Languages.belarusian,
                "__label__ka": Languages.georgian,
                "__label__te": Languages.telugu,
                "__label__kk": Languages.kazakh,
                "__label__war": Languages.waray,
                "__label__lt": Languages.lithuanian,
                "__label__gl": Languages.scottish,
                "__label__sk": Languages.slovak,
                "__label__bn": Languages.benin,
                "__label__eu": Languages.basque,
                "__label__sl": Languages.slovenian,
                "__label__kn": Languages.not_defined,
                "__label__ml": Languages.malayalam,
                "__label__mr": Languages.marathi,
                "__label__et": Languages.estonian,
                "__label__az": Languages.azerbaijani,
                "__label__ms": Languages.not_defined,
                "__label__sq": Languages.albanian,
                "__label__la": Languages.latin,
                "__label__bs": Languages.bosnian,
                "__label__nn": Languages.norwegian_nynorsk,
                "__label__ur": Languages.urdu,
                "__label__lv": Languages.not_defined,
                "__label__my": Languages.not_defined,
                "__label__tt": Languages.not_defined,
                "__label__af": Languages.not_defined,
                "__label__oc": Languages.not_defined,
                "__label__nds": Languages.not_defined,
                "__label__ky": Languages.not_defined,
                "__label__ast": Languages.not_defined,
                "__label__tl": Languages.not_defined,
                "__label__is": Languages.not_defined,
                "__label__ia": Languages.not_defined,
                "__label__si": Languages.not_defined,
                "__label__gu": Languages.not_defined,
                "__label__km": Languages.not_defined,
                "__label__br": Languages.not_defined,
                "__label__ba": Languages.not_defined,
                "__label__uz": Languages.not_defined,
                "__label__bo": Languages.not_defined,
                "__label__pa": Languages.not_defined,
                "__label__vo": Languages.not_defined,
                "__label__als": Languages.not_defined,
                "__label__ne": Languages.not_defined,
                "__label__cy": Languages.not_defined,
                "__label__jbo": Languages.not_defined,
                "__label__fy": Languages.not_defined,
                "__label__mn": Languages.not_defined,
                "__label__lb": Languages.not_defined,
                "__label__ce": Languages.not_defined,
                "__label__ug": Languages.not_defined,
                "__label__tg": Languages.not_defined,
                "__label__sco": Languages.not_defined,
                "__label__sa": Languages.not_defined,
                "__label__cv": Languages.not_defined,
                "__label__jv": Languages.not_defined,
                "__label__min": Languages.not_defined,
                "__label__io": Languages.not_defined,
                "__label__or": Languages.not_defined,
                "__label__as": Languages.not_defined,
                "__label__new": Languages.not_defined,
                "__label__ga": Languages.not_defined,
                "__label__mg": Languages.not_defined,
                "__label__an": Languages.not_defined,
                "__label__ckb": Languages.not_defined,
                "__label__sw": Languages.not_defined,
                "__label__bar": Languages.not_defined,
                "__label__lmo": Languages.not_defined,
                "__label__yi": Languages.not_defined,
                "__label__arz": Languages.not_defined,
                "__label__mhr": Languages.not_defined,
                "__label__azb": Languages.not_defined,
                "__label__sah": Languages.not_defined,
                "__label__pnb": Languages.not_defined,
                "__label__su": Languages.not_defined,
                "__label__bpy": Languages.not_defined,
                "__label__pms": Languages.not_defined,
                "__label__ilo": Languages.not_defined,
                "__label__wuu": Languages.not_defined,
                "__label__ku": Languages.not_defined,
                "__label__ps": Languages.not_defined,
                "__label__ie": Languages.not_defined,
                "__label__xmf": Languages.not_defined,
                "__label__yue": Languages.not_defined,
                "__label__gom": Languages.not_defined,
                "__label__li": Languages.not_defined,
                "__label__mwl": Languages.not_defined,
                "__label__kw": Languages.not_defined,
                "__label__sd": Languages.not_defined,
                "__label__hsb": Languages.not_defined,
                "__label__scn": Languages.not_defined,
                "__label__gd": Languages.not_defined,
                "__label__pam": Languages.not_defined,
                "__label__bh": Languages.not_defined,
                "__label__mai": Languages.not_defined,
                "__label__vec": Languages.not_defined,
                "__label__mt": Languages.not_defined,
                "__label__dv": Languages.not_defined,
                "__label__wa": Languages.not_defined,
                "__label__mzn": Languages.not_defined,
                "__label__am": Languages.not_defined,
                "__label__qu": Languages.not_defined,
                "__label__eml": Languages.not_defined,
                "__label__cbk": Languages.not_defined,
                "__label__tk": Languages.not_defined,
                "__label__rm": Languages.not_defined,
                "__label__os": Languages.not_defined,
                "__label__vls": Languages.not_defined,
                "__label__yo": Languages.not_defined,
                "__label__lo": Languages.not_defined,
                "__label__lez": Languages.not_defined,
                "__label__so": Languages.not_defined,
                "__label__myv": Languages.not_defined,
                "__label__diq": Languages.not_defined,
                "__label__mrj": Languages.not_defined,
                "__label__dsb": Languages.not_defined,
                "__label__frr": Languages.not_defined,
                "__label__ht": Languages.not_defined,
                "__label__gn": Languages.not_defined,
                "__label__bxr": Languages.not_defined,
                "__label__kv": Languages.not_defined,
                "__label__sc": Languages.not_defined,
                "__label__nah": Languages.not_defined,
                "__label__krc": Languages.not_defined,
                "__label__bcl": Languages.not_defined,
                "__label__nap": Languages.not_defined,
                "__label__gv": Languages.not_defined,
                "__label__av": Languages.not_defined,
                "__label__rue": Languages.not_defined,
                "__label__xal": Languages.not_defined,
                "__label__pfl": Languages.not_defined,
                "__label__dty": Languages.not_defined,
                "__label__hif": Languages.not_defined,
                "__label__co": Languages.not_defined,
                "__label__lrc": Languages.not_defined,
                "__label__vep": Languages.not_defined,
                "__label__tyv": Languages.not_defined,
            },
            model_wrapper="FasttextLangDetector",
        ),
    }

    @staticmethod
    def get_full_path_weights(name: str) -> str:
        """
        Returns the absolute path of weights.

        Note, that weights are sometimes not defined by only one artefact. The returned string will only represent one
        weights artefact.

        :param name: model name
        :return: absolute weight path
        """
        profile = ModelCatalog.get_profile(name)
        if profile.name:
            return os.path.join(get_weights_dir_path(), profile.name)
        log_once(
            f"Model {name} is not registered. Please make sure the weights are available in the weights cache "
            f"directory or the full path you provide is correct"
        )
        if os.path.isfile(name):
            return name
        return os.path.join(get_weights_dir_path(), name)

    @staticmethod
    def get_full_path_configs(name: str) -> str:
        """
        Return the absolute path of configs for some given weights. Alternatively, pass last a path to a config file
        (without the base path to the cache config directory).

        Note, that configs are sometimes not defined by only one file. The returned string will only represent one
        file.

        :param name: model name
        :return: absolute path to the config
        """
        profile = ModelCatalog.get_profile(name)
        if profile.config is not None:
            return os.path.join(get_configs_dir_path(), profile.config)
        return os.path.join(get_configs_dir_path(), name)

    @staticmethod
    def get_model_list() -> List[str]:
        """
        Returns a list of absolute paths of registered models.
        """
        return [os.path.join(get_weights_dir_path(), profile.name) for profile in ModelCatalog.CATALOG.values()]

    @staticmethod
    def is_registered(path_weights: str) -> bool:
        """
        Checks if some weights belong to a registered model

        :param path_weights: relative or absolute path
        :return: True if the weights are registered in :class:`ModelCatalog`
        """
        if (ModelCatalog.get_full_path_weights(path_weights) in ModelCatalog.get_model_list()) or (
            path_weights in ModelCatalog.get_model_list()
        ):
            return True
        return False

    @staticmethod
    def get_profile(name: str) -> ModelProfile:
        """
        Returns the profile of given model name, i.e. the config file, size and urls.

        :param name: model name
        :return: A dict of model/weights profiles
        """
        profile = ModelCatalog.CATALOG.get(name)
        if profile is not None:
            return copy(profile)
        return ModelProfile(name="", description="", size=[0], tp_model=False)

    @staticmethod
    def register(name: str, profile: ModelProfile) -> None:
        """
        Register a model with its profile

        :param name: Name of the model. We use the file name of the model along with its path (starting from the
                     weights .cache dir. e.g. 'my_model/model_123.pkl'.
        :param profile: profile of the model
        """
        if name in ModelCatalog.CATALOG:
            raise KeyError("Model already registered")
        ModelCatalog.CATALOG[name] = profile


def get_tp_weight_names(name: str) -> List[str]:
    """
    Given a path to some model weights it will return all file names according to TP naming convention

    :param name: TP model name
    :return: A list of TP file names
    """
    _, file_name = os.path.split(name)
    prefix, _ = file_name.split(".")
    weight_names = []
    for suffix in ["data-00000-of-00001", "index"]:
        weight_names.append(prefix + "." + suffix)

    return weight_names


def print_model_infos(add_description: bool = True, add_config: bool = True, add_categories: bool = True) -> None:
    """
    Prints a table with all registered model profiles and some of their attributes (name, description, config and
    categories)
    """

    profiles = ModelCatalog.CATALOG.values()
    num_columns = min(6, len(profiles))
    infos = []
    for profile in profiles:
        tbl_input: List[Union[Mapping[str, ObjectTypes], str]] = [profile.name]
        if add_description:
            tbl_input.append(profile.description)
        if add_config:
            tbl_input.append(profile.config if profile.config else "")
        if add_categories:
            tbl_input.append(profile.categories if profile.categories else {})
        infos.append(tbl_input)
    tbl_header = ["name"]
    if add_description:
        tbl_header.append("description")
    if add_config:
        tbl_header.append("config")
    if add_categories:
        tbl_header.append("categories")
    table = tabulate(
        infos,
        headers=tbl_header * (num_columns // 2),
        tablefmt="fancy_grid",
        stralign="left",
        numalign="left",
    )
    print(colored(table, "cyan"))


class ModelDownloadManager:
    """
    Class for organizing downloads of config files and weights from various sources. Internally, it will use model
    profiles to know where things are stored.

    .. code-block:: python

        # if you are not sure about the model name use the ModelCatalog
        ModelDownloadManager.maybe_download_weights_and_configs("layout/model-800000_inf_only.data-00000-of-00001")
    """

    @staticmethod
    def maybe_download_weights_and_configs(name: str) -> str:
        """
        Check if some model is registered. If yes, it will check if their weights
        must be downloaded. Only weights that have not the same expected size will be downloaded again.

        :param name: A path to some model weights
        :return: Absolute path to model weights if model is registered
        """

        absolute_path_weights = ModelCatalog.get_full_path_weights(name)
        file_names: List[str] = []
        if ModelCatalog.is_registered(name):
            profile = ModelCatalog.get_profile(name)
            # there is nothing to download if hf_repo_id or urls is not provided
            if not profile.hf_repo_id and not profile.urls:
                return absolute_path_weights
            # determine the right model name
            if profile.tp_model:
                file_names = get_tp_weight_names(name)
            else:
                model_name = profile.hf_model_name
                if model_name is None:
                    # second try. Check if a url is provided
                    if profile.urls is None:
                        raise ValueError("hf_model_name and urls cannot be both None")
                    for url in profile.urls:
                        file_names.append(url.split("/")[-1])
                else:
                    file_names.append(model_name)
            if profile.hf_repo_id:
                ModelDownloadManager.load_model_from_hf_hub(profile, absolute_path_weights, file_names)
                absolute_path_configs = ModelCatalog.get_full_path_configs(name)
                ModelDownloadManager.load_configs_from_hf_hub(profile, absolute_path_configs)
            else:
                ModelDownloadManager._load_from_gd(profile, absolute_path_weights, file_names)

            return absolute_path_weights

        return absolute_path_weights

    @staticmethod
    def load_model_from_hf_hub(profile: ModelProfile, absolute_path: str, file_names: List[str]) -> None:
        """
        Load a model from the Huggingface hub for a given profile and saves the model at the directory of the given
        path.

        :param profile: Profile according to :func:`ModelCatalog.get_profile(path_weights)`
        :param absolute_path: Absolute path (incl. file name) of target file
        :param file_names: Optionally, replace the file name of the ModelCatalog. This is necessary e.g. for Tensorpack
                           models
        """
        repo_id = profile.hf_repo_id
        if repo_id is None:
            raise ValueError("hf_repo_id cannot be None")
        directory, _ = os.path.split(absolute_path)

        for expect_size, file_name in zip(profile.size, file_names):
            size = ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)
            if expect_size is not None and size != expect_size:
                logger.error(
                    "File downloaded from %s does not match the expected size! You may have downloaded"
                    " a broken file, or the upstream may have modified the file.",
                    repo_id,
                )

    @staticmethod
    def _load_from_gd(profile: ModelProfile, absolute_path: str, file_names: List[str]) -> None:
        if profile.urls is None:
            raise ValueError("urls cannot be None")
        for size, url, file_name in zip(profile.size, profile.urls, file_names):
            directory, _ = os.path.split(absolute_path)
            download(str(url), directory, file_name, int(size))

    @staticmethod
    def load_configs_from_hf_hub(profile: ModelProfile, absolute_path: str) -> None:
        """
        Load config file(s) from the Huggingface hub for a given profile and saves the model at the directory of the
        given path.

        :param profile: Profile according to :func:`ModelCatalog.get_profile(path_weights)`
        :param absolute_path:  Absolute path (incl. file name) of target file
        """

        repo_id = profile.hf_repo_id
        if repo_id is None:
            raise ValueError("hf_repo_id cannot be None")
        directory, _ = os.path.split(absolute_path)
        if not profile.hf_config_file:
            raise ValueError("hf_config_file cannot be None")
        for file_name in profile.hf_config_file:
            ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)

    @staticmethod
    def _load_from_hf_hub(repo_id: str, file_name: str, cache_directory: str, force_download: bool = False) -> int:
        url = hf_hub_url(repo_id=repo_id, filename=file_name)
        use_auth_token = os.environ.get("HF_CREDENTIALS")
        f_path = cached_download(
            url,
            cache_dir=cache_directory,
            force_filename=file_name,
            force_download=force_download,
            use_auth_token=use_auth_token,
        )
        stat_info = os.stat(f_path)
        size = stat_info.st_size
        assert size > 0, f"Downloaded an empty file from {url}!"
        return size
