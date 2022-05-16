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
from typing import Any, Dict, List, Optional

from huggingface_hub import cached_download, hf_hub_url  # type: ignore
from tabulate import tabulate
from termcolor import colored

from ..utils.fs import download
from ..utils.logger import logger
from ..utils.settings import names
from ..utils.systools import get_configs_dir_path, get_weights_dir_path

__all__ = ["ModelCatalog", "ModelDownloadManager", "print_model_infos"]


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
    categories: Optional[Dict[str, str]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """
        returns a dict of the dataclass
        """
        return asdict(self)


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well.
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
            categories={"1": names.C.TEXT, "2": names.C.TITLE, "3": names.C.LIST, "4": names.C.TAB, "5": names.C.FIG},
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
            categories={"1": names.C.CELL},
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
            categories={"1": names.C.ROW, "2": names.C.COL},
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
            categories={"1": names.C.ROW, "2": names.C.COL},
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
            categories={"1": names.C.TEXT, "2": names.C.TITLE, "3": names.C.LIST, "4": names.C.TAB, "5": names.C.FIG},
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
            categories={"1": names.C.CELL},
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
            categories={"1": names.C.TEXT, "2": names.C.TITLE, "3": names.C.LIST, "4": names.C.TAB, "5": names.C.FIG},
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
            categories={"1": names.C.CELL},
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
            categories={"1": names.C.ROW, "2": names.C.COL},
        ),
        "fasttext/lid.176.bin": ModelProfile(
            name="fasttext/lid.176.bin",
            description="Fasttext language detection model",
            size=[131266198],
            urls=["https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"],
            categories={
                "__label__en": names.NLP.LANG.ENG,
                "__label__ru": names.NLP.LANG.RUS,
                "__label__de": names.NLP.LANG.DEU,
                "__label__fr": names.NLP.LANG.FRE,
                "__label__it": names.NLP.LANG.ITA,
                "__label__ja": names.NLP.LANG.JPN,
                "__label__es": names.NLP.LANG.SPA,
                "__label__ceb": names.NLP.LANG.CEB,
                "__label__tr": names.NLP.LANG.TUR,
                "__label__pt": names.NLP.LANG.POR,
                "__label__uk": names.NLP.LANG.UKR,
                "__label__eo": names.NLP.LANG.EPO,
                "__label__pl": names.NLP.LANG.POL,
                "__label__sv": names.NLP.LANG.SWE,
                "__label__nl": names.NLP.LANG.DUT,
                "__label__he": names.NLP.LANG.HEB,
                "__label__zh": names.NLP.LANG.CHI,
                "__label__hu": names.NLP.LANG.HUN,
                "__label__ar": names.NLP.LANG.ARA,
                "__label__ca": names.NLP.LANG.CAT,
                "__label__fi": names.NLP.LANG.FIN,
                "__label__cs": names.NLP.LANG.CZE,
                "__label__fa": names.NLP.LANG.PER,
                "__label__sr": names.NLP.LANG.SRP,
                "__label__el": names.NLP.LANG.GRE,
                "__label__vi": names.NLP.LANG.VIE,
                "__label__bg": names.NLP.LANG.BUL,
                "__label__ko": names.NLP.LANG.KOR,
                "__label__no": names.NLP.LANG.NOR,
                "__label__mk": names.NLP.LANG.MAC,
                "__label__ro": names.NLP.LANG.RUM,
                "__label__id": names.NLP.LANG.IND,
                "__label__th": names.NLP.LANG.THA,
                "__label__hy": names.NLP.LANG.ARM,
                "__label__da": names.NLP.LANG.DAN,
                "__label__ta": names.NLP.LANG.TAM,
                "__label__hi": names.NLP.LANG.HIN,
                "__label__hr": names.NLP.LANG.HRV,
                "__label__sh": "",
                "__label__be": names.NLP.LANG.BEL,
                "__label__ka": names.NLP.LANG.GEO,
                "__label__te": names.NLP.LANG.TEL,
                "__label__kk": names.NLP.LANG.KAZ,
                "__label__war": names.NLP.LANG.WAR,
                "__label__lt": names.NLP.LANG.LIT,
                "__label__gl": names.NLP.LANG.GLG,
                "__label__sk": names.NLP.LANG.SLO,
                "__label__bn": names.NLP.LANG.BEN,
                "__label__eu": names.NLP.LANG.BAQ,
                "__label__sl": names.NLP.LANG.SLV,
                "__label__kn": "",
                "__label__ml": names.NLP.LANG.MAL,
                "__label__mr": names.NLP.LANG.MAR,
                "__label__et": names.NLP.LANG.EST,
                "__label__az": names.NLP.LANG.AZE,
                "__label__ms": "",
                "__label__sq": names.NLP.LANG.ALB,
                "__label__la": names.NLP.LANG.LAT,
                "__label__bs": names.NLP.LANG.BOS,
                "__label__nn": names.NLP.LANG.NNO,
                "__label__ur": names.NLP.LANG.URD,
                "__label__lv": "",
                "__label__my": "",
                "__label__tt": "",
                "__label__af": "",
                "__label__oc": "",
                "__label__nds": "",
                "__label__ky": "",
                "__label__ast": "",
                "__label__tl": "",
                "__label__is": "",
                "__label__ia": "",
                "__label__si": "",
                "__label__gu": "",
                "__label__km": "",
                "__label__br": "",
                "__label__ba": "",
                "__label__uz": "",
                "__label__bo": "",
                "__label__pa": "",
                "__label__vo": "",
                "__label__als": "",
                "__label__ne": "",
                "__label__cy": "",
                "__label__jbo": "",
                "__label__fy": "",
                "__label__mn": "",
                "__label__lb": "",
                "__label__ce": "",
                "__label__ug": "",
                "__label__tg": "",
                "__label__sco": "",
                "__label__sa": "",
                "__label__cv": "",
                "__label__jv": "",
                "__label__min": "",
                "__label__io": "",
                "__label__or": "",
                "__label__as": "",
                "__label__new": "",
                "__label__ga": "",
                "__label__mg": "",
                "__label__an": "",
                "__label__ckb": "",
                "__label__sw": "",
                "__label__bar": "",
                "__label__lmo": "",
                "__label__yi": ":" "",
                "__label__arz": "",
                "__label__mhr": "",
                "__label__azb": "",
                "__label__sah": "",
                "__label__pnb": "",
                "__label__su": "",
                "__label__bpy": "",
                "__label__pms": "",
                "__label__ilo": "",
                "__label__wuu": "",
                "__label__ku": "",
                "__label__ps": "",
                "__label__ie": "",
                "__label__xmf": "",
                "__label__yue": "",
                "__label__gom": "",
                "__label__li": "",
                "__label__mwl": "",
                "__label__kw": "",
                "__label__sd": "",
                "__label__hsb": "",
                "__label__scn": "",
                "__label__gd": "",
                "__label__pam": "",
                "__label__bh": "",
                "__label__mai": "",
                "__label__vec": "",
                "__label__mt": "",
                "__label__dv": "",
                "__label__wa": "",
                "__label__mzn": "",
                "__label__am": "",
                "__label__qu": "",
                "__label__eml": "",
                "__label__cbk": "",
                "__label__tk": "",
                "__label__rm": "",
                "__label__os": "",
                "__label__vls": "",
                "__label__yo": "",
                "__label__lo": "",
                "__label__lez": "",
                "__label__so": "",
                "__label__myv": "",
                "__label__diq": "",
                "__label__mrj": "",
                "__label__dsb": "",
                "__label__frr": "",
                "__label__ht": "",
                "__label__gn": "",
                "__label__bxr": "",
                "__label__kv": "",
                "__label__sc": "",
                "__label__nah": "",
                "__label__krc": "",
                "__label__bcl": "",
                "__label__nap": "",
                "__label__gv": "",
                "__label__av": "",
                "__label__rue": "",
                "__label__xal": "",
                "__label__pfl": "",
                "__label__dty": "",
                "__label__hif": "",
                "__label__co": "",
                "__label__lrc": "",
                "__label__vep": "",
                "__label__tyv": "",
            },
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
        if profile.config is not None:
            return os.path.join(get_weights_dir_path(), profile.name)
        logger.info(
            "Model is not registered. Please make sure the weights are available in the weights cache " "directory"
        )
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


def print_model_infos() -> None:
    """
    Prints a table with all registered model profiles and some of their attributes (name, description, config and
    categories)
    """

    profiles = ModelCatalog.CATALOG.values()
    num_columns = min(6, len(profiles))
    infos = []
    for profile in profiles:
        infos.append((profile.name, profile.description, profile.config, profile.categories))
    table = tabulate(
        infos,
        headers=["name", "description", "config", "categories"] * (num_columns // 2),
        tablefmt="fancy_grid",
        stralign="left",
        numalign="left",
    )
    print(colored(table, "cyan"))


class ModelDownloadManager:  # pylint: disable=R0903
    """
    A registry for built-in models. Registered models have weights that can be downloaded and cached. Do not use
    this class for registering your own models as there are much more sophisticated tools for experimenting and
    versioning.
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
            if profile.tp_model:
                file_names = get_tp_weight_names(name)
            else:
                hf_model_name = profile.hf_model_name
                assert isinstance(hf_model_name, str)
                file_names.append(hf_model_name)
            if profile.hf_repo_id:
                ModelDownloadManager.load_model_from_hf_hub(profile, absolute_path_weights, file_names)
                absolute_path_configs = ModelCatalog.get_full_path_configs(name)
                ModelDownloadManager.load_configs_from_hf_hub(profile, absolute_path_configs)
            else:
                ModelDownloadManager._load_from_gd(profile, absolute_path_weights, file_names)

            return absolute_path_weights

        logger.info("Will use not registered model. Make sure path to weights is correctly set")
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
        directory, _ = os.path.split(absolute_path)
        if not file_names:
            file_names = profile.hf_model_name
        for expect_size, file_name in zip(profile.size, file_names):
            size = ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)
            if expect_size is not None and size != expect_size:
                logger.error("File downloaded from %s does not match the expected size!", repo_id)
                logger.error("You may have downloaded a broken file, or the upstream may have modified the file.")

    @staticmethod
    def _load_from_gd(profile: ModelProfile, absolute_path: str, file_names: List[str]) -> None:
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
        directory, _ = os.path.split(absolute_path)
        for file_name in profile.hf_config_file:
            ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)

    @staticmethod
    def _load_from_hf_hub(repo_id: str, file_name: str, cache_directory: str, force_download: bool = False) -> int:
        url = hf_hub_url(repo_id=repo_id, filename=file_name)
        f_path = cached_download(
            url, cache_dir=cache_directory, force_filename=file_name, force_download=force_download
        )
        stat_info = os.stat(f_path)
        size = stat_info.st_size
        assert size > 0, f"Downloaded an empty file from {url}!"
        return size
