# -*- coding: utf-8 -*-
# File: dd.py

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
Module for **deep**doctection analyzer.

-factory build_analyzer for a given config

-user factory with a reduced config setting
"""

from __future__ import annotations

import os
from typing import Optional

from ..extern.pt.ptutils import get_torch_device
from ..extern.tp.tfutils import disable_tp_layer_logging, get_tf_device
from ..pipe.doctectionpipe import DoctectionPipe
from ..utils.env_info import ENV_VARS_TRUE
from ..utils.error import DependencyError
from ..utils.file_utils import tensorpack_available
from ..utils.fs import get_configs_dir_path, get_package_path, maybe_copy_config_to_cache
from ..utils.logger import LoggingRecord, logger
from ..utils.metacfg import set_config_by_yaml
from ..utils.types import PathLikeOrStr
from ._config import cfg
from .factory import ServiceFactory

__all__ = [
    "config_sanity_checks",
    "get_dd_analyzer",
]

_DD_ONE = "deepdoctection/configs/conf_dd_one.yaml"
_TESSERACT = "deepdoctection/configs/conf_tesseract.yaml"
_MODEL_CHOICES = {
    "layout": [
        "layout/d2_model_0829999_layout_inf_only.pt",
        "xrf_layout/model_final_inf_only.pt",
        "microsoft/table-transformer-detection/pytorch_model.bin",
    ],
    "segmentation": [
        "item/model-1620000_inf_only.data-00000-of-00001",
        "xrf_item/model_final_inf_only.pt",
        "microsoft/table-transformer-structure-recognition/pytorch_model.bin",
        "deepdoctection/tatr_tab_struct_v2/pytorch_model.bin",
    ],
    "ocr": ["Tesseract", "DocTr", "Textract"],
    "doctr_word": ["doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt"],
    "doctr_recognition": [
        "doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
        "doctr/crnn_vgg16_bn/pt/pytorch_model.bin",
    ],
    "llm": ["gpt-3.5-turbo", "gpt-4"],
    "segmentation_choices": {
        "item/model-1620000_inf_only.data-00000-of-00001": "cell/model-1800000_inf_only.data-00000-of-00001",
        "xrf_item/model_final_inf_only.pt": "xrf_cell/model_final_inf_only.pt",
        "microsoft/table-transformer-structure-recognition/pytorch_model.bin": None,
        "deepdoctection/tatr_tab_struct_v2/pytorch_model.bin": None,
    },
}


def config_sanity_checks() -> None:
    """Some config sanity checks"""
    if cfg.USE_PDF_MINER and cfg.USE_OCR and cfg.OCR.USE_DOCTR:
        raise ValueError("Configuration USE_PDF_MINER= True and USE_OCR=True and USE_DOCTR=True is not allowed")
    if cfg.USE_OCR:
        if cfg.OCR.USE_TESSERACT + cfg.OCR.USE_DOCTR + cfg.OCR.USE_TEXTRACT != 1:
            raise ValueError(
                "Choose either OCR.USE_TESSERACT=True or OCR.USE_DOCTR=True or OCR.USE_TEXTRACT=True "
                "and set the other two to False. Only one OCR system can be activated."
            )


def get_dd_analyzer(
    reset_config_file: bool = True,
    config_overwrite: Optional[list[str]] = None,
    path_config_file: Optional[PathLikeOrStr] = None,
) -> DoctectionPipe:
    """
    Factory function for creating the built-in **deep**doctection analyzer.

    The Standard Analyzer is a pipeline that comprises the following analysis components:

    - Document layout analysis

    - Table segmentation

    - Text extraction/OCR

    - Reading order

    We refer to the various notebooks and docs for running an analyzer and changing the configs.

    :param reset_config_file: This will copy the `.yaml` file with default variables to the `.cache` and therefore
                              resetting all configurations if set to `True`.
    :param config_overwrite: Passing a list of string arguments and values to overwrite the `.yaml` configuration with
                             highest priority, e.g. ["USE_TABLE_SEGMENTATION=False",
                                                     "USE_OCR=False",
                                                     "TF.LAYOUT.WEIGHTS=my_fancy_pytorch_model"]
    :param path_config_file: Path to a custom config file. Can be outside of the .cache directory.
    :return: A DoctectionPipe instance with given configs
    """
    config_overwrite = [] if config_overwrite is None else config_overwrite
    lib = "TF" if os.environ.get("DD_USE_TF", "0") in ENV_VARS_TRUE else "PT"
    if lib == "TF":
        device = get_tf_device()
    elif lib == "PT":
        device = get_torch_device()
    else:
        raise DependencyError("At least one of the env variables DD_USE_TF or DD_USE_TORCH must be set.")
    dd_one_config_path = maybe_copy_config_to_cache(
        get_package_path(), get_configs_dir_path() / "dd", _DD_ONE, reset_config_file
    )
    maybe_copy_config_to_cache(get_package_path(), get_configs_dir_path() / "dd", _TESSERACT)

    # Set up of the configuration and logging
    file_cfg = set_config_by_yaml(dd_one_config_path if not path_config_file else path_config_file)
    cfg.freeze(freezed=False)
    cfg.overwrite_config(file_cfg)

    cfg.freeze(freezed=False)
    cfg.LANGUAGE = None
    cfg.LIB = lib
    cfg.DEVICE = device
    cfg.freeze()

    if config_overwrite:
        cfg.update_args(config_overwrite)

    config_sanity_checks()
    logger.info(LoggingRecord(f"Config: \n {str(cfg)}", cfg.to_dict()))  # type: ignore

    # will silent all TP logging while building the tower
    if tensorpack_available():
        disable_tp_layer_logging()

    return ServiceFactory.build_analyzer(cfg)
