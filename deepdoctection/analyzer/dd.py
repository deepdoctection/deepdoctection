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
- factory `build_analyzer` for a given config
- user factory with a reduced config setting
"""

from __future__ import annotations

import os
from typing import Optional

from ..extern.pt.ptutils import get_torch_device
from ..extern.tp.tfutils import disable_tp_layer_logging, get_tf_device
from ..pipe.doctectionpipe import DoctectionPipe
from ..utils.env_info import ENV_VARS_TRUE
from ..utils.file_utils import detectron2_available, tensorpack_available
from ..utils.fs import get_configs_dir_path, get_package_path, maybe_copy_config_to_cache
from ..utils.logger import LoggingRecord, logger
from ..utils.metacfg import set_config_by_yaml
from ..utils.types import PathLikeOrStr
from .config import cfg
from .factory import ServiceFactory

__all__ = [
    "config_sanity_checks",
    "get_dd_analyzer",
]

_DD_ONE = "deepdoctection/configs/conf_dd_one.yaml"
_TESSERACT = "deepdoctection/configs/conf_tesseract.yaml"


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
    load_default_config_file: bool = False,
    config_overwrite: Optional[list[str]] = None,
    path_config_file: Optional[PathLikeOrStr] = None,
) -> DoctectionPipe:
    """
    Factory function for creating the built-in **deep**doctection analyzer.

    Info:
        The Standard Analyzer is a pipeline that comprises the following analysis components:

        - Rotation
        - Document layout analysis
        - Table segmentation
        - Text extraction/OCR
        - Reading order
        - Layout linking

    Args:
        reset_config_file: This will copy the `.yaml` file with default variables to the `.cache` and therefore
            resetting all configurations if set to `True`.
        load_default_config_file: This will load the default config file from the `.cache` directory if set to `True`.
            If set to `False`, the config file will be ignored.
        config_overwrite: Passing a list of string arguments and values to overwrite the `.yaml`
            configuration with highest priority, e.g. `["USE_TABLE_SEGMENTATION=False", "USE_OCR=False",
            "TF.LAYOUT.WEIGHTS=my_fancy_pytorch_model"]`.
        path_config_file: Path to a custom config file. Can be outside of the `.cache` directory.

    Returns:
        DoctectionPipe: A `DoctectionPipe` instance with given configs.
    """
    config_overwrite = [] if config_overwrite is None else config_overwrite
    if os.environ.get("DD_USE_TF", "0") in ENV_VARS_TRUE:
        lib = "TF"
        device = get_tf_device()
    elif os.environ.get("DD_USE_TORCH", "0") in ENV_VARS_TRUE:
        lib = "PT"
        device = get_torch_device()
    else:
        lib = None
        device = None
    dd_one_config_path = maybe_copy_config_to_cache(
        get_package_path(), get_configs_dir_path() / "dd", _DD_ONE, reset_config_file
    )
    maybe_copy_config_to_cache(get_package_path(), get_configs_dir_path() / "dd", _TESSERACT)

    cfg.freeze(freezed=False)
    if load_default_config_file:
        # Set up of the configuration and logging
        file_cfg = set_config_by_yaml(dd_one_config_path if not path_config_file else path_config_file)
        cfg.overwrite_config(file_cfg)
    cfg.LANGUAGE = None
    cfg.LIB = lib
    cfg.DEVICE = device
    if not detectron2_available() or cfg.PT.LAYOUT.WEIGHTS is None:
        cfg.PT.ENFORCE_WEIGHTS.LAYOUT = False
    if not detectron2_available() or cfg.PT.ITEM.WEIGHTS is None:
        cfg.PT.ENFORCE_WEIGHTS.ITEM = False
    if not detectron2_available() or cfg.PT.CELL.WEIGHTS is None:
        cfg.PT.ENFORCE_WEIGHTS.CELL = False

    if config_overwrite:
        cfg.update_args(config_overwrite)
    cfg.freeze()

    config_sanity_checks()
    logger.info(LoggingRecord(f"Config: \n {str(cfg)}", cfg.to_dict()))  # type: ignore

    # will silent all TP logging while building the tower
    if tensorpack_available():
        disable_tp_layer_logging()

    return ServiceFactory.build_analyzer(cfg)
