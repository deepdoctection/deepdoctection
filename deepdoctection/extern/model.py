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
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Union

import jsonlines
from huggingface_hub import cached_download, hf_hub_url  # type: ignore
from tabulate import tabulate
from termcolor import colored

from ..utils.fs import download, get_configs_dir_path, get_weights_dir_path
from ..utils.logger import LoggingRecord, log_once, logger
from ..utils.settings import CellType, Languages, LayoutType, ObjectTypes, get_type
from ..utils.types import PathLikeOrStr

__all__ = ["ModelCatalog", "ModelDownloadManager", "print_model_infos", "ModelProfile"]


@dataclass(frozen=True)
class ModelProfile:
    """
    Class for model profile. Add for each model one ModelProfile to the ModelCatalog
    """

    name: str
    description: str

    size: list[int]
    tp_model: bool = field(default=False)
    config: Optional[str] = field(default=None)
    preprocessor_config: Optional[str] = field(default=None)
    hf_repo_id: Optional[str] = field(default=None)
    hf_model_name: Optional[str] = field(default=None)
    hf_config_file: Optional[list[str]] = field(default=None)
    urls: Optional[list[str]] = field(default=None)
    categories: Optional[Mapping[int, ObjectTypes]] = field(default=None)
    categories_orig: Optional[Mapping[str, ObjectTypes]] = field(default=None)
    dl_library: Optional[str] = field(default=None)
    model_wrapper: Optional[str] = field(default=None)
    architecture: Optional[str] = field(default=None)

    def as_dict(self) -> dict[str, Any]:
        """
        returns a dict of the dataclass
        """
        return asdict(self)


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well.

    To get an overview of all registered models

        print(ModelCatalog.get_model_list())

    To get a model card for some specific model:

        profile = ModelCatalog.get_profile("layout/model-800000_inf_only.data-00000-of-00001")
        print(profile.description)

    Some models will have their weights and configs stored in the cache. To instantiate predictors one will sometimes
    need their path. Use

        path_weights = ModelCatalog.get_full_path_configs("layout/model-800000_inf_only.data-00000-of-00001")
        path_configs = ModelCatalog.get_full_path_weights("layout/model-800000_inf_only.data-00000-of-00001")

    To register a new model

        ModelCatalog.get_full_path_configs("my_new_model")
    """

    CATALOG: dict[str, ModelProfile] = {
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
                1: LayoutType.TEXT,
                2: LayoutType.TITLE,
                3: LayoutType.LIST,
                4: LayoutType.TABLE,
                5: LayoutType.FIGURE,
            },
            dl_library="TF",
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
            categories={1: LayoutType.CELL},
            dl_library="TF",
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
            categories={1: LayoutType.ROW, 2: LayoutType.COLUMN},
            dl_library="TF",
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
            categories={1: LayoutType.ROW, 2: LayoutType.COLUMN},
            dl_library="TF",
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
            dl_library="TF",
            categories={
                1: LayoutType.TEXT,
                2: LayoutType.TITLE,
                3: LayoutType.LIST,
                4: LayoutType.TABLE,
                5: LayoutType.FIGURE,
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
            categories={1: LayoutType.CELL},
            dl_library="TF",
            model_wrapper="TPFrcnnDetector",
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
                1: LayoutType.TEXT,
                2: LayoutType.TITLE,
                3: LayoutType.LIST,
                4: LayoutType.TABLE,
                5: LayoutType.FIGURE,
            },
            dl_library="PT",
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
                1: LayoutType.TEXT,
                2: LayoutType.TITLE,
                3: LayoutType.LIST,
                4: LayoutType.TABLE,
                5: LayoutType.FIGURE,
            },
            dl_library="PT",
            model_wrapper="D2FrcnnDetector",
        ),
        "layout/d2_model_0829999_layout_inf_only.ts": ModelProfile(
            name="layout/d2_model_0829999_layout_inf_only.ts",
            description="Detectron2 layout detection model trained on Publaynet. Torchscript export",
            config="dd/d2/layout/CASCADE_RCNN_R_50_FPN_GN_TS.yaml",
            size=[274974842],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            hf_model_name="d2_model_0829999_layout_inf_only.ts",
            hf_config_file=["CASCADE_RCNN_R_50_FPN_GN_TS.yaml"],
            categories={
                1: LayoutType.TEXT,
                2: LayoutType.TITLE,
                3: LayoutType.LIST,
                4: LayoutType.TABLE,
                5: LayoutType.FIGURE,
            },
            dl_library="PT",
            model_wrapper="D2FrcnnTracingDetector",
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
            categories={1: LayoutType.CELL},
            dl_library="PT",
            model_wrapper="D2FrcnnDetector",
        ),
        "cell/d2_model_1849999_cell_inf_only.ts": ModelProfile(
            name="cell/d2_model_1849999_cell_inf_only.ts",
            description="Detectron2 cell detection inference only model trained on Pubtabnet. Torchscript export",
            config="dd/d2/cell/CASCADE_RCNN_R_50_FPN_GN_TS.yaml",
            size=[274898682],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            hf_model_name="d2_model_1849999_cell_inf_only.ts",
            hf_config_file=["CASCADE_RCNN_R_50_FPN_GN_TS.yaml"],
            categories={1: LayoutType.CELL},
            dl_library="PT",
            model_wrapper="D2FrcnnTracingDetector",
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
            categories={1: LayoutType.CELL},
            dl_library="PT",
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
            categories={1: LayoutType.ROW, 2: LayoutType.COLUMN},
            dl_library="PT",
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
            categories={1: LayoutType.ROW, 2: LayoutType.COLUMN},
            dl_library="PT",
            model_wrapper="D2FrcnnDetector",
        ),
        "item/d2_model_1639999_item_inf_only.ts": ModelProfile(
            name="item/d2_model_1639999_item_inf_only.ts",
            description="Detectron2 cell detection inference only model trained on Pubtabnet. Torchscript export",
            config="dd/d2/item/CASCADE_RCNN_R_50_FPN_GN_TS.yaml",
            size=[274910970],
            tp_model=False,
            hf_repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            hf_model_name="d2_model_1639999_item_inf_only.ts",
            hf_config_file=["CASCADE_RCNN_R_50_FPN_GN_TS.yaml"],
            categories={1: LayoutType.ROW, 2: LayoutType.COLUMN},
            dl_library="PT",
            model_wrapper="D2FrcnnTracingDetector",
        ),
        "nielsr/lilt-xlm-roberta-base/pytorch_model.bin": ModelProfile(
            name="nielsr/lilt-xlm-roberta-base/pytorch_model.bin",
            description="LiLT build with a RobertaXLM base model",
            config="nielsr/lilt-xlm-roberta-base/config.json",
            size=[1136743583],
            tp_model=False,
            hf_repo_id="nielsr/lilt-xlm-roberta-base",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
        ),
        "SCUT-DLVCLab/lilt-infoxlm-base/pytorch_model.bin": ModelProfile(
            name="SCUT-DLVCLab/lilt-infoxlm-base/pytorch_model.bin",
            description="Language-Independent Layout Transformer - InfoXLM model by stitching a pre-trained InfoXLM"
            " and a pre-trained Language-Independent Layout Transformer (LiLT) together. It was introduced"
            " in the paper LiLT: A Simple yet Effective Language-Independent Layout Transformer for"
            " Structured Document Understanding by Wang et al. and first released in this repository.",
            config="SCUT-DLVCLab/lilt-infoxlm-base/config.json",
            size=[1136743583],
            tp_model=False,
            hf_repo_id="SCUT-DLVCLab/lilt-infoxlm-base",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
        ),
        "SCUT-DLVCLab/lilt-roberta-en-base/pytorch_model.bin": ModelProfile(
            name="SCUT-DLVCLab/lilt-roberta-en-base/pytorch_model.bin",
            description="Language-Independent Layout Transformer - RoBERTa model by stitching a pre-trained RoBERTa"
            " (English) and a pre-trained Language-Independent Layout Transformer (LiLT) together. It was"
            " introduced in the paper LiLT: A Simple yet Effective Language-Independent Layout Transformer"
            " for Structured Document Understanding by Wang et al. and first released in this repository.",
            config="SCUT-DLVCLab/lilt-roberta-en-base/config.json",
            size=[523151519],
            tp_model=False,
            hf_repo_id="SCUT-DLVCLab/lilt-roberta-en-base",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
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
            dl_library="PT",
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
            dl_library="PT",
        ),
        "microsoft/layoutlmv2-base-uncased/pytorch_model.bin": ModelProfile(
            name="microsoft/layoutlmv2-base-uncased/pytorch_model.bin",
            description="LayoutLMv2 is an improved version of LayoutLM with new pre-training tasks to model the"
            " interaction among text, layout, and image in a single multi-modal framework. It outperforms"
            " strong baselines and achieves new state-of-the-art results on a wide variety of downstream"
            " visually-rich document understanding tasks, including , including FUNSD (0.7895 → 0.8420),"
            " CORD (0.9493 → 0.9601), SROIE (0.9524 → 0.9781), Kleister-NDA (0.834 → 0.852), RVL-CDIP"
            " (0.9443 → 0.9564), and DocVQA (0.7295 → 0.8672). The license is cc-by-nc-sa-4.0",
            size=[802243295],
            tp_model=False,
            config="microsoft/layoutlmv2-base-uncased/config.json",
            hf_repo_id="microsoft/layoutlmv2-base-uncased",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
        ),
        "microsoft/layoutxlm-base/pytorch_model.bin": ModelProfile(
            name="microsoft/layoutxlm-base/pytorch_model.bin",
            description="Multimodal pre-training with text, layout, and image has achieved SOTA performance for "
            "visually-rich document understanding tasks recently, which demonstrates the great potential"
            " for joint learning across different modalities. In this paper, we present LayoutXLM, a"
            " multimodal pre-trained model for multilingual document understanding, which aims to bridge"
            " the language barriers for visually-rich document understanding. To accurately evaluate"
            " LayoutXLM, we also introduce a multilingual form understanding benchmark dataset named XFUN,"
            " which includes form understanding samples in 7 languages (Chinese, Japanese, Spanish, French,"
            " Italian, German, Portuguese), and key-value pairs are manually labeled for each language."
            " Experiment results show that the LayoutXLM model has significantly outperformed the existing"
            " SOTA cross-lingual pre-trained models on the XFUN dataset. The license is cc-by-nc-sa-4.0",
            size=[1476537178],
            tp_model=False,
            config="microsoft/layoutxlm-base/config.json",
            hf_repo_id="microsoft/layoutxlm-base",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
        ),
        "microsoft/layoutlmv3-base/pytorch_model.bin": ModelProfile(
            name="microsoft/layoutlmv3-base/pytorch_model.bin",
            description="LayoutLMv3 is a pre-trained multimodal Transformer for Document AI with unified text and"
            " image masking. The simple unified architecture and training objectives make LayoutLMv3 a"
            " general-purpose pre-trained model. For example, LayoutLMv3 can be fine-tuned for both"
            " text-centric tasks, including form understanding, receipt understanding, and document"
            " visual question answering, and image-centric tasks such as document image classification"
            " and document layout analysis. The license is cc-by-nc-sa-4.0",
            size=[501380823],
            tp_model=False,
            config="microsoft/layoutlmv3-base/config.json",
            hf_repo_id="microsoft/layoutlmv3-base",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
        ),
        "microsoft/table-transformer-detection/pytorch_model.bin": ModelProfile(
            name="microsoft/table-transformer-detection/pytorch_model.bin",
            description="Table Transformer (DETR) model trained on PubTables1M. It was introduced in the paper "
            "PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents by Smock et "
            "al. This model is devoted to table detection",
            size=[115393245],
            tp_model=False,
            config="microsoft/table-transformer-detection/config.json",
            preprocessor_config="microsoft/table-transformer-detection/preprocessor_config.json",
            hf_repo_id="microsoft/table-transformer-detection",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json", "preprocessor_config.json"],
            categories={1: LayoutType.TABLE, 2: LayoutType.TABLE_ROTATED},
            dl_library="PT",
            model_wrapper="HFDetrDerivedDetector",
        ),
        "microsoft/table-transformer-structure-recognition/pytorch_model.bin": ModelProfile(
            name="microsoft/table-transformer-structure-recognition/pytorch_model.bin",
            description="Table Transformer (DETR) model trained on PubTables1M. It was introduced in the paper "
            "PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents by Smock et "
            "al. This model is devoted to table structure recognition and assumes to receive a cropped"
            "table as input. It will predict rows, column and spanning cells",
            size=[115509981],
            tp_model=False,
            config="microsoft/table-transformer-structure-recognition/config.json",
            preprocessor_config="microsoft/table-transformer-structure-recognition/preprocessor_config.json",
            hf_repo_id="microsoft/table-transformer-structure-recognition",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json", "preprocessor_config.json"],
            categories={
                1: LayoutType.TABLE,
                2: LayoutType.COLUMN,
                3: LayoutType.ROW,
                4: CellType.COLUMN_HEADER,
                5: CellType.PROJECTED_ROW_HEADER,
                6: CellType.SPANNING,
            },
            dl_library="PT",
            model_wrapper="HFDetrDerivedDetector",
        ),
        "doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt": ModelProfile(
            name="doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt",
            description="Doctr implementation of DBNet from “Real-time Scene Text Detection with Differentiable "
            "Binarization”. For more information please check "
            "https://mindee.github.io/doctr/using_doctr/using_models.html#. This is the Pytorch artefact.",
            size=[101971449],
            urls=["https://doctr-static.mindee.com/models?id=v0.3.1/db_resnet50-ac60cadc.pt&src=0"],
            categories={1: LayoutType.WORD},
            dl_library="PT",
            model_wrapper="DoctrTextlineDetector",
            architecture="db_resnet50",
        ),
        "doctr/db_resnet50/tf/db_resnet50-adcafc63.zip": ModelProfile(
            name="doctr/db_resnet50/tf/db_resnet50-adcafc63.zip",
            description="Doctr implementation of DBNet from “Real-time Scene Text Detection with Differentiable "
            "Binarization”. For more information please check "
            "https://mindee.github.io/doctr/using_doctr/using_models.html#. This is the Tensorflow artefact.",
            size=[94178964],
            urls=["https://doctr-static.mindee.com/models?id=v0.2.0/db_resnet50-adcafc63.zip&src=0"],
            categories={1: LayoutType.WORD},
            dl_library="TF",
            model_wrapper="DoctrTextlineDetector",
            architecture="db_resnet50",
        ),
        "doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt": ModelProfile(
            name="doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
            description="Doctr implementation of CRNN from “An End-to-End Trainable Neural Network for Image-based "
            "Sequence Recognition and Its Application to Scene Text Recognition”. For more information "
            "please check https://mindee.github.io/doctr/using_doctr/using_models.html#. This is the Pytorch "
            "artefact.",
            size=[63286381],
            urls=["https://doctr-static.mindee.com/models?id=v0.3.1/crnn_vgg16_bn-9762b0b0.pt&src=0"],
            dl_library="PT",
            model_wrapper="DoctrTextRecognizer",
            architecture="crnn_vgg16_bn",
        ),
        "doctr/crnn_vgg16_bn/tf/crnn_vgg16_bn-76b7f2c6.zip": ModelProfile(
            name="doctr/crnn_vgg16_bn/tf/crnn_vgg16_bn-76b7f2c6.zip",
            description="Doctr implementation of CRNN from “An End-to-End Trainable Neural Network for Image-based "
            "Sequence Recognition and Its Application to Scene Text Recognition”. For more information "
            "please check https://mindee.github.io/doctr/using_doctr/using_models.html#. This is the Tensorflow "
            "artefact.",
            size=[58758994],
            urls=["https://doctr-static.mindee.com/models?id=v0.3.0/crnn_vgg16_bn-76b7f2c6.zip&src=0"],
            dl_library="TF",
            model_wrapper="DoctrTextRecognizer",
            architecture="crnn_vgg16_bn",
        ),
        "FacebookAI/xlm-roberta-base": ModelProfile(
            name="FacebookAI/xlm-roberta-base/pytorch_model.bin",
            description="XLM-RoBERTa model pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages."
            " It was introduced in the paper Unsupervised Cross-lingual Representation Learning at Scale"
            " by Conneau et al. and first released in this repository.",
            size=[1115590446],
            tp_model=False,
            config="FacebookAI/xlm-roberta-base/config.json",
            hf_repo_id="FacebookAI/xlm-roberta-base",
            hf_model_name="pytorch_model.bin",
            hf_config_file=["config.json"],
            dl_library="PT",
        ),
        "fasttext/lid.176.bin": ModelProfile(
            name="fasttext/lid.176.bin",
            description="Fasttext language detection model",
            size=[131266198],
            urls=["https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"],
            categories={
                1: Languages.ENGLISH,
                2: Languages.RUSSIAN,
                3: Languages.GERMAN,
                4: Languages.FRENCH,
                5: Languages.ITALIAN,
                6: Languages.JAPANESE,
                7: Languages.SPANISH,
                8: Languages.CEBUANO,
                9: Languages.TURKISH,
                10: Languages.PORTUGUESE,
                11: Languages.UKRAINIAN,
                12: Languages.ESPERANTO,
                13: Languages.POLISH,
                14: Languages.SWEDISH,
                15: Languages.DUTCH,
                16: Languages.HEBREW,
                17: Languages.CHINESE,
                18: Languages.HUNGARIAN,
                19: Languages.ARABIC,
                20: Languages.CATALAN,
                21: Languages.FINNISH,
                22: Languages.CZECH,
                23: Languages.PERSIAN,
                24: Languages.SERBIAN,
                25: Languages.GREEK,
                26: Languages.VIETNAMESE,
                27: Languages.BULGARIAN,
                28: Languages.KOREAN,
                29: Languages.NORWEGIAN,
                30: Languages.MACEDONIAN,
                31: Languages.ROMANIAN,
                32: Languages.INDONESIAN,
                33: Languages.THAI,
                34: Languages.ARMENIAN,
                35: Languages.DANISH,
                36: Languages.TAMIL,
                37: Languages.HINDI,
                38: Languages.CROATIAN,
                39: Languages.NOT_DEFINED,
                40: Languages.BELARUSIAN,
                41: Languages.GEORGIAN,
                42: Languages.TELUGU,
                43: Languages.KAZAKH,
                44: Languages.WARAY,
                45: Languages.LITHUANIAN,
                46: Languages.SCOTTISH,
                47: Languages.SLOVAK,
                48: Languages.BENIN,
                49: Languages.BASQUE,
                50: Languages.SLOVENIAN,
                51: Languages.NOT_DEFINED,
                52: Languages.MALAYALAM,
                53: Languages.MARATHI,
                54: Languages.ESTONIAN,
                55: Languages.AZERBAIJANI,
                56: Languages.NOT_DEFINED,
                57: Languages.ALBANIAN,
                58: Languages.LATIN,
                59: Languages.BOSNIAN,
                60: Languages.NORWEGIAN_NOVOSIBIRSK,
                61: Languages.URDU,
                62: Languages.NOT_DEFINED,
                63: Languages.NOT_DEFINED,
                64: Languages.NOT_DEFINED,
                65: Languages.NOT_DEFINED,
                66: Languages.NOT_DEFINED,
                67: Languages.NOT_DEFINED,
                68: Languages.NOT_DEFINED,
                69: Languages.NOT_DEFINED,
                70: Languages.NOT_DEFINED,
                71: Languages.NOT_DEFINED,
                72: Languages.NOT_DEFINED,
                73: Languages.NOT_DEFINED,
                74: Languages.NOT_DEFINED,
                75: Languages.NOT_DEFINED,
                76: Languages.NOT_DEFINED,
                77: Languages.NOT_DEFINED,
                78: Languages.NOT_DEFINED,
                79: Languages.NOT_DEFINED,
                80: Languages.NOT_DEFINED,
                81: Languages.NOT_DEFINED,
                82: Languages.NOT_DEFINED,
                83: Languages.NOT_DEFINED,
                84: Languages.NOT_DEFINED,
                85: Languages.NOT_DEFINED,
                86: Languages.NOT_DEFINED,
                87: Languages.NOT_DEFINED,
                88: Languages.NOT_DEFINED,
                89: Languages.NOT_DEFINED,
                90: Languages.NOT_DEFINED,
                91: Languages.NOT_DEFINED,
                92: Languages.NOT_DEFINED,
                93: Languages.NOT_DEFINED,
                94: Languages.NOT_DEFINED,
                95: Languages.NOT_DEFINED,
                96: Languages.NOT_DEFINED,
                97: Languages.NOT_DEFINED,
                98: Languages.NOT_DEFINED,
                99: Languages.NOT_DEFINED,
                100: Languages.NOT_DEFINED,
                101: Languages.NOT_DEFINED,
                102: Languages.NOT_DEFINED,
                103: Languages.NOT_DEFINED,
                104: Languages.NOT_DEFINED,
                105: Languages.NOT_DEFINED,
                106: Languages.NOT_DEFINED,
                107: Languages.NOT_DEFINED,
                108: Languages.NOT_DEFINED,
                109: Languages.NOT_DEFINED,
                110: Languages.NOT_DEFINED,
                111: Languages.NOT_DEFINED,
                112: Languages.NOT_DEFINED,
                113: Languages.NOT_DEFINED,
                114: Languages.NOT_DEFINED,
                115: Languages.NOT_DEFINED,
                116: Languages.NOT_DEFINED,
                117: Languages.NOT_DEFINED,
                118: Languages.NOT_DEFINED,
                119: Languages.NOT_DEFINED,
                120: Languages.NOT_DEFINED,
                121: Languages.NOT_DEFINED,
                122: Languages.NOT_DEFINED,
                123: Languages.NOT_DEFINED,
                124: Languages.NOT_DEFINED,
                125: Languages.NOT_DEFINED,
                126: Languages.NOT_DEFINED,
                127: Languages.NOT_DEFINED,
                128: Languages.NOT_DEFINED,
                129: Languages.NOT_DEFINED,
                130: Languages.NOT_DEFINED,
                131: Languages.NOT_DEFINED,
                132: Languages.NOT_DEFINED,
                133: Languages.NOT_DEFINED,
                134: Languages.NOT_DEFINED,
                135: Languages.NOT_DEFINED,
                136: Languages.NOT_DEFINED,
                137: Languages.NOT_DEFINED,
                138: Languages.NOT_DEFINED,
                139: Languages.NOT_DEFINED,
                140: Languages.NOT_DEFINED,
                141: Languages.NOT_DEFINED,
                142: Languages.NOT_DEFINED,
                143: Languages.NOT_DEFINED,
                144: Languages.NOT_DEFINED,
                145: Languages.NOT_DEFINED,
                146: Languages.NOT_DEFINED,
                147: Languages.NOT_DEFINED,
                148: Languages.NOT_DEFINED,
                149: Languages.NOT_DEFINED,
                150: Languages.NOT_DEFINED,
                151: Languages.NOT_DEFINED,
                152: Languages.NOT_DEFINED,
                153: Languages.NOT_DEFINED,
                154: Languages.NOT_DEFINED,
                155: Languages.NOT_DEFINED,
                156: Languages.NOT_DEFINED,
                157: Languages.NOT_DEFINED,
                158: Languages.NOT_DEFINED,
                159: Languages.NOT_DEFINED,
                160: Languages.NOT_DEFINED,
                161: Languages.NOT_DEFINED,
                162: Languages.NOT_DEFINED,
                163: Languages.NOT_DEFINED,
                164: Languages.NOT_DEFINED,
                165: Languages.NOT_DEFINED,
                166: Languages.NOT_DEFINED,
                167: Languages.NOT_DEFINED,
                168: Languages.NOT_DEFINED,
                169: Languages.NOT_DEFINED,
                170: Languages.NOT_DEFINED,
                171: Languages.NOT_DEFINED,
                172: Languages.NOT_DEFINED,
                173: Languages.NOT_DEFINED,
                174: Languages.NOT_DEFINED,
                175: Languages.NOT_DEFINED,
                176: Languages.NOT_DEFINED,
            },
            categories_orig={
                "__label__en": Languages.ENGLISH,
                "__label__ru": Languages.RUSSIAN,
                "__label__de": Languages.GERMAN,
                "__label__fr": Languages.FRENCH,
                "__label__it": Languages.ITALIAN,
                "__label__ja": Languages.JAPANESE,
                "__label__es": Languages.SPANISH,
                "__label__ceb": Languages.CEBUANO,
                "__label__tr": Languages.TURKISH,
                "__label__pt": Languages.PORTUGUESE,
                "__label__uk": Languages.UKRAINIAN,
                "__label__eo": Languages.ESPERANTO,
                "__label__pl": Languages.POLISH,
                "__label__sv": Languages.SWEDISH,
                "__label__nl": Languages.DUTCH,
                "__label__he": Languages.HEBREW,
                "__label__zh": Languages.CHINESE,
                "__label__hu": Languages.HUNGARIAN,
                "__label__ar": Languages.ARABIC,
                "__label__ca": Languages.CATALAN,
                "__label__fi": Languages.FINNISH,
                "__label__cs": Languages.CZECH,
                "__label__fa": Languages.PERSIAN,
                "__label__sr": Languages.SERBIAN,
                "__label__el": Languages.GREEK,
                "__label__vi": Languages.VIETNAMESE,
                "__label__bg": Languages.BULGARIAN,
                "__label__ko": Languages.KOREAN,
                "__label__no": Languages.NORWEGIAN,
                "__label__mk": Languages.MACEDONIAN,
                "__label__ro": Languages.ROMANIAN,
                "__label__id": Languages.INDONESIAN,
                "__label__th": Languages.THAI,
                "__label__hy": Languages.ARMENIAN,
                "__label__da": Languages.DANISH,
                "__label__ta": Languages.TAMIL,
                "__label__hi": Languages.HINDI,
                "__label__hr": Languages.CROATIAN,
                "__label__sh": Languages.NOT_DEFINED,
                "__label__be": Languages.BELARUSIAN,
                "__label__ka": Languages.GEORGIAN,
                "__label__te": Languages.TELUGU,
                "__label__kk": Languages.KAZAKH,
                "__label__war": Languages.WARAY,
                "__label__lt": Languages.LITHUANIAN,
                "__label__gl": Languages.SCOTTISH,
                "__label__sk": Languages.SLOVAK,
                "__label__bn": Languages.BENIN,
                "__label__eu": Languages.BASQUE,
                "__label__sl": Languages.SLOVENIAN,
                "__label__kn": Languages.NOT_DEFINED,
                "__label__ml": Languages.MALAYALAM,
                "__label__mr": Languages.MARATHI,
                "__label__et": Languages.ESTONIAN,
                "__label__az": Languages.AZERBAIJANI,
                "__label__ms": Languages.NOT_DEFINED,
                "__label__sq": Languages.ALBANIAN,
                "__label__la": Languages.LATIN,
                "__label__bs": Languages.BOSNIAN,
                "__label__nn": Languages.NORWEGIAN_NOVOSIBIRSK,
                "__label__ur": Languages.URDU,
                "__label__lv": Languages.NOT_DEFINED,
                "__label__my": Languages.NOT_DEFINED,
                "__label__tt": Languages.NOT_DEFINED,
                "__label__af": Languages.NOT_DEFINED,
                "__label__oc": Languages.NOT_DEFINED,
                "__label__nds": Languages.NOT_DEFINED,
                "__label__ky": Languages.NOT_DEFINED,
                "__label__ast": Languages.NOT_DEFINED,
                "__label__tl": Languages.NOT_DEFINED,
                "__label__is": Languages.NOT_DEFINED,
                "__label__ia": Languages.NOT_DEFINED,
                "__label__si": Languages.NOT_DEFINED,
                "__label__gu": Languages.NOT_DEFINED,
                "__label__km": Languages.NOT_DEFINED,
                "__label__br": Languages.NOT_DEFINED,
                "__label__ba": Languages.NOT_DEFINED,
                "__label__uz": Languages.NOT_DEFINED,
                "__label__bo": Languages.NOT_DEFINED,
                "__label__pa": Languages.NOT_DEFINED,
                "__label__vo": Languages.NOT_DEFINED,
                "__label__als": Languages.NOT_DEFINED,
                "__label__ne": Languages.NOT_DEFINED,
                "__label__cy": Languages.NOT_DEFINED,
                "__label__jbo": Languages.NOT_DEFINED,
                "__label__fy": Languages.NOT_DEFINED,
                "__label__mn": Languages.NOT_DEFINED,
                "__label__lb": Languages.NOT_DEFINED,
                "__label__ce": Languages.NOT_DEFINED,
                "__label__ug": Languages.NOT_DEFINED,
                "__label__tg": Languages.NOT_DEFINED,
                "__label__sco": Languages.NOT_DEFINED,
                "__label__sa": Languages.NOT_DEFINED,
                "__label__cv": Languages.NOT_DEFINED,
                "__label__jv": Languages.NOT_DEFINED,
                "__label__min": Languages.NOT_DEFINED,
                "__label__io": Languages.NOT_DEFINED,
                "__label__or": Languages.NOT_DEFINED,
                "__label__as": Languages.NOT_DEFINED,
                "__label__new": Languages.NOT_DEFINED,
                "__label__ga": Languages.NOT_DEFINED,
                "__label__mg": Languages.NOT_DEFINED,
                "__label__an": Languages.NOT_DEFINED,
                "__label__ckb": Languages.NOT_DEFINED,
                "__label__sw": Languages.NOT_DEFINED,
                "__label__bar": Languages.NOT_DEFINED,
                "__label__lmo": Languages.NOT_DEFINED,
                "__label__yi": Languages.NOT_DEFINED,
                "__label__arz": Languages.NOT_DEFINED,
                "__label__mhr": Languages.NOT_DEFINED,
                "__label__azb": Languages.NOT_DEFINED,
                "__label__sah": Languages.NOT_DEFINED,
                "__label__pnb": Languages.NOT_DEFINED,
                "__label__su": Languages.NOT_DEFINED,
                "__label__bpy": Languages.NOT_DEFINED,
                "__label__pms": Languages.NOT_DEFINED,
                "__label__ilo": Languages.NOT_DEFINED,
                "__label__wuu": Languages.NOT_DEFINED,
                "__label__ku": Languages.NOT_DEFINED,
                "__label__ps": Languages.NOT_DEFINED,
                "__label__ie": Languages.NOT_DEFINED,
                "__label__xmf": Languages.NOT_DEFINED,
                "__label__yue": Languages.NOT_DEFINED,
                "__label__gom": Languages.NOT_DEFINED,
                "__label__li": Languages.NOT_DEFINED,
                "__label__mwl": Languages.NOT_DEFINED,
                "__label__kw": Languages.NOT_DEFINED,
                "__label__sd": Languages.NOT_DEFINED,
                "__label__hsb": Languages.NOT_DEFINED,
                "__label__scn": Languages.NOT_DEFINED,
                "__label__gd": Languages.NOT_DEFINED,
                "__label__pam": Languages.NOT_DEFINED,
                "__label__bh": Languages.NOT_DEFINED,
                "__label__mai": Languages.NOT_DEFINED,
                "__label__vec": Languages.NOT_DEFINED,
                "__label__mt": Languages.NOT_DEFINED,
                "__label__dv": Languages.NOT_DEFINED,
                "__label__wa": Languages.NOT_DEFINED,
                "__label__mzn": Languages.NOT_DEFINED,
                "__label__am": Languages.NOT_DEFINED,
                "__label__qu": Languages.NOT_DEFINED,
                "__label__eml": Languages.NOT_DEFINED,
                "__label__cbk": Languages.NOT_DEFINED,
                "__label__tk": Languages.NOT_DEFINED,
                "__label__rm": Languages.NOT_DEFINED,
                "__label__os": Languages.NOT_DEFINED,
                "__label__vls": Languages.NOT_DEFINED,
                "__label__yo": Languages.NOT_DEFINED,
                "__label__lo": Languages.NOT_DEFINED,
                "__label__lez": Languages.NOT_DEFINED,
                "__label__so": Languages.NOT_DEFINED,
                "__label__myv": Languages.NOT_DEFINED,
                "__label__diq": Languages.NOT_DEFINED,
                "__label__mrj": Languages.NOT_DEFINED,
                "__label__dsb": Languages.NOT_DEFINED,
                "__label__frr": Languages.NOT_DEFINED,
                "__label__ht": Languages.NOT_DEFINED,
                "__label__gn": Languages.NOT_DEFINED,
                "__label__bxr": Languages.NOT_DEFINED,
                "__label__kv": Languages.NOT_DEFINED,
                "__label__sc": Languages.NOT_DEFINED,
                "__label__nah": Languages.NOT_DEFINED,
                "__label__krc": Languages.NOT_DEFINED,
                "__label__bcl": Languages.NOT_DEFINED,
                "__label__nap": Languages.NOT_DEFINED,
                "__label__gv": Languages.NOT_DEFINED,
                "__label__av": Languages.NOT_DEFINED,
                "__label__rue": Languages.NOT_DEFINED,
                "__label__xal": Languages.NOT_DEFINED,
                "__label__pfl": Languages.NOT_DEFINED,
                "__label__dty": Languages.NOT_DEFINED,
                "__label__hif": Languages.NOT_DEFINED,
                "__label__co": Languages.NOT_DEFINED,
                "__label__lrc": Languages.NOT_DEFINED,
                "__label__vep": Languages.NOT_DEFINED,
                "__label__tyv": Languages.NOT_DEFINED,
            },
            model_wrapper="FasttextLangDetector",
        ),
    }

    @staticmethod
    def get_full_path_weights(name: PathLikeOrStr) -> PathLikeOrStr:
        """
        Returns the absolute path of weights.

        Note, that weights are sometimes not defined by only one artefact. The returned string will only represent one
        weights artefact.

        :param name: model name
        :return: absolute weight path
        """
        try:
            profile = ModelCatalog.get_profile(os.fspath(name))
        except KeyError:
            logger.info(
                LoggingRecord(
                    f"Model {name} not found in ModelCatalog. Make sure, you have places model weights "
                    f"in the cache dir"
                )
            )
            profile = ModelProfile(name="", description="", size=[])
        if profile.name:
            return os.path.join(get_weights_dir_path(), profile.name)
        log_once(
            f"Model {name} is not registered. Please make sure the weights are available in the weights "
            f"cache directory or the full path you provide is correct"
        )
        if os.path.isfile(name):
            return name
        return os.path.join(get_weights_dir_path(), name)

    @staticmethod
    def get_full_path_configs(name: PathLikeOrStr) -> PathLikeOrStr:
        """
        Return the absolute path of configs for some given weights. Alternatively, pass last a path to a config file
        (without the base path to the cache config directory).

        Note, that configs are sometimes not defined by only one file. The returned string will only represent one
        file.

        :param name: model name
        :return: absolute path to the config
        """
        try:
            profile = ModelCatalog.get_profile(os.fspath(name))
        except KeyError:
            logger.info(
                LoggingRecord(
                    f"Model {name} not found in ModelCatalog. Make sure, you have places model "
                    f"configs in the cache dir"
                )
            )
            profile = ModelProfile(name="", description="", size=[])
        if profile.config is not None:
            return os.path.join(get_configs_dir_path(), profile.config)
        return os.path.join(get_configs_dir_path(), name)

    @staticmethod
    def get_full_path_preprocessor_configs(name: Union[str]) -> PathLikeOrStr:
        """
        Return the absolute path of preprocessor configs for some given weights. Preprocessor are occasionally provided
        by the transformer library.

        :param name: model name
        :return: absolute path to the preprocessor config
        """

        try:
            profile = ModelCatalog.get_profile(name)
        except KeyError:
            profile = ModelProfile(name="", description="", size=[])
            logger.info(
                LoggingRecord(
                    f"Model {name} not found in ModelCatalog. Make sure, you have places preprocessor configs "
                    f"in the cache dir",
                )
            )
        if profile.preprocessor_config is not None:
            return os.path.join(get_configs_dir_path(), profile.preprocessor_config)
        return os.path.join(get_configs_dir_path(), name)

    @staticmethod
    def get_model_list() -> list[PathLikeOrStr]:
        """
        Returns a list of absolute paths of registered models.
        """
        return [os.path.join(get_weights_dir_path(), profile.name) for profile in ModelCatalog.CATALOG.values()]

    @staticmethod
    def get_profile_list() -> list[str]:
        """
        Returns a list profile keys.
        """
        return list(ModelCatalog.CATALOG.keys())

    @staticmethod
    def is_registered(path_weights: PathLikeOrStr) -> bool:
        """
        Checks if some weights belong to a registered model

        :param path_weights: relative or absolute path
        :return: True if the weights are registered in `ModelCatalog`
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
            return profile
        raise KeyError(f"Model Profile {name} does not exist. Please make sure the model is registered")

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

    @staticmethod
    def load_profiles_from_file(path: Optional[PathLikeOrStr] = None) -> None:
        """
        Load model profiles from a jsonl file and extend `CATALOG` with the new profiles.

        :param path: Path to the file. `None` is allowed but it will do nothing.
        """
        if not path:
            return
        with jsonlines.open(path) as reader:
            for obj in reader:
                if not obj["name"] in ModelCatalog.CATALOG:
                    categories = obj.get("categories") or {}
                    obj["categories"] = {int(key): get_type(val) for key, val in categories.items()}
                    ModelCatalog.register(obj["name"], ModelProfile(**obj))

    @staticmethod
    def save_profiles_to_file(target_path: PathLikeOrStr) -> None:
        """
        Save model profiles to a jsonl file.

        :param target_path: Path to the file.
        """
        with jsonlines.open(target_path, mode="w") as writer:
            for profile in ModelCatalog.CATALOG.values():
                writer.write(profile.as_dict())
        writer.close()


# Additional profiles can be added
ModelCatalog.load_profiles_from_file(os.environ.get("MODEL_CATALOG", None))


def get_tp_weight_names(name: str) -> list[str]:
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
        tbl_input: list[Union[Mapping[int, ObjectTypes], str]] = [profile.name]
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

        # if you are not sure about the model name use the ModelCatalog
        ModelDownloadManager.maybe_download_weights_and_configs("layout/model-800000_inf_only.data-00000-of-00001")
    """

    @staticmethod
    def maybe_download_weights_and_configs(name: str) -> PathLikeOrStr:
        """
        Check if some model is registered. If yes, it will check if their weights
        must be downloaded. Only weights that have not the same expected size will be downloaded again.

        :param name: A path to some model weights
        :return: Absolute path to model weights if model is registered
        """

        absolute_path_weights = ModelCatalog.get_full_path_weights(name)
        file_names: list[str] = []
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
                        file_names.append(url.split("/")[-1].split("&")[0])
                else:
                    file_names.append(model_name)
            if profile.hf_repo_id:
                if not os.path.isfile(absolute_path_weights):
                    ModelDownloadManager.load_model_from_hf_hub(profile, absolute_path_weights, file_names)
                absolute_path_configs = ModelCatalog.get_full_path_configs(name)
                if not os.path.isfile(absolute_path_configs):
                    ModelDownloadManager.load_configs_from_hf_hub(profile, absolute_path_configs)
            else:
                ModelDownloadManager._load_from_gd(profile, absolute_path_weights, file_names)

            return absolute_path_weights

        return absolute_path_weights

    @staticmethod
    def load_model_from_hf_hub(profile: ModelProfile, absolute_path: PathLikeOrStr, file_names: list[str]) -> None:
        """
        Load a model from the Huggingface hub for a given profile and saves the model at the directory of the given
        path.

        :param profile: Profile according to `ModelCatalog.get_profile(path_weights)`
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
                    LoggingRecord(
                        f"File downloaded from {repo_id} does not match the expected size! You may have downloaded"
                        " a broken file, or the upstream may have modified the file."
                    )
                )

    @staticmethod
    def _load_from_gd(profile: ModelProfile, absolute_path: PathLikeOrStr, file_names: list[str]) -> None:
        if profile.urls is None:
            raise ValueError("urls cannot be None")
        for size, url, file_name in zip(profile.size, profile.urls, file_names):
            directory, _ = os.path.split(absolute_path)
            download(str(url), directory, file_name, int(size))

    @staticmethod
    def load_configs_from_hf_hub(profile: ModelProfile, absolute_path: PathLikeOrStr) -> None:
        """
        Load config file(s) from the Huggingface hub for a given profile and saves the model at the directory of the
        given path.

        :param profile: Profile according to `ModelCatalog.get_profile(path_weights)`
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
    def _load_from_hf_hub(
        repo_id: str, file_name: str, cache_directory: PathLikeOrStr, force_download: bool = False
    ) -> int:
        url = hf_hub_url(repo_id=repo_id, filename=file_name)
        token = os.environ.get("HF_CREDENTIALS", None)
        f_path = cached_download(
            url,
            cache_dir=cache_directory,
            force_filename=file_name,
            force_download=force_download,
            token=token,
            legacy_cache_layout=True,
        )
        if f_path:
            stat_info = os.stat(f_path)
            size = stat_info.st_size

            assert size > 0, f"Downloaded an empty file from {url}!"
            return size
        raise TypeError("Returned value from cached_download cannot be Null")
