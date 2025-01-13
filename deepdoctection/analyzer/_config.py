# -*- coding: utf-8 -*-
# File: config.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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

"""Pipeline configuration for deepdoctection analyzer. Do not change the defaults in this file. """

from ..datapoint.view import IMAGE_DEFAULTS
from ..utils.metacfg import AttrDict
from ..utils.settings import CellType, LayoutType

cfg = AttrDict()


cfg.LANGUAGE = None
cfg.LIB = None
cfg.DEVICE = None
cfg.USE_ROTATOR = False
cfg.USE_LAYOUT = True
cfg.USE_TABLE_SEGMENTATION = True

cfg.TF.LAYOUT.WEIGHTS = "layout/model-800000_inf_only.data-00000-of-00001"
cfg.TF.LAYOUT.FILTER = None

cfg.TF.CELL.WEIGHTS = "cell/model-1800000_inf_only.data-00000-of-00001"
cfg.TF.CELL.FILTER = None

cfg.TF.ITEM.WEIGHTS = "item/model-1620000_inf_only.data-00000-of-00001"
cfg.TF.ITEM.FILTER = None

cfg.PT.LAYOUT.WEIGHTS = "layout/d2_model_0829999_layout_inf_only.pt"
cfg.PT.LAYOUT.WEIGHTS_TS = "layout/d2_model_0829999_layout_inf_only.ts"
cfg.PT.LAYOUT.FILTER = None
cfg.PT.LAYOUT.PAD.TOP = 60
cfg.PT.LAYOUT.PAD.RIGHT = 60
cfg.PT.LAYOUT.PAD.BOTTOM = 60
cfg.PT.LAYOUT.PAD.LEFT = 60

cfg.PT.ITEM.WEIGHTS = "item/d2_model_1639999_item_inf_only.pt"
cfg.PT.ITEM.WEIGHTS_TS = "item/d2_model_1639999_item_inf_only.ts"
cfg.PT.ITEM.FILTER = None
cfg.PT.ITEM.PAD.TOP = 60
cfg.PT.ITEM.PAD.RIGHT = 60
cfg.PT.ITEM.PAD.BOTTOM = 60
cfg.PT.ITEM.PAD.LEFT = 60

cfg.PT.CELL.WEIGHTS = "cell/d2_model_1849999_cell_inf_only.pt"
cfg.PT.CELL.WEIGHTS_TS = "cell/d2_model_1849999_cell_inf_only.ts"
cfg.PT.CELL.FILTER = None

cfg.USE_LAYOUT_NMS = False
cfg.LAYOUT_NMS_PAIRS.COMBINATIONS = None
cfg.LAYOUT_NMS_PAIRS.THRESHOLDS = None
cfg.LAYOUT_NMS_PAIRS.PRIORITY = None

cfg.SEGMENTATION.ASSIGNMENT_RULE = "ioa"
cfg.SEGMENTATION.THRESHOLD_ROWS = 0.4
cfg.SEGMENTATION.THRESHOLD_COLS = 0.4
cfg.SEGMENTATION.FULL_TABLE_TILING = True
cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS = 0.001
cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS = 0.001
cfg.SEGMENTATION.CELL_CATEGORY_ID = 12
cfg.SEGMENTATION.TABLE_NAME = LayoutType.TABLE
cfg.SEGMENTATION.PUBTABLES_CELL_NAMES = [
    CellType.SPANNING,
    CellType.ROW_HEADER,
    CellType.COLUMN_HEADER,
    CellType.PROJECTED_ROW_HEADER,
    LayoutType.CELL,
]
cfg.SEGMENTATION.PUBTABLES_SPANNING_CELL_NAMES = [
    CellType.SPANNING,
    CellType.ROW_HEADER,
    CellType.COLUMN_HEADER,
    CellType.PROJECTED_ROW_HEADER,
]
cfg.SEGMENTATION.PUBTABLES_ITEM_NAMES = [LayoutType.ROW, LayoutType.COLUMN]
cfg.SEGMENTATION.PUBTABLES_SUB_ITEM_NAMES = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]
cfg.SEGMENTATION.CELL_NAMES = [CellType.HEADER, CellType.BODY, LayoutType.CELL]
cfg.SEGMENTATION.ITEM_NAMES = [LayoutType.ROW, LayoutType.COLUMN]
cfg.SEGMENTATION.SUB_ITEM_NAMES = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]
cfg.SEGMENTATION.PUBTABLES_ITEM_HEADER_CELL_NAMES = [CellType.COLUMN_HEADER, CellType.ROW_HEADER]
cfg.SEGMENTATION.PUBTABLES_ITEM_HEADER_THRESHOLDS = [0.6, 0.0001]
cfg.SEGMENTATION.STRETCH_RULE = "equal"

cfg.USE_TABLE_REFINEMENT = True
cfg.USE_PDF_MINER = False

cfg.PDF_MINER.X_TOLERANCE = 3
cfg.PDF_MINER.Y_TOLERANCE = 3

cfg.USE_OCR = True

cfg.OCR.USE_TESSERACT = True
cfg.OCR.USE_DOCTR = False
cfg.OCR.USE_TEXTRACT = False
cfg.OCR.CONFIG.TESSERACT = "dd/conf_tesseract.yaml"

cfg.OCR.WEIGHTS.DOCTR_WORD.TF = "doctr/db_resnet50/tf/db_resnet50-adcafc63.zip"
cfg.OCR.WEIGHTS.DOCTR_WORD.PT = "doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt"
cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.TF = "doctr/crnn_vgg16_bn/tf/crnn_vgg16_bn-76b7f2c6.zip"
cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.PT = "doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"

cfg.TEXT_CONTAINER = IMAGE_DEFAULTS["text_container"]
cfg.WORD_MATCHING.PARENTAL_CATEGORIES = [
    LayoutType.TEXT,
    LayoutType.TITLE,
    LayoutType.LIST,
    LayoutType.CELL,
    CellType.COLUMN_HEADER,
    CellType.PROJECTED_ROW_HEADER,
    CellType.SPANNING,
    CellType.ROW_HEADER,
]
cfg.WORD_MATCHING.RULE = "ioa"
cfg.WORD_MATCHING.THRESHOLD = 0.6
cfg.WORD_MATCHING.MAX_PARENT_ONLY = True

cfg.TEXT_ORDERING.TEXT_BLOCK_CATEGORIES = IMAGE_DEFAULTS["text_block_categories"]
cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES = IMAGE_DEFAULTS["floating_text_block_categories"]
cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER = False
cfg.TEXT_ORDERING.STARTING_POINT_TOLERANCE = 0.005
cfg.TEXT_ORDERING.BROKEN_LINE_TOLERANCE = 0.003
cfg.TEXT_ORDERING.HEIGHT_TOLERANCE = 2.0
cfg.TEXT_ORDERING.PARAGRAPH_BREAK = 0.035

cfg.USE_LAYOUT_LINK = False
cfg.LAYOUT_LINK.PARENTAL_CATEGORIES = []
cfg.LAYOUT_LINK.CHILD_CATEGORIES = []

cfg.freeze()
