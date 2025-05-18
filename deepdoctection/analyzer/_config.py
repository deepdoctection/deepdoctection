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

"""Pipeline configuration for deepdoctection analyzer."""

from ..datapoint.view import IMAGE_DEFAULTS
from ..utils.metacfg import AttrDict
from ..utils.settings import CellType, LayoutType

cfg = AttrDict()

# General note: All models used in *.WEIGHTS must be registered in the ModelCatalog.
# Registered models are listed in deepdoctection/profiles.jsonl. To add new models,
# either extend this file with additional JSON objects or provide a separate JSONL file
# and reference it via the MODEL_CATALOG environment variable.

# Relevant only for Tesseract OCR. Specifies the language model to use.
# Supported language codes are listed at:
# https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.
# Note: models must be downloaded in advance.
cfg.LANGUAGE = None

# Deep learning framework. Choose either 'TF' (TensorFlow) or 'PT' (PyTorch).
# Selection is made via environment variables: DD_USE_TF or DD_USE_PT.
cfg.LIB = None

# Device configuration.
# For PyTorch: torch.device("cpu"), torch.device("mps"), or torch.device("cuda")
# For TensorFlow: tf.device("/cpu:0") or tf.device("/gpu:0")
cfg.DEVICE = None

# Enables the initial pipeline component using TesseractRotationTransformer to auto-rotate pages
# by 90-degree increments. All subsequent components process the rotated page.
cfg.USE_ROTATOR = False

# Enables layout analysis component (second in the pipeline) for either full document layout analysis (DLA)
# or single-object detection. Additional configurations via PT.LAYOUT.*, TF.LAYOUT.*, and PT.ENFORCE_WEIGHTS.LAYOUT.
cfg.USE_LAYOUT = True

# Enables optional fine-grained Non-Maximum Suppression (NMS) after layout detection.
# Configure via LAYOUT_NMS_PAIRS.* settings.
cfg.USE_LAYOUT_NMS = True

# Enables table segmentation (third and later pipeline components).
# Applies row/column detection, optional cell detection, and segmentation services.
# Configure sub-services via PT.ITEM.*, TF.ITEM.*, PT.CELL.*, TF.CELL.*, and SEGMENTATION.*
cfg.USE_TABLE_SEGMENTATION = True

# Enables optional refinement of table structure to ensure valid HTML generation.
# Should be set to False when using the Table Transformer approach.
cfg.USE_TABLE_REFINEMENT = False

# Enables text extraction using PDFPlumber. Only works on PDFs with embedded text layers.
# Configure additional behavior using PDF_MINER.*
cfg.USE_PDF_MINER = False

# Enables OCR functionality using Tesseract, DocTr, or Textract.
# Also activates MatchingService and TextOrderingService to associate text with layout elements.
# Further configurations via OCR.*, WORD_MATCHING.*, TEXT_CONTAINER, and TEXT_ORDERING.*
cfg.USE_OCR = True

# Enables MatchingService to associate nearby layout elements (e.g., figures and captions).
cfg.USE_LAYOUT_LINK = False

# Enables line matching in post-processing. Useful when synthetic line elements are created
# (e.g., by grouping orphan text containers). Only applicable if list items were previously grouped.
cfg.USE_LINE_MATCHER = False

# Relevant when LIB = TF. Specifies the layout detection model.
# This model should detect multiple or single objects across an entire page.
# Currently, only one default model is supported.
cfg.TF.LAYOUT.WEIGHTS = "Aryn/deformable-detr-DocLayNet/model.safetensors"

# Filters out unnecessary categories from the layout detection model output.
# Accepts either a list of strings (e.g., ['list', 'figure']) or a list of ObjectTypes
# (e.g., [LayoutType.LIST, LayoutType.FIGURE]).
cfg.TF.LAYOUT.FILTER = None

# Relevant when LIB = PT. Allows selection between two model formats:
# 1. Standard PyTorch weights (.pt or .safetensors), or
# 2. TorchScript weights (.ts), which require only the Torch runtime and not the model implementation.
# If PT.ENFORCE_WEIGHTS.LAYOUT is set to True, PT.LAYOUT.WEIGHTS will take precedence.
# The get_dd_analyzer() function will set PT.ENFORCE_WEIGHTS.LAYOUT = False automatically
# if Detectron2 is not installed or PT.LAYOUT.WEIGHTS is None.
cfg.PT.ENFORCE_WEIGHTS.LAYOUT = True

# Specifies the PyTorch layout detection model (standard weights).
# Must detect single or multiple objects across the full page.
# Acceptable formats: .pt or .safetensors (e.g.,
# layout/d2_model_0829999_layout_inf_only.pt,
# microsoft/table-transformer-detection/pytorch_model.bin,
# Aryn/deformable-detr-DocLayNet/model.safetensors).
cfg.PT.LAYOUT.WEIGHTS = "layout/d2_model_0829999_layout_inf_only.pt"

# Specifies the TorchScript version of the layout model.
# Must detect single or multiple objects across the full page.
# Acceptable format: .ts files (e.g., layout/d2_model_0829999_layout_inf_only.ts).
cfg.PT.LAYOUT.WEIGHTS_TS = "layout/d2_model_0829999_layout_inf_only.ts"

# Filters out unwanted categories from the model’s predictions.
# Accepts either string values (e.g., ['list', 'figure']) or ObjectTypes
# (e.g., [LayoutType.LIST, LayoutType.FIGURE]).
cfg.PT.LAYOUT.FILTER = None

# Adds padding to the image, which may be required for some models such as
# microsoft/table-transformer-detection/pytorch_model.bin to improve detection accuracy.
# Padding values should not be manually set; they are defined in the ModelProfile inside ServiceFactory.
# If PT.LAYOUT.PADDING is True, you must also set the values for PT.LAYOUT.PAD.TOP, .RIGHT, .BOTTOM, and .LEFT.
cfg.PT.LAYOUT.PADDING = False

# Padding value for the top edge of the image. Required by some layout detection models.
cfg.PT.LAYOUT.PAD.TOP = 0

# Padding value for the right edge of the image. Required by some layout detection models.
cfg.PT.LAYOUT.PAD.RIGHT = 0

# Padding value for the bottom edge of the image. Required by some layout detection models.
cfg.PT.LAYOUT.PAD.BOTTOM = 0

# Padding value for the left edge of the image. Required by some layout detection models.
cfg.PT.LAYOUT.PAD.LEFT = 0

# Non-Maximum Suppression (NMS) configuration for overlapping layout elements.
# For each element pair, define:
# 1. the combination of element types,
# 2. the IoU threshold, and
# 3. which element has priority (or None).
#
# Example:
# LAYOUT_NMS_PAIRS.COMBINATIONS = [['table', 'title'], ['table', 'text']]
# LAYOUT_NMS_PAIRS.THRESHOLDS = [0.001, 0.01]
# LAYOUT_NMS_PAIRS.PRIORITY = ['table', None]
cfg.LAYOUT_NMS_PAIRS.COMBINATIONS = [[LayoutType.TABLE,LayoutType.TITLE],
                                     [LayoutType.TABLE,LayoutType.TEXT],
                                     [LayoutType.TABLE,LayoutType.KEY_VALUE_AREA],
                                     [LayoutType.TABLE,LayoutType.LIST_ITEM],
                                     [LayoutType.TABLE,LayoutType.LIST],
                                     [LayoutType.TITLE,LayoutType.TEXT],
                                     [LayoutType.TEXT,LayoutType.KEY_VALUE_AREA],
                                     [LayoutType.TEXT,LayoutType.LIST_ITEM],
                                     [LayoutType.KEY_VALUE_AREA,LayoutType.LIST_ITEM]]
cfg.LAYOUT_NMS_PAIRS.THRESHOLDS = [0.001,0.01,0.01,0.001,0.01,0.05,0.01,0.01,0.01]
cfg.LAYOUT_NMS_PAIRS.PRIORITY = [LayoutType.TABLE,
                                 LayoutType.TABLE,
                                 LayoutType.TABLE,
                                 LayoutType.TABLE,
                                 LayoutType.TABLE,
                                 LayoutType.TEXT,
                                 LayoutType.TEXT,
                                 None,
                                 LayoutType.KEY_VALUE_AREA]

# Relevant when LIB = TF. Specifies the item detection model (for rows and columns).
# Currently, only the default model is supported.
cfg.TF.ITEM.WEIGHTS = "item/model-1620000_inf_only.data-00000-of-00001"

# Filters out unnecessary categories from the item detection model.
# Accepts either a list of strings (e.g., ['row', 'column']) or ObjectTypes.
cfg.TF.ITEM.FILTER = None

# Relevant when LIB = PT. Use either TorchScript weights via PT.ITEM.WEIGHTS_TS
# or standard PyTorch weights via PT.ITEM.WEIGHTS (.pt or .safetensors).
# If PT.ENFORCE_WEIGHTS.ITEM = True, PT.ITEM.WEIGHTS will take precedence over TorchScript.
cfg.PT.ENFORCE_WEIGHTS.ITEM = True

# Specifies the PyTorch model weights for item detection.
# Use either .pt or .safetensors files.
cfg.PT.ITEM.WEIGHTS = "deepdoctection/tatr_tab_struct_v2/pytorch_model.bin"

# Specifies the TorchScript model for item detection.
# Use .ts files for deployment without model implementation dependencies.
cfg.PT.ITEM.WEIGHTS_TS = "item/d2_model_1639999_item_inf_only.ts"

# Filters out unnecessary categories from the item detection model.
# For example, the model microsoft/table-transformer-structure-recognition/pytorch_model.bin
# predicts not only rows and columns, but also tables. To prevent redundant outputs, use:
# PT.ITEM.FILTER = ['table']
cfg.PT.ITEM.FILTER = ['table']

# Enables image padding for item detection. Required for models such as
# microsoft/table-transformer-structure-recognition/pytorch_model.bin to optimize accuracy.
# Padding values are derived from the ModelProfile within the ServiceFactory and should not be manually set.
# If PT.ITEM.PADDING = True, you must define all edge values: TOP, RIGHT, BOTTOM, and LEFT.
cfg.PT.ITEM.PADDING = False

# Padding value for the top edge of the sub-image used in item detection.
cfg.PT.ITEM.PAD.TOP = 60

# Padding value for the right edge of the sub-image used in item detection.
cfg.PT.ITEM.PAD.RIGHT = 60

# Padding value for the bottom edge of the sub-image used in item detection.
cfg.PT.ITEM.PAD.BOTTOM = 60

# Padding value for the left edge of the sub-image used in item detection.
cfg.PT.ITEM.PAD.LEFT = 60

# Configuration for the second SubImagePipelineComponent.
# This is only used in the original Deepdoctection table recognition approach,
# not with the Table Transformer method.
# The CELL configuration structure mirrors that of the ITEM component.
cfg.TF.CELL.WEIGHTS = "cell/model-1800000_inf_only.data-00000-of-00001"

# Filters out unnecessary categories from the cell detection model output.
cfg.TF.CELL.FILTER = None

# Determines whether PT.CELL.WEIGHTS should take priority over PT.CELL.WEIGHTS_TS.
# If set to True, standard PyTorch weights are enforced.
cfg.PT.ENFORCE_WEIGHTS.CELL = True

# Specifies the PyTorch model weights for cell detection using standard formats (.pt or .safetensors).
cfg.PT.CELL.WEIGHTS = "cell/d2_model_1849999_cell_inf_only.pt"

# Specifies the TorchScript model for cell detection (.ts format).
cfg.PT.CELL.WEIGHTS_TS = "cell/d2_model_1849999_cell_inf_only.ts"

# Filters out unwanted categories from the cell detection model.
cfg.PT.CELL.FILTER = None

# Enables padding for the sub-image used in cell detection.
# Required for certain models to enhance prediction quality.
# If set to True, padding values for all four edges must be defined.
cfg.PT.CELL.PADDING = False

# Padding value for the top edge of the sub-image used in cell detection.
cfg.PT.CELL.PAD.TOP = 60

# Padding value for the right edge of the sub-image used in cell detection.
cfg.PT.CELL.PAD.RIGHT = 60

# Padding value for the bottom edge of the sub-image used in cell detection.
cfg.PT.CELL.PAD.BOTTOM = 60

# Padding value for the left edge of the sub-image used in cell detection.
cfg.PT.CELL.PAD.LEFT = 60

# Specifies the rule used to assign detected cells to rows and columns.
# Can be either 'iou' (Intersection over Union) or 'ioa' (Intersection over Area).
# In the Table Transformer approach, this also applies to special cell types like spanning or header cells.
cfg.SEGMENTATION.ASSIGNMENT_RULE = "ioa"

# Threshold for assigning a (special) cell to a row based on the chosen rule (IOU or IOA).
# The row assignment is based on the highest-overlapping row.
# Multiple overlaps can lead to increased rowspan.
cfg.SEGMENTATION.THRESHOLD_ROWS = 0.4

# Threshold for assigning a (special) cell to a column based on the chosen rule (IOU or IOA).
# The column assignment is based on the highest-overlapping column.
cfg.SEGMENTATION.THRESHOLD_COLS = 0.4

# Removes overlapping rows based on an IoU threshold.
# Helps prevent multiple row spans caused by overlapping detections.
# Note: for better alignment, SEGMENTATION.FULL_TABLE_TILING can be enabled.
# Using a low threshold here may result in a very coarse grid.
cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS = 0.2

# Same as above, but applied to columns.
cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS = 0.2

# Ensures that predicted rows and columns fully cover the table region.
# When enabled, rows will be stretched horizontally and vertically to fit the full region.
# For rows, the first row will be stretched to the top, and the space to the second row is used to estimate the
# bottom edge. This rule applies similarly to columns.
cfg.SEGMENTATION.FULL_TABLE_TILING = True

# Defines how row and column boundaries are stretched when tiling is enabled.
# Options:
# - "left": lower edge equals the upper edge of the next row
# - "equal": lower edge is halfway between two adjacent rows
cfg.SEGMENTATION.STRETCH_RULE = "equal"

# Specifies the layout category used to identify tables.
# Used in both Deepdoctection and Table Transformer approaches.
cfg.SEGMENTATION.TABLE_NAME = LayoutType.TABLE

# Lists the layout or cell types used in the original Deepdoctection approach.
# Used by TableSegmentationService for cell assignments.
cfg.SEGMENTATION.CELL_NAMES = [CellType.HEADER, CellType.BODY, LayoutType.CELL]

# Lists all cell types used by the Table Transformer approach (PubtablesSegmentationService).
# LayoutType.CELL is synthetically generated and not predicted by the structure recognition model.
cfg.SEGMENTATION.PUBTABLES_CELL_NAMES = [
    CellType.SPANNING,
    CellType.ROW_HEADER,
    CellType.COLUMN_HEADER,
    CellType.PROJECTED_ROW_HEADER,
    LayoutType.CELL,
]

# Subset of PUBTABLES_CELL_NAMES that represent spanning/header cells.
# These need to be matched with row or column elements.
cfg.SEGMENTATION.PUBTABLES_SPANNING_CELL_NAMES = [
    CellType.SPANNING,
    CellType.ROW_HEADER,
    CellType.COLUMN_HEADER,
    CellType.PROJECTED_ROW_HEADER,
]

# Lists the layout categories used to identify row and column elements.
# Used by TableSegmentationService.
cfg.SEGMENTATION.ITEM_NAMES = [LayoutType.ROW, LayoutType.COLUMN]

# Equivalent to ITEM_NAMES but used in the Table Transformer approach.
cfg.SEGMENTATION.PUBTABLES_ITEM_NAMES = [LayoutType.ROW, LayoutType.COLUMN]

# Used in TableSegmentationService to specify sub-category annotations for row and column numbers.
cfg.SEGMENTATION.SUB_ITEM_NAMES = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]

# Equivalent to SUB_ITEM_NAMES, but used with the Table Transformer approach.
cfg.SEGMENTATION.PUBTABLES_SUB_ITEM_NAMES = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]

# Used in PubtablesSegmentationService.
# Specifies which cells should be treated as header cells that need to be linked to row/column elements.
cfg.SEGMENTATION.PUBTABLES_ITEM_HEADER_CELL_NAMES = [CellType.COLUMN_HEADER, CellType.ROW_HEADER]

# Defines the threshold values for matching column/row header cells to their respective rows/columns
# in the Table Transformer approach. The matching rule is defined in SEGMENTATION.ASSIGNMENT_RULE.
cfg.SEGMENTATION.PUBTABLES_ITEM_HEADER_THRESHOLDS = [0.6, 0.0001]

# Configuration options for PDF text extraction using PDFPlumber.
# These values are passed directly to pdfplumber.utils.extract_words().
# For reference, see:
# https://github.com/jsvine/pdfplumber/blob/main/pdfplumber/utils/text.py

# Horizontal tolerance when merging characters into words.
# Characters that are horizontally closer than this value will be grouped into a single word.
cfg.PDF_MINER.X_TOLERANCE = 3

# Vertical tolerance when grouping characters into lines.
# Characters within this vertical range will be considered part of the same line.
cfg.PDF_MINER.Y_TOLERANCE = 3

# OCR engine selection.
# If cfg.USE_OCR = True, then one of the following must be set to True:
# - cfg.OCR.USE_TESSERACT
# - cfg.OCR.USE_DOCTR
# - cfg.OCR.USE_TEXTRACT
# All other engines must be set to False.

# Enables Tesseract as the OCR engine.
# Note: Tesseract must be installed separately. This integration does not use pytesseract.
# Configuration options are defined in a separate file: conf_tesseract.yaml.
cfg.OCR.USE_TESSERACT = False

# Path to the Tesseract configuration file.
cfg.OCR.CONFIG.TESSERACT = "dd/conf_tesseract.yaml"

# Enables DocTR as the OCR engine.
# DocTR provides flexible and lightweight OCR models with strong accuracy and versatility.
cfg.OCR.USE_DOCTR = True

# Enables AWS Textract as the OCR engine.
# Requires the following environment variables to be set:
# AWS_ACCESS_KEY, AWS_SECRET_KEY, and AWS_REGION.
# Alternatively, AWS credentials can be configured via the AWS CLI.
cfg.OCR.USE_TEXTRACT = False

# DocTR OCR uses a two-stage process: word detection followed by text recognition.
# The following weights configure each stage for TensorFlow and PyTorch.

# TensorFlow weights for the word detection model used by DocTR.
cfg.OCR.WEIGHTS.DOCTR_WORD.TF = "doctr/db_resnet50/tf/db_resnet50-adcafc63.zip"

# PyTorch weights for the word detection model used by DocTR.
cfg.OCR.WEIGHTS.DOCTR_WORD.PT = "doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt"

# TensorFlow weights for the text recognition model used by DocTR.
cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.TF = "doctr/crnn_vgg16_bn/tf/crnn_vgg16_bn-76b7f2c6.zip"

# PyTorch weights for the text recognition model used by DocTR.
cfg.OCR.WEIGHTS.DOCTR_RECOGNITION.PT = "doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"

# Specifies the annotation type used as a text container.
# A text container is typically an ImageAnnotation generated by the OCR engine or PDF mining tool.
# It contains a sub-annotation of type WordType.CHARACTERS.
# Most commonly, text containers are of type LayoutType.WORD, but LayoutType.LINE may also be used.
# It is recommended to align this value with IMAGE_DEFAULTS.TEXT_CONTAINER
# rather than modifying it directly in the config.
cfg.TEXT_CONTAINER = IMAGE_DEFAULTS.TEXT_CONTAINER

# Configuration for matching text containers (e.g., words or lines) to layout elements
# such as titles, paragraphs, tables, etc., using spatial overlap.
# When a match occurs, a parent-child relationship (Relationships.CHILD) is assigned.

# Specifies the layout categories considered as potential parents of text containers.
cfg.WORD_MATCHING.PARENTAL_CATEGORIES = IMAGE_DEFAULTS.TEXT_BLOCK_CATEGORIES

# Rule used for matching: either 'iou' (intersection over union) or 'ioa' (intersection over area).
cfg.WORD_MATCHING.RULE = "ioa"

# Threshold for the selected matching rule (IOU or IOA).
# Text containers must exceed this threshold to be assigned to a layout section.
cfg.WORD_MATCHING.THRESHOLD = 0.3

# If a text container overlaps with multiple layout sections,
# setting this to True will assign it only to the best-matching (i.e., highest-overlapping) section.
# Prevents duplication of text in the output.
cfg.WORD_MATCHING.MAX_PARENT_ONLY = True

# Specifies which layout categories must be ordered (e.g., paragraphs, list items).
# These are layout blocks that will be processed by the TextOrderingService.
cfg.TEXT_ORDERING.TEXT_BLOCK_CATEGORIES = IMAGE_DEFAULTS.TEXT_BLOCK_CATEGORIES

# Specifies which text blocks are considered floating (not aligned with strict columns or grids).
# These will be linked with a subcategory of type Relationships.READING_ORDER.
cfg.TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES = IMAGE_DEFAULTS.FLOATING_TEXT_BLOCK_CATEGORIES

# Determines whether residual (unmatched) text containers should be included in the ordering process.
# If set to True, orphaned text containers are grouped into lines and added to the layout ordering.
# If set to False, unmatched text containers will not appear in the output.
cfg.TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER = True

# Tolerance used to determine whether a text block's left/right coordinate lies within a column’s boundary.
# Helps with assigning text blocks to columns based on horizontal alignment.
cfg.TEXT_ORDERING.STARTING_POINT_TOLERANCE = 0.005

# Horizontal distance threshold for grouping words into the same line.
# If the gap between words exceeds this value, they will be treated as belonging to separate lines or columns.
cfg.TEXT_ORDERING.BROKEN_LINE_TOLERANCE = 0.003

# Used for ordering vertically broken floating text blocks into coherent columns.
# Defines vertical alignment tolerance between adjacent text blocks.
cfg.TEXT_ORDERING.HEIGHT_TOLERANCE = 2.0

# Defines the spacing threshold that indicates a paragraph break in vertically arranged text blocks.
# Helps determine reading order in multi-column, broken layouts.
cfg.TEXT_ORDERING.PARAGRAPH_BREAK = 0.035

# Configuration for linking spatially related layout sections
# (e.g., associating figures with their captions) based on proximity.
# The distance is calculated using the center points of the layout elements.

# Specifies the parent layout categories in the link relationship.
# These are the elements to which related components (e.g., captions) should be linked.
cfg.LAYOUT_LINK.PARENTAL_CATEGORIES = [LayoutType.FIGURE,LayoutType.TABLE]

# Specifies the child layout categories in the link relationship.
# These are typically smaller or subordinate elements (e.g., captions).
cfg.LAYOUT_LINK.CHILD_CATEGORIES = [LayoutType.CAPTION]

# Freezes the configuration to make it immutable.
# This prevents accidental modification at runtime.
cfg.freeze()
