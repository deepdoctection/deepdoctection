# Analyzer Configuration

Document extraction from PDFs and scans is nothing that can be done in only one way. On the contrary, one needs 
flexibility in the extraction process and this flexibility will be provided by an extensive customization of 
**deep**doctection analyzer. 

We will now discuss the most important configuration options. We will assume familiarity with the 
[Get started tutorial](./get_started_notebook.md) as well as the [pipeline tutorial](./pipelines_notebook.md). 

## How to change configuration

Configuration can be done either directly in the `.yaml` file or by explicitly overwriting the default variables. 


```python
import deepdoctection as dd

config_overwrite = ["USE_TABLE_SEGMENTATION=False","USE_OCR=False"]

analyzer = dd.get_dd_analyzer(config_overwrite=config_overwrite)
```

## High level Configuration

The analyzer consists of four steps: Layout detection, table recognition, text extraction and recombination. You can 
switch each block on and off:

```yaml
USE_LAYOUT: True

USE_TABLE_SEGMENTATION: True   # this will determine cells and rows/column of a detected table
USE_TABLE_REFINEMENT: True     # this will guarantee that the table structure can be expressed in a valid html-format

USE_PDF_MINER: False           # when processing a PDF file, it will first try to extract words using pdfplumber 
USE_OCR: True                  # using one of the three available OCR tools to be specified later  
```

## Layout models

Layout detection uses either Tensorpack's Cascade-RCNN or Detectron2 Cascade-RCNN by default, depending on which DL 
framework PyTorch or Tensorflow has been installed. The model have been trained on Publaynet and therefore provide 
detection of one of the following segments: `text, title, table, figure, list`. You can filter any of the segments, e.g.

```yaml
PT:
   LAYOUT:
      WEIGHTS: layout/d2_model_0829999_layout_inf_only.pt
      FILTER:
         - figure
```

### Table transformer

You can use the [table transformer model](https://github.com/microsoft/table-transformer) for table detection. 


```yaml
PT:
   LAYOUT:
      WEIGHTS: microsoft/table-transformer-detection/pytorch_model.bin
   PAD:
      TOP: 60
      RIGHT: 60
      BOTTOM: 60
      LEFT": 60

```

Table transformer requires image padding for more accurate results. The default padding provided might not be optimal. 
You can tweak and change it according to your needs.


### Custom model

A custom model can be added as well, but it needs to be registered. The same holds true for some special categories. 
We refer to [this tutorial](./running_pre_trained_models_from_third_party_libraries_notebook.md) for adding your own or 
third party models.

## Table segmentation

Table segmentation, e.g. the determination of cells, rows and columns as well as multi-span cells can be done in two 
different ways. 

*The original **deep**doctection process*: With one cell detector and one row-column detector each, the basis for the 
table structure is created. Then, by superimposing cells to rows and columns, the row and column number of the cell is 
determined. This admittedly rather short description should not hide the fact that this derivation consists of many 
intermediate steps. Row and column numbers must be stretched to completely overlay the table. If necessary, some 
overlapping rows and columns must be removed. An overlay rule of cells to rows and columns must be set, etc. 

*The Table Transformer process*: Here, rows and columns as well as multi-spanning cells are detected in only one 
detector. In contrast to the process proposed in the original repo, cells are simply detected by overlapping rows and 
columns. Then it is looked which of the simple cells can be replaced by multi-spanning cells. In a last step, the row 
and column number of a cell is determined. 

Table transformer is only available with PyTorch.

The default configuration uses the **deep**doctection process.

```yaml
SEGMENTATION:
  ASSIGNMENT_RULE: ioa  # iou is another cell/row/column overlapping rule
  THRESHOLD_ROWS: 0.4
  THRESHOLD_COLS: 0.4
  FULL_TABLE_TILING: True # in order to guarantee that the table is completely covered with rows and columns, resp.
  REMOVE_IOU_THRESHOLD_ROWS: 0.001
  REMOVE_IOU_THRESHOLD_COLS: 0.001
  STRETCH_RULE: equal  # how to stretch row/columns: left is another choice
```

As a rule of thumb for the configuration of the segmentation the following can be stated: The better the detectors work 
for the use case, the higher the thresholds should be chosen to leverage the results.

To perform table segmentation with Table transformer, the configuration must be set as follows:

```yaml
PT:
   ITEM:
     WEIGHTS: microsoft/table-transformer-structure-recognition/pytorch_model.bin
     FILTER:
        - table # model detects tables which are redundant and must be filtered
```

In our own experience, it has been shown that the recognition of multi spanning cells does not work reliably for tables 
that do not originate from medical articles. If you can do without determining multi-spanning cells/headers, it is 
recommended to filter them.

```yaml
PT:
   ITEM:
     WEIGHTS: microsoft/table-transformer-structure-recognition/pytorch_model.bin
     FILTER:
        - table # model detects tables which are redundant and must be filtered
        - column_header
        - projected_row_header
        - spanning
```

## Text extraction

There are four different options for text extraction.

### PDFPlumber

Extraction with pdfplumber. This requires native PDF documents where the text can be extracted from the byte encoding. 
Scans and text from images are not included here. 

```yaml
USE_PDF_MINER: True
```

The remaining three are all OCR methods.

It is possible to select PdfPlumber in combination with an OCR (exception: DocTr). If no text was extracted with 
PdfPlumber, the OCR service will be called, otherwise it will be omitted. There is currently not option to grap 
everything with OCR that cannot be extracted with PdfPlumber. It is all or nothing.  

```yaml
USE_PDF_MINER: True
USE_OCR: True
OCR:
  USE_TESSERACT: True
  USE_DOCTR: False
  USE_TEXTRACT: False
```

### Tesseract

Tesseract has its own configuration file `conf_tesseract.yaml`, which is also located in the `.cache`. Here you can 
enter all parameters that are also valid via the Tesseract CLI. We refer to Tesseract's documentation. 

### DocTr

DocTr is a powerful OCR library with different models. Only one model is currently registered for PyTorch/Tensorflow, 
but there are more pre-trained models that can also be used in this framework after registration. 

DocTr uses a textline detector and a text recognizer whose models can both be loaded. These are both included in the 
default configuration. 

```yaml
OCR:
  WEIGHTS:
    DOCTR_WORD:
      TF: doctr/db_resnet50/tf/db_resnet50-adcafc63.zip
      PT: doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt
    DOCTR_RECOGNITION:
      TF: doctr/crnn_vgg16_bn/tf/crnn_vgg16_bn-76b7f2c6.zip
      PT: doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt
```

### AWS Textract

Textract is the AWS OCR solution that can be accessed via an API. This is a paid service and requires an AWS account, 
installation of the AWS CLI, and a token. We refer to the official documentation to access the service via API.

## Word matching

We have already discussed word matching in the pipeline notebook, we can cover the main topics quickly here.

We determine which layout segments should be considered as candidates for matching words. A relation is then created 
using an overlap rule. Layout segments that are not listed are not available as candidates for a relation.

```yaml
WORD_MATCHING:
  PARENTAL_CATEGORIES:
    - text
    - title
    - list
    - cell  # Note, that there is no relationship between tables and words. The reason is, that we want to relate cells and words. 
  RULE: ioa  # choose iou otherwise
  THRESHOLD: 0.6
```

## Text ordering

We have also dealt with reading order in more detail in the pipeline notebook.

The layout sections that contain text and that need to be sorted need to be configured. If the section is listed in 
`TEXT_BLOCK_CATEGORIES`, the assigned words are run through the sorting algorithm.

The layout sections that are to be merged into continuous text across the entire page must also be configured. All sections listed in `FLOATING_TEXT_BLOCK_CATEGORIES` are taken into account.

The reading order is determined heuristically. The following parameters can be changed, but we refer to the API 
documentation for their meaning.

```yaml
TEXT_ORDERING:
  INCLUDE_RESIDUAL_TEXT_CONTAINER: False
  STARTING_POINT_TOLERANCE: 0.005
  BROKEN_LINE_TOLERANCE: 0.003
  HEIGHT_TOLERANCE: 2.0
  PARAGRAPH_BREAK: 0.035
```

Words that have not been assigned to a layout segment are initially no longer displayed. To avoid losing words or 
even phrases, set

```yaml
TEXT_ORDERING:
  INCLUDE_RESIDUAL_TEXT_CONTAINER: True
```

When doing this, words will be grouped into lines of synthetic text and the lines are treated and arranged as layout 
segments.
