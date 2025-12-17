<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
</p>


# deepdoctection

**deepdoctection** is the main package for running and training models. It provides the
pipeline framework, model wrappers, built-in pipelines, training scripts and evaluation methods.

The base package only installs the necessary dependencies for running inference with some selected models. 
For training, evaluating as well as running all available models, the full package needs to be installed. 

## Overview

- **analyzer**: Configuration and factory functions for creating document analysis pipelines and the built-in analyzer.
- **configs**: YAML configuration for pipelines and model profiles for the model catalogue.
- **extern**: External model wrappers (Detectron2, DocTr, HuggingFace Transformers, Tesseract, PdfPlumber, etc.)
- **pipe**: Pipeline components and services.
- **eval**: Evaluation metrics and Evaluator.
- **train**: Training utilities and training scripts for Detectron2 and selected Transformer models.


## Installation

### Basic Installation

For inference use cases, install the base package:

```bash
(uv) pip install deepdoctection
```

**Important**: Various dependencies must be installed separately:

- **PyTorch**: Follow instructions at https://pytorch.org/get-started/locally/ according to your os and hardware.
- **Transformers**: `pip install transformers>=4.48.0` (if using HF models)
- **Timm**: `pip install timm>=0.9.16` (necessary for if using some dedicated HF models)
- **DocTr**: `pip install python-doctr>=1.0.0` (if using DocTr models)
- **Detectron2**: Follow instructions at https://detectron2.readthedocs.io/en/latest/tutorials/install.html
- **PDFPlumber**: `pip install pdfplumber>=0.11.0`
- **JDeskew**: `pip install jdeskew>=0.2.2`
- **Boto3**: `pip install boto3==1.34.102`

For running evaluation with various metrics you can also install in then use:

- **APTED**: `pip install apted==1.0.3`
- **Distance**: `pip install distance==0.1.3`
- **Pycocotools**: `pip install pycocotools>=2.0.2`

Image processing is supported by PIL or OpenCV. PIL is used by default and will always be installed. If 
you prefer to use OpenCV, you can install it:

- **OpenCV**: `pip install opencv-python==4.8.0.76`


### Full Installation (Training & Evaluation)

For a one large install with all dependencies (except PyTorch), run:

```bash
(uv) pip install deepdoctection[full]
```

### Development Installation

For development purpose use clone the repository and install in editable mode.

## License

Apache License 2.0

## Author

Dr. Janis Meyer