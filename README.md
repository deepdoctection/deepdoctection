<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
</p>

![GitHub Repo stars](https://img.shields.io/github/stars/deepdoctection/deepdoctection)
![PyPI - Version](https://img.shields.io/pypi/v/deepdoctection)
![PyPI - License](https://img.shields.io/pypi/l/deepdoctection)


------------------------------------------------------------------------------------------------------------------------
# NEW 

Version `v.1.0` includes a major refactoring.  Key changes include:

* PyTorch-only support for all deep learning models.
* Decomposition into small sub-packages: dd-core, dd-datasets and deepdoctection
* Type validations of core data structures
* New test suite

------------------------------------------------------------------------------------------------------------------------

<p align="center">
  <h1 align="center">
  A Package for Document Understanding
  </h1>
</p>


**deep**doctection is a Python library that orchestrates Scan and PDF document layout analysis, OCR and document 
and token classification. Build and run a pipeline for your document extraction tasks, devlop your own document
extraction workflow and use pre-trained models for inference.

It also provides a framework for training, evaluating and inferencing Document AI models.

# Overview

- Document layout analysis and table recognition in PyTorch with 
[**Detectron2**](https://github.com/facebookresearch/detectron2/tree/main/detectron2) and 
[**Transformers**](https://github.com/huggingface/transformers),
- OCR with support of [**Tesseract**](https://github.com/tesseract-ocr/tesseract), [**DocTr**](https://github.com/mindee/doctr) and 
  [**AWS Textract**](https://aws.amazon.com/textract/),
- Document and token classification with the [**LayoutLM**](https://github.com/microsoft/unilm) family,
  [**LiLT**](https://github.com/jpWang/LiLT) and selected
  [**Bert**](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)-style including features like sliding windows.
- Text mining for native PDFs with [**pdfplumber**](https://github.com/jsvine/pdfplumber),
- Language detection with with transformer based `papluca/xlm-roberta-base-language-detection`. 
- Deskewing and rotating images with [**jdeskew**](https://github.com/phamquiluan/jdeskew) or [**Tesseract**](https://github.com/tesseract-ocr/tesseract).
- Fine-tuning object detection, document or token classification models and evaluating whole pipelines.
- Lot's of [tutorials](https://github.com/deepdoctection/notebooks)

Have a look at the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Analyzer_Get_Started.ipynb) for an easy start.

Check the [**release notes**](https://github.com/deepdoctection/deepdoctection/releases) for recent updates.

----------------------------------------------------------------------------------------

# Hugging Face Space Demo

Check the demo of a document layout analysis pipeline with OCR on 🤗
[**Hugging Face spaces**](https://huggingface.co/spaces/deepdoctection/deepdoctection).

--------------------------------------------------------------------------------------------------------

# Example

The following example shows how to use the built-in analyzer to decompose a PDF document into its layout structures.

```python
import deepdoctection as dd
from IPython.core.display import HTML
from matplotlib import pyplot as plt

analyzer = dd.get_dd_analyzer()  # instantiate the built-in analyzer similar to the Hugging Face space demo

df = analyzer.analyze(path = "/path/to/your/doc.pdf")  # setting up pipeline
df.reset_state()                 # Trigger some initialization

doc = iter(df)
page = next(doc) 

image = page.viz(show_figures=True, show_residual_layouts=True)
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
```

<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_sample.png" 
alt="sample" width="40%">
</p>

```
HTML(page.tables[0].html)
```

<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_table.png" 
alt="table" width="40%">
</p>

```
print(page.text)
```

<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_text.png" 
alt="text" width="40%">
</p>


-----------------------------------------------------------------------------------------

# Requirements

![requirements](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/install_01.png)

- Python >= 3.10
- PyTorch >= 2.6
- To fine-tune models, a GPU is recommended.

| Task | PyTorch | Torchscript |
|---------------------------------------------|:-------:|----------------|
| Layout detection via Detectron2 | ✅ | ✅ (CPU only) |
| Table recognition via Detectron2 | ✅ | ✅ (CPU only) |
| Table transformer via Transformers | ✅ | ❌ |
| Deformable-Detr | ✅ | ❌ |
| DocTr | ✅ | ❌ |
| LayoutLM (v1, v2, v3, XLM) via Transformers | ✅ | ❌ |

------------------------------------------------------------------------------------------

# Installation

We recommend using a virtual environment.

## Get started installation

For a simple setup which is enough to parse documents with the default setting, install the following

```
pip install timm  # needed for the default setup
pip install transformers
pip install python-doctr
pip install deepdoctection
```

This setup is sufficient to run the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb).

### Full installation

The following installation will give you a general setup so that you can experiment with various configurations.
Remember, that you always have to install PyTorch separately.

First install **Detectron2** separately as it is not distributed via PyPi. Check the instruction
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) or try:

```
pip install --no-build-isolation detectron2@git+https://github.com/deepdoctection/detectron2.git
```

Then install **deep**doctection with all its dependencies:

```
pip install deepdoctection[full]
```


For further information, please consult the [**full installation instructions**](https://deepdoctection.readthedocs.io/en/latest/install/).


## Installation from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

The easiest way is to install with make. A virtual environment is required

```bash
make install-dd
```


## Running a Docker container from Docker hub

Pre-existing Docker images can be downloaded from the [Docker hub](https://hub.docker.com/r/deepdoctection/deepdoctection).

Additionally, specify a working directory to mount files to be processed into the container.

```
docker compose up -d
```

will start the container. There is no endpoint exposed, though.

-----------------------------------------------------------------------------------------------

# Credits

We thank all libraries that provide high quality code and pre-trained models. Without, it would have been impossible
to develop this framework.


# If you like **deep**doctection ...

...you can easily support the project by making it more visible. Leaving a star or a recommendation will help.

# License

Distributed under the Apache 2.0 License. Check [LICENSE](https://github.com/deepdoctection/deepdoctection/blob/master/LICENSE) for additional information.
