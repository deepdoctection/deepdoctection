
<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  A Document AI Package
  </h3>
</p>


**deep**doctection is a Python library that orchestrates document extraction and document layout analysis tasks using deep learning models. It does 
not implement models but enables you to build pipelines using highly acknowledged libraries for object detection, OCR 
and selected NLP tasks and provides an integrated framework for fine-tuning, evaluating and running models. For more
 specific text processing tasks use one of the many other great NLP libraries.

**deep**doctection focuses on applications and is made for those who want to solve real world problems related to 
document extraction from PDFs or scans in various image formats.

Check the demo of a document layout analysis pipeline with OCR on 
:hugs: [**Hugging Face spaces**](https://huggingface.co/spaces/deepdoctection/deepdoctection).

# Overview

**deep**doctection provides model wrappers of supported libraries for various tasks to be integrated into 
pipelines. Its core function does not depend on any specific deep learning library. Selected models for the following 
 tasks are currently supported:       

 - Document layout analysis including table recognition in Tensorflow with [**Tensorpack**](https://github.com/tensorpack), 
   or PyTorch with [**Detectron2**](https://github.com/facebookresearch/detectron2/tree/main/detectron2),
 - OCR with support of [**Tesseract**](https://github.com/tesseract-ocr/tesseract), [**DocTr**](https://github.com/mindee/doctr)
   (Tensorflow and PyTorch implementations available) and a wrapper to an API for a commercial solution, 
 - Text mining for native PDFs with  [**pdfplumber**](https://github.com/jsvine/pdfplumber), 
 - Language detection with [**fastText**](https://github.com/facebookresearch/fastText),
 - Deskewing and rotating images with [**jdeskew**](https://github.com/phamquiluan/jdeskew). 
 - Document and token classification with all LayoutLM models provided by the 
   [**Transformer library**](https://github.com/huggingface/transformers). 
   (Yes, you can use any LayoutLM-model with any of the provided OCR-or pdfplumber tools straight away!).
 - Table detection and table structure recognition with 
   [**table-transformer**](https://github.com/microsoft/table-transformer). 
 - There is a small dataset for token classification [available](https://huggingface.co/datasets/deepdoctection/FRFPE)
   and a lot of new [tutorials](https://github.com/deepdoctection/notebooks/blob/main/Layoutlm_v2_on_custom_token_classification.ipynb) 
   to show, how to train and evaluate this dataset using LayoutLMv1, LayoutLMv2, LayoutXLM and LayoutLMv3.
 - Comprehensive configuration of **analyzer** like choosing different models, output parsing, OCR selection.
   Check this [notebook](https://github.com/deepdoctection/notebooks/blob/main/Analyzer_Configuration.ipynb) or the 
   [docs](https://deepdoctection.readthedocs.io/en/latest/tutorials/analyzer_configuration_notebook/) for more infos.
 - Document layout analysis and table recognition now runs with 
   [**Torchscript**](https://pytorch.org/docs/stable/jit.html) (CPU) as well and [**Detectron2**](https://github.com/facebookresearch/detectron2/tree/main/detectron2) is not required 
   anymore for basic inference.
 - More angle predictors for determining the rotation of a document based on [**Tesseract**](https://github.com/tesseract-ocr/tesseract) and [**DocTr**](https://github.com/mindee/doctr)
 - Token classification with [**LiLT**](https://github.com/jpWang/LiLT) via 
   [**transformers**](https://github.com/huggingface/transformers). 
   We have added a model wrapper for token classification with LiLT and added a some LiLT models to the model catalog 
   that seem to look promising, especially if you want to train a model on non-english data. The training script for 
   LayoutLM can be used for LiLT as well.
 - [**new**] There are two notebooks available that show, how to write a 
   [custom predictor](https://github.com/deepdoctection/notebooks/blob/main/Doclaynet_Analyzer_Config.ipynb) based on 
   a third party library that has not been supported yet and how to use 
   [advanced configuration](https://github.com/deepdoctection/notebooks/blob/main/Doclaynet_Analyzer_Config.ipynb) to 
   get links between layout segments e.g. captions and tables or figures.

**deep**doctection provides on top of that methods for pre-processing inputs to models like cropping or resizing and to 
post-process results, like validating duplicate outputs, relating words to detected layout segments or ordering words 
into contiguous text. You will get an output in JSON format that you can customize even further by yourself. 
     
Have a look at the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb) in the 
[notebook repo](https://github.com/deepdoctection/notebooks) for an easy start.

Check the [**release notes**](https://github.com/deepdoctection/deepdoctection/releases) for recent updates.

## Models    

**deep**doctection or its support libraries provide pre-trained models that are in most of the cases available at the 
[**Hugging Face Model Hub**](https://huggingface.co/deepdoctection) or that will be automatically downloaded once 
requested. For instance, you can find pre-trained object detection models from the Tensorpack or Detectron2 framework
 for coarse layout analysis, table cell detection and table recognition. 

## Datasets and training scripts

Training is a substantial part to get pipelines ready on some specific domain, let it be document layout analysis, 
document classification or NER. **deep**doctection provides training scripts for models that are based on trainers
developed from the library that hosts the model code. Moreover, **deep**doctection hosts code to some well established 
datasets like **Publaynet** that makes it easy to experiment. It also contains mappings from widely used data 
formats like COCO and it has a dataset framework (akin to [**datasets**](https://github.com/huggingface/datasets) so that
 setting up training on a custom dataset becomes very easy. [**This notebook**](https://github.com/deepdoctection/notebooks/blob/main/Datasets_and_Eval.ipynb)
shows you how to do this.
   
## Evaluation

**deep**doctection comes equipped with a framework that allows you to evaluate predictions of a single or multiple 
models in a pipeline against some ground truth. Check again [**here**](https://github.com/deepdoctection/notebooks/blob/main/Datasets_and_Eval.ipynb) how it is 
done.  

## Inference

Having set up a pipeline it takes you a few lines of code to instantiate the pipeline and after a for loop all pages will 
be processed through the pipeline. 

```python
import deepdoctection as dd
from IPython.core.display import HTML
from matplotlib import pyplot as plt

analyzer = dd.get_dd_analyzer()  # instantiate the built-in analyzer similar to the Hugging Face space demo

df = analyzer.analyze(path = "/path/to/your/doc.pdf")  # setting up pipeline
df.reset_state()                 # Trigger some initialization

doc = iter(df)
page = next(doc) 

image = page.viz()
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
```

![text](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_sample.png)

```
HTML(page.tables[0].html)
```

![table](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_table.png)


```
print(page.text)
```

![table](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_text.png)
 

## Documentation

There is an extensive [**documentation**](https://deepdoctection.readthedocs.io/en/latest/index.html#) available 
containing tutorials, design concepts and the API. We want to present things as comprehensively and understandably 
as possible. However, we are aware that there are still many areas where significant improvements can be made in terms 
of clarity, grammar and correctness. We look forward to every hint and comment that increases the quality of the 
documentation.


## Requirements

![requirements](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/requirements_deepdoctection_081124.png)

Everything in the overview listed below the **deep**doctection layer are necessary requirements and have to be installed 
separately. 

- Linux or macOS. (Windows is not supported but there is a [Dockerfile](./docker/pytorch-cpu-jupyter/Dockerfile) available)
- Python >= 3.9
- 1.13 <= PyTorch  **or** 2.11 <= Tensorflow < 2.16. (For lower Tensorflow versions the code will only run on a GPU).
In general, if you want to train or fine-tune models, a GPU is required.

- With respect to the Deep Learning framework, you must decide between [Tensorflow](https://www.tensorflow.org/install?hl=en)
  and [PyTorch](https://pytorch.org/get-started/locally/).
- [Tesseract](https://github.com/tesseract-ocr/tesseract) OCR engine will be used through a Python wrapper. The core 
  engine has to be installed separately.


- For release `v.0.34.0` and below **deep**doctection uses Python wrappers for [Poppler](https://poppler.freedesktop.org/) to convert PDF
  documents into images. For release `v.0.35.0` this dependency will be optional.

The following overview shows the availability of the models in conjunction with the DL framework.

| Task                                          | PyTorch | Torchscript    |  Tensorflow  |
|-----------------------------------------------|:-------:|----------------|:------------:|
| Layout detection via Detectron2/Tensorpack    |    ✅    | ✅ (CPU only)   | ✅ (GPU only) |
| Table recognition via Detectron2/Tensorpack   |    ✅    | ✅ (CPU only)   | ✅ (GPU only) |
| Table transformer via Transformers            |    ✅    | ❌              |      ❌       |
| DocTr                                         |    ✅    | ❌              |      ✅       |
| LayoutLM (v1, v2, v3, XLM) via Transformers   |    ✅    | ❌              | ❌            |



## Installation

We recommend using a virtual environment. You can install the package via pip or from source. 

### Install with pip from PyPi

#### Minimal installation 

If you want to get started with a minimal setting (e.g. running the **deep**doctection analyzer with 
default configuration or trying the 'Get started notebook'), install **deep**doctection with

```
pip install deepdoctection
```

If you want to use the Tensorflow framework, please install Tensorpack separately. Detectron2 will not be installed 
and layout models/ table recognition models will run with Torchscript on a CPU.

#### Full installation

The following installation will give you ALL models available within the Deep Learning framework as well as all models
that are independent of Tensorflow/PyTorch. Please note, that the dependencies are very complex. We try hard to keep 
the requirements up to date though.

For **Tensorflow**, run

```
pip install deepdoctection[tf]
```

For **PyTorch**, 

first install **Detectron2** separately as it is not distributed via PyPi. Check the instruction 
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Then run

```
pip install deepdoctection[pt]
```

This will install **deep**doctection with all dependencies listed above the **deep**doctection layer. Use this setting, 
if you want to get started or want to explore all features. 

If you want to have more control with your installation and are looking for fewer dependencies then 
install **deep**doctection with the basic setup only.

```
pip install deepdoctection
```

This will ignore all model libraries (layers above the **deep**doctection layer in the diagram) and you 
will be responsible to install them by yourself. Note, that you will not be able to run any pipeline with this setup.

For further information, please consult the [**full installation instructions**](https://deepdoctection.readthedocs.io/en/latest/install/).


### Installation from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

To get started with **Tensorflow**, run:

```
cd deepdoctection
pip install ".[tf]"
```

Installing the full **PyTorch** setup from source will also install **Detectron2** for you:
 
```
cd deepdoctection
pip install ".[source-pt]"
```

### Running a Docker container from Docker hub

Starting from release `v.0.27.0`, pre-existing Docker images can be downloaded from the 
[Docker hub](https://hub.docker.com/r/deepdoctection/deepdoctection).

```
docker pull deepdoctection/deepdoctection:<release_tag> 
```

To start the container, you can use the Docker compose file `./docker/pytorch-gpu/docker-compose.yaml`. 
In the `.env` file provided, specify the host directory where **deep**doctection's cache should be stored. 
This directory will be mounted. Additionally, specify a working directory to mount files to be processed into the 
container.

```
docker compose up -d
```

will start the container.

## Credits

We thank all libraries that provide high quality code and pre-trained models. Without, it would have been impossible 
to develop this framework.

## Problems

We try hard to eliminate bugs. We also know that the code is not free of issues. We welcome all issues relevant to this
repo and try to address them as quickly as possible. Bug fixes or enhancements will be deployed in a new release every 10 
to 12 weeks.

## If you like **deep**doctection ...
 
 ...you can easily support the project by making it more visible. Leaving a star or a recommendation will help. 


## License

Distributed under the Apache 2.0 License. Check [LICENSE](https://github.com/deepdoctection/deepdoctection/blob/master/LICENSE) 
for additional information.
