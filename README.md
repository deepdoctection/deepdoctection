
<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/notebooks/pics/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  A Document AI Package
  </h3>
</p>


**deep**doctection is a Python package that enables document analysis pipelines to be built using deep learning models.

Extracting information from documents is difficult. Documents often have a complex visual structure and the information 
they contain is not tagged. **deep**doctection is a tool box that is intended to facilitate entry into this topic. 

Parse your document by detecting layout structures like tables with full table semantics (cells, rows, columns), 
get text in reading order with OCR, detect language and do many other things.

The focus should be on application. **deep**doctection is made for data scientists who are tasked with supporting
departments in process optimization or for analysts who have to investigate into large sets of documents.

For further text processing tasks, use one of the many other great NLP libraries.

![image info](./notebooks/pics/dd_rm_sample.png)

## Characteristics

1. Use an **off the shelf analyzer** for restructuring your **PDF** or **scanned documents**:
         
   - Layout recognition with deep neural networks from well renowned open source libraries (Cascade-RCNN from 
     Tensorpack or Detectron2) trained on large public datasets. Tensorflow or PyTorch models available. 
   - Table extraction with full table semantics (rows, columns, multi line cell spans), again with help of Cascade-RCNN
   - OCR or text mining with  [Tesseract](https://github.com/tesseract-ocr/tesseract), 
     [DocTr](https://github.com/mindee/doctr), [pdfplumber](https://github.com/jsvine/pdfplumber) or other
   - reading order
   - language detection with [fastText](https://github.com/facebookresearch/fastText)
   - parsed output available as JSON object for further NLP tasks, labeling or reviewing

Off the shelf actually means off the shelf. The results will look okay, but useful outputs for downstream tasks will 
only come out when models are adapted to actual documents you deal with. Therefore:

2. **Fine-tune pre-trained DNN** on your own labeled dataset. Use generally acknowledged metrics for evaluating training
    improvements. Training scripts available.


3. **Compose your document analyzer** by choosing a model and plug it into your own pipeline. For example, you can use
    pdfplumber if you have native PDF documents. Or you can benchmark OCR results with AWS Textract (account needed and 
    paid service).


5. Wrap DNNs from open source projects into the **deep**doctections API and **enrich your pipeline easily with SOTA 
   models**.


6. All models are now available at the :hugs: [**Huggingface Model Hub**](https://huggingface.co/deepdoctection) .
You can acquire more details in the respective model cards.

Check [**this notebook**](./notebooks/Get_Started.ipynb) for an easy start, as  well as the full
[**documentation**](https://deepdoctection.readthedocs.io/en/latest/index.html#).

## Requirements

### Platform and Python

Before you start, please ensure your installation fulfills the following baseline requirements:

- Linux **or** macOS
- Python >=  3.8 
- PyTorch >= 1.8 and torchvision **or** Tensorflow >=2.4.1 and CUDA

Windows is not supported.

You can run on PyTorch with a CPU only. For Tensorflow a GPU is required.

### Other

**deep**doctection uses Python wrappers for [Poppler](https://poppler.freedesktop.org/) to convert PDF documents into 
images and for calling [Tesseract](https://github.com/tesseract-ocr/tesseract) OCR engine. 
If you get started and want to run the notebooks for the first time it is required to have them installed as well.

## Installation

We recommend using a virtual environment. You can install the package via pip or from source. Bug fixes or enhancements
will be deployed tp PyPi every 4 to 6 weeks.

### Install with pip from PyPi

[Dataflow](https://github.com/tensorpack/dataflow) is not available on the PyPi server and must be installed separately.

```
pip install  "dataflow @ git+https://github.com/tensorpack/dataflow.git"
```

Depending on which Deep Learning library is available, use the following installation option:

For **Tensorflow**, run

```
pip install deepdoctection[tf]
```

For **PyTorch**, first install **Detectron2** separately as it is not on the PyPi, either. Check the instruction 
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Then run

```
pip install deepdoctection[pt]
```

This will install the basic setup which is needed to run the first two notebooks and do some inference with pipelines.

Some libraries are not added to the requirements in order to keep the dependencies as small as possible (e.g. DocTr,
pdfplumber, fastText, ...). If you want to use them, you have to pip install them individually by yourself. 
Alternatively, consult the 
[**full installation instructions**](https://deepdoctection.readthedocs.io/en/latest/manual/install.html).


### Installation from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

To get started with **Tensorflow**, run:

```
cd deepdoctection
pip install ".[source-tf]"
```

or with **PyTorch**:
 
```
cd deepdoctection
pip install ".[source-pt]"
```

This will install the basic dependencies to get started with the first notebooks. To get all package extensions,

```
cd deepdoctection
pip install ".[source-all-tf]"
```

or 

```
cd deepdoctection
pip install ".[source-all-pt]"
```

will install all available external libraries that can be used for inference (e.g. DocTr, pdfplumber, fastText, ...).

For more installation options check [**this**](https://deepdoctection.readthedocs.io/en/latest/manual/install.html) site.


## Credits

Many utils, concepts and some models are inspired and taken from [**Tensorpack**](https://github.com/tensorpack) . 
We heavily make use of [Dataflow](https://github.com/tensorpack/dataflow) for loading and streaming data.  


## Problems

We try hard to eliminate bugs. We also know that the code is not free of issues. We welcome all issues relevant to this
repo and try to address them as quickly as possible.


## Citing **deep**doctection

If you use **deep**doctection in your research or in your project, please cite:

```
@misc{jmdeepdoctection,
  title={deepdoctection},
  author={Meyer, Dr. Janis and others},
  howpublished={\url{https://github.com/deepdoctection/deepdoctection}},
  year={2021}
}
```


## License

Distributed under the Apache 2.0 License. Check [LICENSE](https://github.com/deepdoctection/deepdoctection/blob/master/LICENSE) 
for additional information.
