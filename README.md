
<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/notebooks/pics/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  A Document AI Package
  </h3>
</p>


**deep**doctection is a Python package that enables document analysis pipelines to be built using deep learning models.

Extracting information from documents is difficult. Documents often have a complex visual structure and the information 
they contain is not tagged. **deep**doctection is a tool box that is intended to facilitate entry into this topic. 

The focus should be on application. **deep**doctection is made for data scientists who are tasked with supporting
departments in process optimization. For analysts who have to investigate into large sets of documents. And also maybe 
for researchers who would like to see how well their new model fits into an extraction pipeline.

It currently focuses on raw text extraction. For further text processing tasks, use one of the many other great NLP 
libraries.

![image info](./notebooks/pics/dd_rm_sample.png)

## Characteristics

1. Use an **off the shelf analyzer** for restructuring your **PDF** or **scanned documents**:
         
   - layout recognition with deep neural networks (Mask-RCNN and more) trained on large public datasets
   - table extraction with full table semantics (rows, columns, multi line cell spans), again with DNN
   - OCR and word assignment to detected layouts components
   - reading order 

   Off the shelf actually means off the shelf. The results will look okay, but useful outputs for downstream tasks will 
   only come out when models are adapted to actual documents you deal with. Therefore:

2. **Fine-tune pre-trained DNN** on your own labeled dataset. Use generally acknowledged metrics for evaluating training
    improvements.

3. **Compose your document analyzer** by choosing a model and plug it into your own pipeline.

4. Wrap DNNs from open source projects into the **deep**doctections API and **enrich your pipeline easily with SOTA 
   models**.

5. All models are now available at the :hugs: [**Huggingface Model Hub**](https://huggingface.co/deepdoctection) .
You can acquire more details in the respective model cards.

Check [**this notebook**](./notebooks/Get_Started.ipynb) for an easy start, as  well as the full
[**documentation**](https://deepdoctection.readthedocs.io/en/latest/index.html#).

## Requirements

- Linux **or** macOS
- Python >=  3.8
- PyTorch >= 1.8 **or** Tensorflow >=2.4.1 and CUDA

You can run on PyTorch with a CPU only. For Tensorflow a GPU is required. 

**deep**doctection uses Tensorpack as training framework as well as its vision models for layout analysis. 
For PyTorch, Detectron2 is used. All models have been trained on Tensorflow and converted into Detectron2 consumable 
artefacts. Prediction results in PyTorch are therefore slightly worse. 

If you do not work on Linux, one easy way to fulfill the requirements is to use the Docker image. A 
[Dockerfile](./docker/TF/Dockerfile) is provided, please follow the official instructions on how to use it. 

Depending on the pipeline you want to use, you will be notified if further installations are necessary, e.g.

- [Tesseract](https://github.com/tesseract-ocr/tesseract) 
- [Poppler](https://poppler.freedesktop.org/)


## Installation

We recommend using a virtual environment. You can install the package via pip or from source.

### Install with pip

Since dataflow is not available via pip, it must be installed separately.

```
pip install  "dataflow @ git+https://github.com/tensorpack/dataflow.git"
```

Depending on which Deep Learning is available, use the following installation option:

For Tensorflow, run

```
pip install deepdoctection[tf]
```

For PyTorch, 

first install Detectron2 separately. Check the instruction [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
Then run

```
pip install deepdoctection[pt]
```

### Installation from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

There is a **Makefile** that guides you though the installation process. To get started, try:

```
cd deepdoctection
make clean
make venv
source venv/bin/activate
```

For Tensorflow, run
 
```
make install-dd-tf
```

If you want to use the PyTorch framework, run:

```
make install-dd-pt
```

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
