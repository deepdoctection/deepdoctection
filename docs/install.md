# Installation


## Requirements

![](./tutorials/_imgs/requirements_deepdoctection_081124.png)

Everything in the overview listed below the **deep**doctection layer are necessary requirements and have to be installed 
by the user. 

- Linux **or** macOS. (Windows is not supported but there is [Dockerfile](../docker/pytorch-cpu-jupyter/Dockerfile) available)
- Python >=  3.9
- 1.13 <= PyTorch **or** 2.11 <= Tensorflow <2.16. On lower Tensorflow versions the code will only run inference on 
a GPU. In general, if you want to train or fine-tune models, a GPU is required.

For release `v.0.34.0` and below:

- [Poppler](https://poppler.freedesktop.org/)

is required for PDF processing. Starting from release `v.0.35.0`, the package `pypdfmium2` is used for PDF processing. 

With respect to the deep learning framework, you must decide between [Tensorflow](https://www.tensorflow.org/install?hl=en) 
and [PyTorch](https://pytorch.org/get-started/locally/).

We use [Pillow](https://pillow.readthedocs.io/en/stable/) or [OpenCV](https://github.com/opencv/opencv-python) for 
image processing tasks. PIL is more lightweight, easier to install and the default choice. 
OpenCV is faster when loading images and can be beneficial especially when training. If you want to use OpenCV, please
install this framework separately and set the environment variable `USE_OPENCV=True`. 

You can run on PyTorch with a CPU only. For Tensorflow a GPU is required.

If you only want to run some Tensorpack models, Tensorflow >= 2.4.1 will suffice.
Tensorpack has been developed for TF1, models however runs on TF2 as well by using tf.compat.v1. This is not ideal, 
however transferring all Tensorpack features into a TF2 framework will take a significant amount of work and will not be
the top priority.

The code has been tested on Ubuntu 20.04. Functions not involving a GPU have also been tested on macOS. 

In many applications Tesseract is used to ocr documents. Check the [installation](https://github.com/tesseract-ocr/tesseract) 
instruction.

The following overview shows the availability of the models in conjunction with the DL framework.

| Task                                          | PyTorch | Torchscript    |  Tensorflow  |
|-----------------------------------------------|:-------:|----------------|:------------:|
| Layout detection via Detectron2/Tensorpack    |    ✅    | ✅ (CPU only)   | ✅ (GPU only) |
| Table recognition via Detectron2/Tensorpack   |    ✅    | ✅ (CPU only)   | ✅ (GPU only) |
| Table transformer via Transformers            |    ✅    | ❌              |      ❌       |
| DocTr                                         |    ✅    | ❌              |      ✅       |
| LayoutLM (v1, v2, v3, XLM) via Transformers   |    ✅    | ❌              | ❌            |


## Install with pip

We recommend using a virtual environment. You can install **deep**doctection via pip or from source. 

If you want to get started with a minimal setting (e.g. running the **deep**doctection analyzer with 
default configuration or trying the 'Get started notebook'), install **deep**doctection with

```
pip install deepdoctection
```

The following installation will give you ALL models available within the Deep Learning framework as well as all models
that are independent of Tensorflow/PyTorch. Please note, that the dependencies are very complex. We try hard to keep 
the requirements up to date though.

Depending on which Deep Learning library you have available, use the following installation option:

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

This will install **deep**doctection with all dependencies listed in the dependency diagram above the **deep**doctection 
layer. This includes:

- **DocTr**, an OCR library as alternative to Tesseract
- **Pdfplumber**, a PDF text miner based on Pdfminer.six
- **Fasttext**, a library for efficient learning of word representations and sentence classification. Used for language
  recognition only.
- **Boto3**, the AWS SDK for Python to provide an API to AWS Textract (only OCR service). This is a paid service and 
  requires an AWS account.
- **Tensorpack**, if the Tensorflow setting has been installed. Tensorpack is a library for training models and also 
  provides many examples. We only use the object detection model.
- **Transformers**, if the PyTorch setting has been installed. The library provides a lot of different models in various
  frameworks. We currently only provide some PyTorch model wrappers. 
    

Use the setting above, if you want to get started or want to explore all features. 

If you want to have more control with your installation and are looking for fewer dependencies then 
install **deep**doctection with the basic setup only.

```
pip install deepdoctection
```

This will discard all libraries that provide models (layers above the **deep**doctection in the diagram) and you 
will be responsible to install them by yourself. It will, however, install all intrinsic dependencies. You can find them
in requirement.txt. Note, that you will not be able to run any pipeline with this setup though.


## Install from source

If you want all files, notebooks and latest additions etc. then download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

Install the package in a virtual environment. Learn more about 
[`virtualenv`](https://docs.python.org/3/tutorial/venv.html). 

To get started with **Tensorflow**, run:

```
cd deepdoctection 
pip install ".[tf]"
```

or with **PyTorch**:

```
cd deepdoctection
pip install ".[source-pt]"
```

Opposite to the PyPi installation, this will also install Detectron2 for you.

This installation will give you the basic usage of this package and will allow you to run the tutorials.


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

## IPkernel and jupyter notebooks

For running notebooks with kernels pointing to a virtual environment first create a kernel with

```
make install-kernel-dd
```

This command must be run with active venv mode. You can then start a notebook as always and choose the 
kernel 'deep-doc' in the kernel drop down menu.

## Testing the environment

To check, if the installation has been successful you can run some tests. You need to install the test suite for that.

```
pip install -e ".[test]"
```

To run the test cases, use

```
make test-basic
```

This will run some test cases. Run

```
make test-tf
```

or 

```
make test-pt
```

otherwise. 

## Develop environment

To make a full dev installation with an additional update of all requirements, run 


```
make install-dd-dev-tf
```

or 

```
make install-dd-dev-pt
```

Before submitting a PR, format, lint, type-check the code and run the tests:

```
make format-and-qa
```

## Makefile

Check the Makefile for more functionalities (jupyter lab installation etc.)
