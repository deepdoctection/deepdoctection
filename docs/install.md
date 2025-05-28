# Installation


## Requirements

![](./tutorials/_imgs/requirements_deepdoctection_220525.png)

Everything in the overview listed below the **deep**doctection layer are necessary requirements and have to be installed 
by the user. 

- Linux or macOS. Windows is not supported but there is a [Dockerfile](https://github.com/deepdoctection/deepdoctection/tree/master/docker/pytorch-cpu-jupyter) available.
- Python >= 3.9
- 1.13 <= PyTorch  **or** 2.11 <= Tensorflow < 2.16. For lower Tensorflow versions the code will only run on a GPU. 
  Tensorflow support will be stopped from Python 3.11 onwards.
- To fine-tune models, a GPU is recommended.

For release `v.0.34.0` and below [Poppler](https://poppler.freedesktop.org/) is required for PDF processing. Starting 
from release `v.0.35.0`, the package `pypdfmium2` is used for PDF processing and the default choice. If both are 
available you can choose which one to use by setting environment variables, e.g. `USE_DD_POPPLER=True` and 
`USE_DD_PDFIUM=False`.


With respect to the deep learning framework, you must decide between [Tensorflow](https://www.tensorflow.org/install?hl=en) 
and [PyTorch](https://pytorch.org/get-started/locally/).

We use [Pillow](https://pillow.readthedocs.io/en/stable/) or [OpenCV](https://github.com/opencv/opencv-python) for 
image processing tasks. Pillow is more lightweight, easier to install and the default choice. 
OpenCV is faster when loading images and can be beneficial especially when training. If you want to use OpenCV, please
install this framework separately and set the environment variables `USE_DD_OPENCV=True` and `USE_DD_PILLOW=False`. 

If you only want to run some Tensorpack models, Tensorflow >= 2.4.1 will suffice.
Tensorpack has been developed for TF1, models however runs on TF2 as well by using tf.compat.v1. We will stop supporting
Tensorflow for Python 3.11 and above.

The code has been tested on Ubuntu 20.04-24.04 and on macOS. 

If you want to use Tesseract, check the [installation](https://github.com/tesseract-ocr/tesseract) instructions.

The following overview shows the availability of the models in conjunction with the DL framework.

| Task                                        | PyTorch | Torchscript    |  Tensorflow  |
|---------------------------------------------|:-------:|----------------|:------------:|
| Layout detection via Detectron2/Tensorpack  |    ✅    | ✅ (CPU only)   | ✅ (GPU only) |
| Table recognition via Detectron2/Tensorpack |    ✅    | ✅ (CPU only)   | ✅ (GPU only) |
| Table transformer via Transformers          |    ✅    | ❌              |      ❌       |
| Deformable-Detr                             |    ✅    | ❌              |      ❌       |
| DocTr                                       |    ✅    | ❌              |      ✅       |
| LayoutLM (v1, v2, v3, XLM) via Transformers |    ✅    | ❌              |      ❌       |


## Install with package manager

We recommend using a virtual environment. You can install **deep**doctection from PyPi or from source. 
### Minimal setup

#### PyTorch

```
pip install transformers
pip install python-doctr
pip install deepdoctection
```

#### Tensorflow

```
pip install tensorpack
pip install python-doctr
pip install deepdoctection
```

Both setups are sufficient to run the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb). 

### Full setup

The following installation will give you ALL models available within the Deep Learning framework as well as all models
that are independent of Tensorflow/PyTorch.

#### PyTorch 

First install **Detectron2** separately as it is not distributed via PyPi. 

Check the instruction 
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) or use our fork:

```
pip install detectron2@git+https://github.com/deepdoctection/detectron2.git
```

Then install **deep**doctection with the PyTorch setting:

```
pip install deepdoctection[pt]
```

#### Tensorflow

```
pip install deepdoctection[tf]
```



This will install **deep**doctection with all dependencies listed in the dependency diagram above the **deep**doctection 
layer. This includes:

- **Boto3**, the AWS SDK for Python to provide an API to AWS Textract (only OCR service). This is a paid service and 
  requires an AWS account.
- **Pdfplumber**, a PDF text miner based on Pdfminer.six
- **Fasttext**, a library for efficient learning of word representations and sentence classification. Used for language
  recognition only.
- **Jdeskew**, a library for automatic deskewing of images.
- **Transformers**, a library for state-of-the-art NLP models. 
- **DocTr**, an OCR library as alternative to Tesseract
- **Tensorpack**, if the Tensorflow setting has been installed. Tensorpack is a library for training models and also 
  provides many examples. We only use the object detection model.

Use the setting above, if you want to explore all features. 

If you want to have more control with your installation and are looking for fewer dependencies then 
install **deep**doctection with the basic setup only and add the dependencies you need manually.


### Install from source

If you want all files and latest additions etc. then download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

Install the package in a virtual environment. Learn more about [`virtualenv`](https://docs.python.org/3/tutorial/venv.html). 


#### PyTorch

Again, install **Detectron2** separately.

```
cd deepdoctection
pip install ".[source-pt]"
```

#### Tensorflow

```
cd deepdoctection 
pip install ".[tf]"
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


## Testing the environment

To check, if the installation has been successful you can run some tests. You need to install the test suite for that.

```
pip install -e ".[test]"
```

To run the test cases use `make` and check the Makefile for the available targets.
 

## Developing environment

To make a full dev installation with an additional update of all requirements, run 


```
make install-dd-dev-pt
```

or 

```
make install-dd-dev-tf
```

## Formatting, linting and type checking

```
make format-and-qa
```
