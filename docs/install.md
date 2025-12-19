# 


## Requirements

![](./tutorials/_imgs/install_01.png)

Everything in the overview listed below the **deep**doctection layer are necessary requirements and have to be installed 
by the user.

- Linux, macOS and Windows should work. We haven't tried on Windows but there is a [Dockerfile](https://github.com/deepdoctection/deepdoctection/tree/master/docker/gpu/Dockerfile) available.
- Python >= 3.10
- 2.6 <= PyTorch
- To fine-tune models, a GPU is recommended.


??? info "Poppler vs. PDFium"

    **deep**doctection supports two different libraries for PDF processing:
    For release `v.0.34.0` and below [Poppler](https://poppler.freedesktop.org/) is required for PDF processing. Starting 
    from release `v.0.35.0`, [`pypdfmium2`](https://github.com/pypdfium2-team/pypdfium2) is used for PDF processing and 
    the default choice. If both are available you can choose which one to use by setting environment variables, e.g. 
    `USE_DD_POPPLER=True` and `USE_DD_PDFIUM=False`.

??? info "PIL vs. OpenCV"

    We use [Pillow](https://pillow.readthedocs.io/en/stable/) or [OpenCV](https://github.com/opencv/opencv-python) for 
    image processing tasks. Pillow is more lightweight, easier to install and the default choice. 
    OpenCV is faster when loading images and can be beneficial especially when training. If you want to use OpenCV, please
    install this framework separately and set the environment variables `USE_DD_OPENCV=True` and `USE_DD_PILLOW=False`. 

??? info "Tesseract"

    Tesseract must be [installed](https://github.com/tesseract-ocr/tesseract) separately. For using Tesseract, a 
    configuration file is available at `~/.cache/deepdoctection/configs/dd/conf_tesseract.yaml`. In addition to the 
    `LANGUAGES` and `LINES` arguments, all other configuration parameters provided by Tesseract can also be used.

??? info "Detectron2"

    The default setting does not require Detectron2 anymore, but if you want to use D2-models you can install it. See 
    [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for more information.


The following overview shows the availability of the models in conjunction with the DL framework.

| Task                                        | PyTorch | Torchscript    |
|---------------------------------------------|:-------:|----------------|
| Layout detection via Detectron2             |    ✅    | ✅ (CPU only)   |
| Table recognition via Detectron2            |    ✅    | ✅ (CPU only)   |
| Table transformer via Transformers          |    ✅    | ❌              |
| Deformable-Detr                             |    ✅    | ❌              |
| DocTr                                       |    ✅    | ❌              | 
| LayoutLM (v1, v2, v3, XLM) via Transformers |    ✅    | ❌              |


## Install with package manager

We recommend using a virtual environment. You can install **deep**doctection from PyPi or from source. 

### Minimal setup

```
uv pip install timm
uv pip install "transformers>=4.48.0,<5.0.0"
uv pip install python-doctr>=1.0.0
uv pip install deepdoctection
```


### Full setup

Install [**Detectron2**](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) separately as it is not distributed via PyPi. Note that PyTorch must be installed first.

You can use our fork:

```
pip install detectron2@git+https://github.com/deepdoctection/detectron2.git --no-build-isolation
```

Then install all remaining dependencies with:

```
pip install deepdoctection[full]
```

!!! info 

    This will install **deep**doctection with all dependencies listed in the dependency diagram above the **deep**doctection 
    layer. This includes:

    - **Boto3**, the AWS SDK for Python to provide an API to AWS Textract (only OCR service). This is a paid service and 
      requires an AWS account.
    - **Pdfplumber**, a PDF text miner based on Pdfminer.six
    - **Jdeskew**, a library for automatic deskewing of images.
    - **Transformers**, a library for state-of-the-art NLP models. Some vision and Bert-like models can be run with
      **deep**doctection.
    - **DocTr**, an OCR library as alternative to Tesseract

    It will also install `dd_datasets` which is necessary for fine-tuning models on custom datasets.


### Install from source

If you want all files and latest additions etc. then download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

Install the package in a virtual environment. Learn more about [`virtualenv`](https://docs.python.org/3/tutorial/venv.html). Then use our Makefile to install
in editable mode.


```bash
make install-dd
```

The Makefile is the starting point for other install options. Use `make help` to see all available targets.


### Running a Docker container from Docker hub

Starting from release `v.0.27.0`, pre-existing Docker images can be downloaded from the [Docker hub](https://hub.docker.com/r/deepdoctection/deepdoctection).

```
docker pull deepdoctection/deepdoctection:<release_tag> 
```

To start the container, you can use the Docker compose file `./docker/gpu/docker-compose.yaml`. 
In the `.env` file provided, specify the host directory where **deep**doctection's cache should be stored. 
This directory will be mounted. Additionally, specify a working directory to mount files to be processed into the 
container.

```
docker compose up -d
```

will start the container.


## Developing and testing

Again use our Makefile to install with full dev- and test dependencies.

```bash
make install-dd-dev
```

We use [tox](https://tox.wiki/en/4.32.0/) for orchestrating tests across multiple Python versions, for formatting and
type checking.

To run tests:

```bash
make test
```

To format code:

```bash
make format
```

To run QA suite:

```bash
make qa
```