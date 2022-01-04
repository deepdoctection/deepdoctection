# Installation


## Requirements

If you want to do the interesting stuff with **deep**doctection you will need a GPU. 

You can run some notebooks on Google Colab. However, if you want to do something more elaborate which will take 
more time it is better to look for something else.

## Software Requirements

- Python >=3.8
- NVIDIA CUDA 11.0
- CUDNN8
- Tensorflow >=2.4.1

The code has been tested on Ubuntu20.04. Functions not involving a GPU have also been test on MacOS. It is known that 
some code components will have some issues on Windows.

**deep**doctection might depend on other open source packages that have to be installed separately. 

If you want to use Tesseract for extracting text using OCR:
- [Tesseract](https://github.com/tesseract-ocr/tesseract)

If you want to convert PDF into numpy arrays:
- [Poppler](https://poppler.freedesktop.org/)


### AWS 

### Vast.ai

Vast.ai is a platform where businesses/private people offer cheap cloud rentals empowered with GPU. If 
data protection of project is not the biggest concern this might be an option, as GPU rental costs can be reduced 
3x to 5x. Offers range from small private machines for as little as $0.1 p/h up to $13.2 for 8xA100! You can easily 
choose a docker image satisfying your needs and start within a few seconds. 

```
Image: nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
```

## Install

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

Install the package in a virtual environment. Learn more about [`virtualenv`](https://docs.python.org/3/tutorial/venv.html). 

```
cd deepdoctection
make clean
make venv
source venv/bin/activate
make up-reqs-dev
```

## IPkernel and jupyter notebooks

For running notebooks setup of a kernel pointing to the venv is required.

```
make install-kernel-deepdoc
```


## AWS Textract

AWS Textract OCR service can be called in a pipeline. To make use of this service install the necessary components

```
make install-aws-dependencies
```

## Testing the environment

You can check if the installation has been successful by running the tests.

```
make test
```
