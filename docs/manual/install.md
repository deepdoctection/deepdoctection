# Installation


## Requirements

- Linux **or** macOS
- Python >=  3.8
- PyTorch >= 1.8 **or** Tensorflow >=2.4.1 and CUDA

You can run on PyTorch with a CPU only. For Tensorflow a GPU is required.

For fine-tuning layout models **deep**doctection uses Tensorpack as training
framework and therefore only Tensorflow training scripts are provided.

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
```

For Tensorflow, run 

```
make install-dd-tf
```

If you want to use the PyTorch framework, run:

```
make install-dd-pt
```

This installation will give you the basic usage of this package and will allow you to run the tutorial notebooks.

There are options to install features which require additional dependencies. E.g, if you want to call AWS Textract OCR
while a pipeline you can get the necessary components via 

```
install-dd-aws
```

Run 

```
install-dd-all
```

to install everything which is available. 


## IPkernel and jupyter notebooks

For running notebooks setup of a kernel pointing to the venv is required.

```
make install-kernel-dd
```


## Testing the environment

To check, if the installation has been throughout successful you can run some tests. You need to install the test 
dependencies for that.

```
make install-dd-test
```

To run the test cases:

```
make test
```

## Developing for the environment

To make a full dev installation with update of requirements, run

```
make up-reqs-dev
```

Before submitting a PR, format, lint, type-check the code and run the tests:

```
make format-and-qa
```
