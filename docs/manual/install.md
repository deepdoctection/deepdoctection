# Installation


## Requirements

- Linux **or** macOS
- Python >=  3.8
- PyTorch >= 1.8 **or** Tensorflow >=2.4.1 and CUDA

You can run on PyTorch with a CPU only. For Tensorflow a GPU is required.

For fine-tuning layout models **deep**doctection uses Tensorpack as training framework and therefore only 
Tensorflow training scripts are provided. Tensorpack has been developed for TF1, the model however runs on TF2
as well by using tf.compat.v1. This is not ideal, however transferring all Tensorpack features into a TF2 framework
will take a significant amount of work.

The code has been tested on Ubuntu20.04. Functions not involving a GPU have also been test on MacOS. It is known that 
some code components will have some issues on Windows. We cannot provide support for Windows.

**deep**doctection might depend on other open source packages that have to be installed separately. 

If you want to use Tesseract for extracting text using OCR:
- [Tesseract](https://github.com/tesseract-ocr/tesseract)

If you want to convert PDF into numpy arrays:
- [Poppler](https://poppler.freedesktop.org/)

If we discover projects that cover utilities for additional features to be used in a pipeline we implement wrappers
that simplify usage. In order to not overload requirements and incorporating never used dependencies these projects have 
to be added separately. In most of the cases they directly accessible by a simple pip install. To discover what is 
currently available, please check the 
[API documentation](https://deepdoctection.readthedocs.io/en/latest/modules/deepdoctection.extern.html)

### Install with pip

Dataflow is not available via pip and must be installed separately.

```
pip install  "dataflow @ git+https://github.com/tensorpack/dataflow.git"
```

Depending on which Deep Learning library is available, use the following installation option:

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

## Install from source

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

The installation process will not install the deep frameworks (e.g. Tensorflow or Pytorch) and
therefore to be done by the user itself. We recommend installing the latest version of Tensorflow (2.7) as 
this version is compatible with the required numpy version. Lower versions will not break the code, 
but you will see a compatibility error during the installation process.


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
within a pipeline you will need the boto3. Please install those packages by yourself.  

Run 

```
install-dd-all
```

to install the Tensorflow and Pytorch version in one environment. Note however, that it is not possible 
to run pipeline that depend on both TF and Pytorch components.


## IPkernel and jupyter notebooks

For running notebooks with kernels pointing to a virtual environment first create a kernel with

```
make install-kernel-dd
```

This command must be run with active venv mode. You can then start a notebook as always and choose the 
kernel in the kernel drop down menu.

## Testing the environment

To check, if the installation has been throughout successful you can run some tests. You need to install the test 
dependencies for that.

```
make install-dd-test
```

To run the test cases, use

```
make test-des-pt
```

if you decided to run in Tensorflow framework. This will run all tests that do not 
require Pytorch. Run

```
make test-des-tf
```

otherwise. Note that some tests depend on packages that are not listed in the requirements. In order to run
them successfully install these packages separately. 

## Develop environment

To make a full development installation with an additional update of all requirements, run depending on the work
stream


```
up-reqs-dev-pt
```

or 

```
up-reqs-dev-tf
```

Before submitting a PR, format, lint, type-check the code and run the tests:

```
make format-and-qa
```

## Makefile

Check Makefile for more functionalities.
