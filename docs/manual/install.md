# Installation


## Requirements

- Linux **or** macOS
- Python >=  3.8
- PyTorch >= 1.8 **or** Tensorflow >=2.4.1 and CUDA

You can run on PyTorch with a CPU only. For Tensorflow a GPU is required.

As of release v.014 it is possible to fine-tune vision models either with Tensorpack or Detectron2. Experiments
show, that Detectron2 trains faster and evaluation results are more accurate. 
Tensorpack has been developed for TF1, models however run on TF2 as well by using tf.compat.v1. This is not ideal, 
however transferring all Tensorpack features into a TF2 framework will take a significant amount of work and it is not 
our top priority.

The code has been tested on Ubuntu20.04. Functions not involving a GPU have also been tested on MacOS. It is known that 
some code components will have some issues on Windows. We therefore do not support Windows.

If you want to use Tesseract for extracting text using OCR check the installation instructions here
- [Tesseract](https://github.com/tesseract-ocr/tesseract)

If you want to convert PDF into numpy arrays, please consult 
- [Poppler](https://poppler.freedesktop.org/)

If we discover projects that cover utilities for additional features to be used in a pipeline we implement wrappers
that simplify usage. In order to not overload requirements and incorporating never used dependencies these projects have 
to be added separately (unless you use the 'source-all-tf', 'source-all-pt' option as described below). In most of the 
cases they are directly accessible by a simple pip install. To discover what is currently available, please check the 
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

first install Detectron2 separately as it is not on the pypi server, either. Check the instruction 
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Then run

```
pip install deepdoctection[pt]
```


This will install the basic setup which is needed to run the first two notebooks and do some inference with pipelines.

Some libraries are not added to the requirements in order to keep the dependencies as small as possible (e.g. DocTr,
pdfplumber, fastText, ...). If you want to use them, please pip install these separately or install this package from 
source (see below). 

To use more features (e.g. run all available notebooks), try:

```
pip install deepdoctection[full-tf]
```

Note, that this option is not available for PyTorch.


## Install from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

Install the package in a virtual environment. Learn more about 
[`virtualenv`](https://docs.python.org/3/tutorial/venv.html). 

The installation process will not install the deep learning frameworks Tensorflow or PyTorch and
therefore has to be installed by the user itself. We recommend installing Tensorflow >=2.7 as 
this version is compatible with the required numpy version. Lower versions will not break the code, 
but you will see a compatibility error during the installation process.

To get started with Tensorflow, run:

```
cd deepdoctection 
pip install ".[source-tf]"
```

or with **PyTorch**:

```
cd deepdoctection
pip install ".[source-pt]"
```
Note that pip must run where `setup.py` module resides. 

This installation will give you the basic usage of this package and will allow you to run the first two tutorial 
notebooks.

To run evaluation, using datasets or fine tuning models, further dependencies need to be respected. 
Instead of the above, run for **Tensorflow**:

```
pip install ".[source-full-tf]"
```

There is no corresponding installation available for PyTorch, as the basic installation already covers all dependencies
for using all datasets or running evaluation.

There are options to install features which require additional dependencies. E.g, if you want to call AWS Textract OCR
within a pipeline you will need the boto3. Please install those packages by yourself.  

### Installing with all dependencies

If you want to have the full flexibility by composing pipelines with all available model wrappers provided by 
deepdoctection, use

```
cd deepdoctection
pip install ".[source-all-tf]"
```

or respectively,

```
cd deepdoctection
pip install ".[source-all-pt]"
```


## IPkernel and jupyter notebooks

For running notebooks with kernels pointing to a virtual environment first create a kernel with

```
make install-kernel-dd
```

This command must be run with active venv mode. You can then start a notebook as always and choose the 
kernel 'deep-doc' in the kernel drop down menu.

## Testing the environment

To check, if the installation has been throughout successful you can run some tests. You need to install the 
test dependencies for that.

```
pip install -e ".[test]"
```

To run the test cases, use

```
make test-tf-basic
```

if you decided to run in Tensorflow framework. This will run some test cases that do not 
require Pytorch. Run

```
make test-pt-full
```

otherwise. Note that some tests depend on packages that are not listed in the requirements. In order to run
them successfully install these packages separately. 

## Develop environment

To make a full development installation with an additional update of all requirements, run depending on the work
stream


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
