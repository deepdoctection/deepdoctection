

# Contribute to **deep**doctection

We are happy to welcome you as a contributor to **deep**doctection. This document will guide you through the process of
contributing to the project.


## Contributing to the code base

There are countless ways to contribute to the improvement and expansion of the code base. Most of these contributions
do not contain any implementations of their own and yet are just as valuable:

### Improving the documentation

Starting with the introductory REAMDE, through the tutorials and deep dive tutorials to the docstrings in the code:
Documentation is never complete or perfect. It is always possible that things are poorly explained and no longer up to
date. Suggestions or corrections in this regard are incredibly valuable. In fact, if you don't understand a piece of
documentation, it's highly likely that (assuming you've thought about it a bit) others won't understand it either. We
therefore encourage everyone to come around the corner with suggestions for improvement.

### Reporting errors

Most of the issues reported concern difficulties during installation. Deepdoctection is a package with many third party
dependencies and can cause frustration during installation. Problems can be reported as issues. Depending on whether 
these are already being discussed, reference is made to older issues. In order not to force other users to rummage in 
the depths of the already closed issues (although the search works quite well), such issues are also gladly 
converted into a discussion point.

Of course, deepdoctection is not free of logical errors and aborts can occur at any time. It is important to
distinguish whether the error occurs because a model makes an unexpected prediction or whether the written code
leads to unexpected results:

Problems caused by models are due to the fact that they were not trained on the right data to handle your use case.
In this case, we recommend that you adapt the model to your data. Unfortunately, we cannot offer any help if the
predictions do not meet the quality you desire for your use case. 

Errors that are attributable to human-written code can occur in all kinds of places. We are happy to receive reports
of any problems, as this is the only way to improve the code and eliminate errors. The most important thing here is
that we are able to reproduce the error or that we have an idea of the constellation under which the error occurred.

One possibility is to hand over the entire installation palette. This is sufficient:

```python
import deepdoctection as dd

print(dd.collect_env_info())
```

### Adding new features

Deepdoctection was designed and developed almost exclusively by Dr. Janis Meyer. 
Before you have an idea for a major adaptation or extension of the code base, it is 
advisable to get in touch with him to discuss a possible implementation. 

## Setting up the development environment


If you want to develop your own deepdoctection, the easiest way is to fork the repo. 

There is a Makefile for installation in development mode that can simplify the work. 

```bash
make install-dd-dev-pt
```
or

```bash
make install-dd-dev-tf
```

installs deepdoctection in editable mode together with the dependencies for testing, development and documentation.

## Code quality and type checking

We use Pylint for checking code quality and isort as well as black for code formatting:

```bash
make lint
```

```bash
make format
```

We use mypy for type checking:

```bash
make analyze
```

## Testing the environment

Test cases are divided into six groups, which differ in terms of which dependency the test case is based on. The 
dependencies are based on the installation options as specified in setup.py.

`basic`: The basic installation without DL-Libary. Only the packages that are assigned to the dist_deps in setup.py are 
required.

`additional`: Extended installation without DL library. All dist_deps and additional_deps packages are required.

`tf_deps`: Installation with Tensorflow

`pt_deps`: Installation with PyTorch

`requires_gpu`: Test cases that run functions using the GPU

`integration`: Integration test, where a complete pipeline is tested

Test cases with bundled groups can be executed in the Makefile. The most important bundles are

```bash
make test-basic  # Runs only the basic package
make test-pt  # Runs the basic, additional, pt_deps and integration groups.
make test-tf  # Runs the basic, additional and tf_deps groups.
``` 

You can trigger Github Actions to run the tests by pushing your changes to your fork by adding `[force ci]` to your 
commit message. This will run formatting checks, mypy, pylint and tests und Python 3.8 and 3.10. 



