.. deepdoctection documentation master file, created by
   sphinx-quickstart on Fri Oct 15 16:51:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../notebooks/pics/dd_logo.png
   :align: center

Introduction
==============================

**deep**\doctection is a Python package that supports the extraction of visual rich documents.

Extracting from documents is difficult because they generally have a complex visual structure, but the information they
contain is not tagged. **deep**\doctection is a tool box that is intended to facilitate entry into this topic.

The focus should be on application. **deep**\doctection is made for data scientists who are tasked with supporting
departments in process optimization. For analysts who have to investigate into large sets of documents. And also maybe
for researchers who would like to see how well their new model fits into an extraction pipeline.

**deep**\doctection has a modular structure that allows you to combine individual pipelines and thus quickly try out
different approaches. Read more about :ref:`Why deepdoctection ?`



.. toctree::
   :maxdepth: 1
   :caption: Install

   manual/install.md

.. toctree::
   :maxdepth: 2
   :caption: Notebooks and Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: API

   modules/index
