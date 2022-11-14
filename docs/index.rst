.. deepdoctection documentation master file, created by
   sphinx-quickstart on Fri Oct 15 16:51:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../notebooks/pics/dd_logo.png
   :align: center

Introduction
==============================

**deep**\doctection is a Python library that orchestrates document extraction and document layout analysis tasks using
deep learning models. It does not implement models but enables you to build pipelines using highly acknowledged
libraries for object detection, OCR and selected NLP tasks and provides an integrated frameworks for fine-tuning,
evaluating and running models. For more specific text processing tasks use one of the many other great NLP libraries.

**deep**\doctection focuses on applications and is made for those who want to solve real world problems related to
document extraction from PDFs or scans in various image formats.


**deep**\doctection has a modular structure that allows you to combine individual pipelines and thus quickly try out
different approaches. Read more about :ref:`Why deepdoctection ?`



.. toctree::
   :maxdepth: 1
   :caption: Install

   manual/install.md

.. toctree::
   :maxdepth: 1
   :caption: Notebooks and Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: API

   modules/index
