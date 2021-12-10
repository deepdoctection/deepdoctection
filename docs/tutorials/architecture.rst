Architecture
========================================


.. image:: ../../notebooks/pics/dd_architecture.png
   :align: center

We have decided to keep the structure as simple as possible and to dispense with an overhead that is too abstract.
That may be at the expense of efficiency, on the other hand, this makes it easier for others to get started.

Dataflow
_______________________________________

`Dataflow  <https://github.com/tensorpack/dataflow>`_ represent the glue between components and are used for streaming
the data. Please refer to the comprehensive and extremely well presented
`tutorials <https://tensorpack.readthedocs.io/en/latest/tutorial/index.html#dataflow-tutorials>`_ for details.


Mapper
_______________________________________

Mappers are functions for transforming data structures. They accept a data point (as a Json object, image, page, ...)
and return a data point.

Mappers must be compatible with the mapping functions of the Dataflow module (MapData, MultiProcessMapData, ...).
This means that the datapoint must be the first input parameter.
Further input parameters for mapping configuration are permitted. However, it must be taken into account that these
must be specified before the dataflow mapping is carried out. A currying decorator simplifies the implementation.

Define my_mapper

.. code:: python

   @cur
   def  my_mapper(dp: Image, config_1: ... , config_2: ...) -> Image:
       # map Image to Image

and use it

.. code:: python

   df = Dataflow(df)
   mapper = my_mapper(config_1,config_2)
   df = MapData(df, mapper)
   ...


:ref:`Pipelines`
_______________________________________

Pipeline components stick together form a pipeline. Data points are passed through between components with data flows.
The essential execution script of the pipeline component is the :meth:`serve`, which records a data point and
coordinates the prediction, transformation and enrichment of the result.

In contrast to the Pipeline Component, Predictor Pipeline Components contain a Predictor. Predictors can be anything:
object detectors, language models and they can be written in any DL-library as the core functions of **deep**doctection
are platform agnostic. Predictors contain the smallest possible number of parameters in order to be able to carry out
the prediction. The predictions are also returned raw and must be transferred to the :meth:`serve` image convention.

