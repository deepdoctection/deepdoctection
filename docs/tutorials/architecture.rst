Architecture
========================================


.. image:: ../../notebooks/pics/dd_architecture.png
   :align: center


Dataflow
_______________________________________

`Dataflow  <https://github.com/tensorpack/dataflow>`_ is a package for loading data. It has originally been developed
for training purposes. The idea is basically to have blocks of generators you can chain so that building a pipeline
for loading and transforming data becomes easy. Using the `MapData` allows you to write a simple function and
do not worry about converting the mapping into generator. If you have a bottleneck, `MapProcessMapData` allows
you to easily spawn multiple processes in order to run the bottleneck function in parallel.
We have integrated Dataflow into **deep**doctection as of Release 0.17 so a separate installation process is not
necessary anymore. Please refer to the comprehensive and extremely well presented
`tutorials <https://tensorpack.readthedocs.io/en/latest/tutorial/index.html#dataflow-tutorials>`_ for details.

Datapoint
---------------------------------------

The datapoint package adds the internal data structure to the library. You can view a datapoint as a document
page. The highest level object is provided by the :class:`Image`. The image carries all data retrieved from ground
truth annotations in data sets or from models. This can be layout sections, words and relations between these objects.
Visual lower level objects are modeled as :class:`ImageAnnotation`. They have, among other things,
an attribute `category_name` to define the object type and a :class:`BoundingBox`. To store additional attributes
that depend on the object type (think of table cells where row and column numbers are needed), a generic attribute
`sub_categories` is provided. A generic `relationships` allows to save object specific attributes that relate
different :class:`ImageAnnotation` to each other.


Datasets
---------------------------------------

Please check :ref:`Datasets` for additional information regarding this package.


Extern
---------------------------------------

Models from third party packages must be wrapped into a **deep**doctection class structure so that they are
available for pipelines in unified way. This package provides these wrapper classes.

Mapper
_______________________________________

Mappers are functions (not generators!) for transforming data structures. They accept a data point
(as a Json object, image, page, ...) and return a data point. Mappers are used within pipelines:

.. code:: python

    def my_func(dp: Image) -> Image:
        # do something
        return dp

    df = Dataflow(df)
    df = MapData(df, my_func)

    # or if my_func does some heavy transformation and turns out to be the bottleneck

    df = Dataflow(df)
    df = MultiProcessMapData(df, my_func)


Mappers must be compatible with dataflows. On the other hand, mappers should be flexible enough and therefore they
must be able to accept additional arguments so that additional configuration within the mapping can be applied.
To resolve the problem, a function must be callable twice, i.e.

.. code:: python

    dp = my_func(cfg_param_1, cfg_param_2)(dp)

    # you can also run my_func in a Dataflow with some pre-defined setting cfg_param_1, cfg_param_2

    df = Dataflow(df)
    df = MapData(df, my_func(cfg_param_1, cfg_param_2))
    ...

The `curry` operator disentangles the first argument of a function from the remaining ones.

.. code:: python

   # this makes my_mapper callable twice
   @curry
   def  my_mapper(dp: Image, config_1: ... , config_2: ...) -> Image:
       # map Image to Image


:ref:`Pipelines`
_______________________________________

As said by its name, this package provides you with pipeline components for tasks like layout detection, ocr and several
other services needed. Pipeline components lined up define eventually a pipeline. Check :ref:`Building a custom pipeline`
to learn, how to build pipeline.



