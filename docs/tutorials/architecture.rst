Architecture
========================================


.. image:: ../../notebooks/pics/dd_architecture.png
   :align: center


Dataflow
_______________________________________

`Dataflow  <https://github.com/tensorpack/dataflow>`_ is a package for loading data. It has originally been developed
for training purposes. The idea is basically to have blocks of generators you can chain so that building a pipeline
for loading and transforming data becomes easy. We have integrated the most important `Dataflow` classes into
**deep**\doctection. Please refer to the comprehensive and extremely well presented
`tutorials <https://tensorpack.readthedocs.io/en/latest/tutorial/index.html#dataflow-tutorials>`_ for details. Let's
cover some basic facts.

You can load .jsonlines, .json or file paths with serializers.

.. code:: python

    df = SerializerJsonlines.load("path/to/dir",max_datapoints=100)
    df.reset_state()
    for dp in df:
        # dp is dict

or

.. code:: python

    df = SerializerCoco("path/to/dir")
    df.reset_state()
    for dp in df:
        # dp is a dict with {'image':{'id',...},
                             'annotations':[{'id':…,'bbox':...}]}

You can load a pdf and convert the `SerializerPdfDoc` into the internal data structure.

.. code:: python

    df = SerializerPdfDoc("path/to/dir")

    def _to_image(dp: str) -> Optional[Image]:
        _, file_name = os.path.split(dp)
        dp_dict = {"file_name": file_name, "location": dp}
        return dp_dict

    df = MapData(df, _to_image)
    df.reset_state()
    for dp in df:
       # is now an Image

The snippet above already shows how you transform you data structure or how you perform any other operation: You simply
write a mapper function and use `MapData`. If you see that your mapper is a bottleneck in your data process you can
speed the bottleneck function by using a `MultiProcessMapData` or `MultiThreadMapData`. This class will spawn multiple
processes and parallelize the mapping function to increase throughput.
Check the module `mapper` for tools and samples on how to write mappers to be used with dataflows.



Datapoint
---------------------------------------

The datapoint package adds the internal data structure to the library. You can view a datapoint as a document
page. The highest level object is provided by the :class:`Image`.

.. code:: python

    image = Image(file_name="image_1.png", location = "path/to/dir")

The image carries all data retrieved from ground truth annotations in data sets or from models in pipelines.
This can be detected layout sections, words and relations between various objects.
Visual lower level objects are modeled as :class:`ImageAnnotation`. They have, among other things,
an attribute :class:`category_name` to define the object type and a :class:`BoundingBox`.

.. code:: python

    bounding_box = BoundingBox(absolute_coords=True,ulx=100.,uly=120.,lrx=200.,lry=250.)
    table = ImageAnnotation(bounding_box = bounding_box,
                            category_name = LayoutType.table,
                            category_id="1")     # always use a string. ids will be used for model training
    image.dump(table)    # this adds the table annotation to the image. It generates a md5 hash that you can get
                         # with table.annotation_id

To store additional attributes that depend on the object type (think of table cells where row and column numbers
are needed), a generic attribute :class:`sub_categories` is provided.

.. code:: python

    cell = ImageAnnotation(bounding_box,category_name = "cell", category_id="2")
    row_num = CategoryAnnotation(category_name=CellType.row_number,category_id="6)
    cell.dump_sub_category(CellType.row_number,row_num)

ObjectTypes are enums whose members define all categories. All ObjectTypes are registered with the
`object_types_registry`. If you want to add new categories you have to define a sub class of ObjectTypes
and add the members you want. Do not forget to register your ObjectTypes.

.. code:: python

    @object_types_registry.register("custom_lables")
    class CustomLabel(ObjectTypes):
          train_ticket = "train_ticket"
          bus_tocket = "bus_ticket"

A generic :class:`relationships` allows to save object specific attributes that relate different
:class:`ImageAnnotation` to each other.

.. code:: python

    cell = ImageAnnotation(bounding_box,category_name = "cell", category_id="2")

    for word in word_in_cell:
        cell.dump_relationship(Relationships.child,word.annotation_id)


Datasets
---------------------------------------

Please check :ref:`Datasets` for additional information regarding this package.


Extern
---------------------------------------

Models from third party packages must be wrapped into a **deep**\doctection class structure so that they are
available for pipelines in unified way. This package provides these wrapper classes.

In many cases, model wrappers will be instantiated by providing a config file, some weights
and a Mapping of category_ids to category names.

.. code:: python

    path_weights = ModelCatalog.get_full_path_weights(model_name)
    path_yaml = ModelCatalog.get_full_path_configs(model_name)
    categories = ModelCatalog.get_profile(model_name).categories
    tp_detector = TPFrcnnDetector(path_yaml,path_weights,categories)

However, a few do not require any argument:


.. code:: python

     doct_detector = DoctrTextlineDetector()

To get an overview of all models use `print_model_infos`. For more specific information
consult the `ModelCatalog`.

.. code:: python

    print_model_infos()
    profile = ModelCatalog.get_profile(model_name)

    profile.model_wrapper  # the deepdoctection wrapper, where you can plug in the model
    profile.categories     # dict of category ids and their category names.
    profile.hf_repo_id     # remote storage

Download a model with `ModelDownloadManager`:

.. code:: python

    ModelDownloadManager.maybe_download_weights_and_configs(model_name)




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

This package provides you with pipeline components for tasks like layout detection, ocr and several other services
needed. Chained pipeline components will form a pipeline. Check :ref:`Building a custom pipeline`
to learn, how to build pipelines for a concrete task. Here, we will be giving only a short overview.

There is a registry

.. code:: python

    print(pipeline_component_registry.get_all())


Predictor pipeline components will generally require a model, e.g. ObjectDetector. The following is a full OCR system
with a word detector (generating bounding boxes around words) and a text recognizer (recognizing text within each word
bounding box defines by the word detector).

.. code:: python

    text_line_predictor = DoctrTextlineDetector()
    layout = ImageLayoutService(text_line_predictor,
                                to_image=True)     # ImageAnnotation created from this service will get a nested image
                                                   # defined by the bounding boxes of its annotation. This is helpful
                                                   # if you want to call a service only on the region of the
                                                   # ImageAnnotation

    text_recognizer = DoctrTextRecognizer()
    text = TextExtractionService(text_recognizer, extract_from_roi="word") # text recognition on the region of word
                                                                           # ImageAnnotation
    analyzer = DoctectionPipe(pipeline_component_list=[layout, text])      # defining the pipeline


    path_to_pdf = "path/to/doc.pdf"

    df = analyzer.analyze(path=path_to_pdf)
    SerializerJsonlines.save(df, path= "path/to",
                                 file_name="doc.jsonl",
                                 max_datapoints=20)
