
Datasets
=========

Regardless of whether you use data to fine-tune a task, carry out an evaluation or experiment: The creation of a
dataset provides a way to make data available in a standard format so that it can be processed through components
of the library.

Document Layout Analysis and Visual Document Understanding require image datasets most of the time.
This in turn means that they cannot be loaded into memory. Instead one has a small file with
annotations and links that can be used as a hook to load additional material like images or text data when it is needed.

Let's say you start with a small annotation file containing some ground truth data for some images as well as references
to each image. You load this file into memory having a list of e.g. dicts. You then need something so that you can
iterate over each list element step by step. This is the place where generators come into place.

For users familiar with Pytorch datasets, datasets in **deep**doctection are related to the :class:`IterableDataset`.
In Pytorch you can iterate over samples like:

.. code:: python

    dp = next(iter(MyPytorchDataset))

In **deep**doctection, data sets have a :meth:`build` method in the :class:`DataFlowBuilder` attribute that
returns a :class:`Dataflow`. The :meth:`build` accepts arguments, so that you can change the representation of
datapoints up to some degree or so that you can filter some unwanted samples or reduce the size of the dataset.

A data set consists of three components modelled as attributes: :class:`DatasetInfos`, :class:`DatasetCategories` and a
:class:`Dataflowbuilder` class that have to be implemented individually. :meth:`build` of
:class:`Dataflowbuilder` returns a generator, a :class:`Dataflow` instance from which data points can be streamed.

A paradigm of **deep**doctection is that data points, if they are streamed via the :meth:`build`, are not returned in the
raw annotation format of the annotation file, but in :class:`Image` format. An :class:`Image` is a class in which
annotations from common Document-AI tasks can be mapped in a standardized manner. This uniform mapping may seem
redundant at first, but once having standard data point formats, it is relatively simple to try out different components
of the **deep**doctection framework or to merge datasets.
First of all, an :class:`Image` contains information about the image sample itself. This includes metadata such as
storage location or pixel width and height. The image sample itself can also be saved as a numpy array. However, for the
reasons mentioned above, one should not initially save the image sample when loading annotations, but only provide the
location. While training or evaluation, the image itself is loaded from its storage location just when needed, that is
just before it is loaded into the model to perform a forward path. After that it will be immediately removed from memory.

An :class:`Image` contains the information about the annotations of an image in the :class:`ImageAnnotations`.
:class:`ImageAnnotation` is a class that allows storing bounding boxes, category names, but also subcategories and
relations.

As far as mapping is concerned, there are already some important mapping functions that convert datapoints from a raw
annotation format into an :class:`Image`. It's a good idea to look at a mapping function like :func:`coco_to_image`,
where a data point in coco format is mapped into an :class:`Image`.


Custom Data set
---------------

The easiest way is to physically store a dataset in the .cache directory of **deep**doctection (usually this is
~/.cache/deepdoctection/datasets). If you pass the argument

.. code:: python

    location = "custom_dataset"

in the dataflow builder, it is assumed that the dataset was physically stored in the "custom_dataset" sub directory of
datasets. We assume that in "custom_dataset" the data set was physically placed following the structure:


|    custom_dataset
|    ├── train
|    │ ├── 01.png
|    │ ├── 02.png
|    ├── gt_train.json



.. code:: python

    import deepdoctection as dd

    _NAME = "dataset name"
    _DESCRIPTION = "a short description"
    _SPLITS = {"train": "/train"}
    _LOCATION = "custom_dataset"
    _ANNOTATION_FILES = {"train": "gt_train.json"}
    _CATEGORIES = ["label_1","label_2"]

    class CustomDataset(dd.DatasetBase):

        @classmethod
        def _info(cls):
            return dd.DatasetInfo(name=_NAME, description=_DESCRIPTION, splits=_SPLITS)

        def _categories(self):
            return dd.DatasetCategories(init_categories=_CATEGORIES)

        def _builder(self):
            return CustomDataFlowBuilder(location=_LOCATION,annotation_files=_ANNOTATION_FILES)



Three methods :meth:`_info`, :meth:`_categories` and :meth:`_builder` must be implemented for a data set, each of which
return an instance :class:`DatasetInfo`, :class:`DatasetCategories` or None and a class derived from
:class:`DataFlowBaseBuilder`.

DatasetInfo
~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`DatasetInfo` instance must be returned. :class:`DatasetInfo` essentially only stores attributes that have
informative characters. The instance must be created, but all arguments, with the exception of :param:`name`, can be
defaulted.

DatasetCategories
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`DatasetCategories` provides a way to manage categories and sub-categories.
This proves to be useful if, for example, you want to filter out certain categories in the dataset. Another application
arises, if you have annotations with categories and sub-categories in the dataset and want to see annotations labeled
with their sub-category name instead of their category name.

:class:`DatasetCategories` takes as argument a list of init_categories, with category names. Category names must be
members of an :class:`Enum` class that must be derived from :class:`ObjectTypes`. If there are sub-
categories, init_sub_categories returns a dict with category names as key and a list of subcategory names as value.

Example: In the annotation file there is a category "TABLE_CELL", where "TABLE_CELL" can contain two possible
sub categories "TABLE_HEADER" and "TABLE_BODY". Suppose there are no more categories and sub categories. Then we
define a :class:`ObjectTypes` for new categories and initialize :class:`DatasetCategories`.

.. code:: python

    @object_types_registry.register("TableCellType")  # we need to register the ObjectType
    class CellType(ObjectTypes):
        table_cell = "TABLE_CELL"
        table_header = "TABLE_HEADER"
        table_body = "TABLE_BODY"

    DatasetCategories(init_categories=[CellType.table_cell],
                      init_sub_categories={CellType.table_cell:[CellType.table_header, CellType.table_body]}).

When initializing :class:`DatasetCategories` it is important to know the meta data of the dataset annotation file
(available labels etc.) otherwise, logical errors can occur too quickly. That means, if you are in doubt, what
categories might occur, or how sub-categories are related to categories, it is worth the time to perform a quick
analysis on the annotation file.

DataflowBuilder
~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataflow builder is the tool to create a stream for the dataset. The base class contains an abstract method
:meth:`build`. The following has to be implemented:

- Loading a data point (e.g. ground truth data and additional components, such as an image or a path) in raw format.

- Transforming the raw data into the core data model.

Various tools are available for loading and transforming. If the ground truth is in Coco format,
for example, the annotation file can be loaded with SerializerCoco. The instance returns a data flow through which each
sample is streamed individually.

A mapping is required for the transformation, which transfers raw data into the core data model. Here, too, there
are some functions available for different annotation syntax in the mapper package.

.. code:: python

    class CustomDataFlowBuilder(DataFlowBaseBuilder):

        def build(self, **kwargs) :

            # Load
            path = os.path.join(self.location,self.annotation_files["train"])
            df = dd.SerializerCoco.load(path)
            # yields {'image':{'id',...},'annotations':[{'id':..,'bbox':...}]}

            # Map
            coco_to_image_mapper = dd.coco_to_image(self.categories.get_categories(),
                                                 load_image=True,
                                                 filter_empty_image=True,
                                                 fake_score=False)
            df = dd.MapData(df,coco_to_image_mapper)
            # yields Image(file_name= ... ,location= ...,annotations = ...)

            return df

Built-in Dataset
----------------

A DatasetRegistry facilitates the construction of built-in datasets. We refer to the API documentation for the available
build configurations of the dataflows.

.. code:: python

   dataset = dd.get_dataset("dataset_name")
   df = dataset.dataflow.build(**kwargs_config)

   for sample in df:
       print(sample)
