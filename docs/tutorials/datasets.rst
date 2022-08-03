
Datasets
==========================

Regardless of whether you use data to fine-tune a task, carry out an evaluation or experiment: The creation of a
dataset provides a standardized option to make data available for various tasks in the package.

It is due to the task Document Layout Analysis and Document Information Extraction that all datasets are image datasets.
This means that these usually cannot be loaded into memory. At most the annotation file can be loaded into memory
(e.g. if it is a .json file) and even here the situation is often that there is a separate annotation file for each
image or that the annotation file is a .jsonl file that has to be loaded iteratively. For this reason, deepdoctection
are mapped as a so-called IterableDataset.

For users familiar with Pytorch datasets, datasets in deepdoctection are related to the :class:`IterableDataset`. But
instead of, as in Pytorch, being an iterator itself and iterating over samples like

.. code:: python

    dp = next(iter(MyPytorchDataset))

deepdoctection has a :meth:`build` method in the :class:`DataFlowBuilder` attribute that returns a :class:`Dataflow`.
This allows customization of datapoints being streamed, like special filtering among annotations.

A dataset consists of three components modelled as attributes: :class:`DatasetInfos`, :class:`DatasetCategories` and a
:class:`Dataflowbuilder` class that has to be implemented separately for a dataset. The :meth:`build` of
:class:`Dataflowbuilder` returns a generator, a :class:`Dataflow` from which allows data point streaming.

A paradigm of deepdoctection is that data points, if they are streamed via the :meth:`build`, are not returned in the
raw annotation format of the annotation file, but in an class:`Image` format. An :class:`Image` is a class in which
annotations from common Document-AI tasks can be mapped in a standardized manner. This uniform mapping may seem
redundant at first, but once having standard data point formats, it is relatively simple to try out different components
of the deepdoctection framework or to merge datasets.
First of all, an :class:`Image` contains information about the image sample itself. This includes metadata such as
storage location or pixel width and height. The image sample itself can also be saved as a numpy array. However, for the
reasons mentioned above, one should not initially save the image sample when loading annotations, but only providing the
location. In the training and evaluation scripts, the image sample then is loaded from its storage location just before
the data is loaded into the model and then thereafter immediately removed from memory.

An :class:`Image` contains the information about the annotations of an image in the :class:`ImageAnnotations`.
ImageAnnotation is a class that allows to store bounding boxes, category names, but also subcategories and relations.
As far as mapping is concerned, there are already some important mapping functions that convert an annotation format
into an :class:`image`. These are stored in the mapper module. It's a good idea to look at a mapping function like
:func:`coco_to_image`, where a data point in coco format is mapped into an :class:`Image`.


Custom Dataset
--------------------------

The easiest way is to physically store a dataset in the .cache directory of **deepdoctection** (usually this is
~/.cache/deepdoctection/datasets). If you pass the argument

.. code:: python

    location = "custom_dataset"

in the dataflow builder, it is assumed that the dataset was physically stored in the "custom_dataset" subdirectory of
datasets. We assume that in "custom_dataset" the dataset was physically placed in the following structure:


|    custom_dataset
|    ├── train
|    │ ├── 01.png
|    │ ├── 02.png
|    ├── gt_train.json



.. code:: python

    _NAME = "dataset name"
    _DESCRIPTION = "a short description"
    _SPLITS = {"train": "/train"}
    _LOCATION = "custom_dataset"
    _ANNOTATION_FILES = {"train": "gt_train.json"}
    _CATEGORIES = ["label_1","label_2"]

    class CustomDataset(DatasetBase):

        @classmethod
        def _info(cls):
            return DatasetInfo(name=_NAME, description=_DESCRIPTION, splits=_SPLITS)

        def _categories(self):
            return DatasetCategories(init_categories=_CATEGORIES)

        def _builder(self):
            return CustomDataFlowBuilder(location=_LOCATION,annotation_files=_ANNOTATION_FILES)



Three methods :meth:`_info`, :meth:`_categories` and :meth:`_builder` must be implemented for a dataset, each of which
returns an instance :class:`DatasetInfo`, :class:`DatasetCategories` or None and a class derived from
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
arises, for example, if you have annotations with categories and sub-categories in the dataset and want to see
annotations labeled with their sub-category name instead of their category name.

:class:`DatasetCategories` takes as argument a list of init_categories, with category names as string. If there are sub-
categories, init_sub_categories returns a dict with category names as key and a list of subcategory names as value.

Example: In the annotation file there is a category "TABLE_CELL", where "TABLE_CELL" can contain two possible
subcategories "TABLE_HEADER" and "TABLE_BODY". Suppose there are no more categories and subcategories. Then we
initialize

.. code:: python

    DatasetCategories(init_categories=["TABLE_CELL"],init_sub_categories={"TABLE_CELL":[ "TABLE_HEADER", "TABLE_BODY"]}).

When initializing :class:`DatasetCategories` it is important to know the meta data of the dataset annotation file
otherwise, logical errors can occur too quickly during processing. That means, if you are in doubt, what categories
might occur, or how sub-categories are related to categories, it is worth the time to perform a quick analysis on the
annotation file.

DataflowBuilder
~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataflow builder is the tool to create a stream for the dataset. The base class contains an abstract method
:meth:`build`. The following has to be implemented:

- Loading a data point (e.g. ground truth data and additional components, such as an image or a path) in the raw form.

- Transforming the raw data into the core data model.

Various tools are available for loading and transforming and even more is available when using :ref:`Dataflow
<https://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.htmlpackage>`. If the ground truth is in Coco format,
for example, the annotation file can be loaded with SerializerCoco. The instance returns a data flow through which each
sample is streamed individually.

A mapping is required for the transformation, which transfers raw data into the core data model. Here, too, there
are some functions available for different annotation syntax in the mapper package.

.. code:: python

    class CustomDataFlowBuilder(DataFlowBaseBuilder):

        def build(self, **kwargs) :

            # Load
            path = os.path.join(self.location,self.annotation_files["train"])
            df = SerializerCoco.load(path)
            # yields {'image':{'id',...},'annotations':[{'id':..,'bbox':...}]}

            # Map
            coco_to_image_mapper = coco_to_image(self.categories.get_categories(),
                                                 load_image=True,
                                                 filter_empty_image=True,
                                                 fake_score=False)
            df = MapData(df,coco_to_image_mapper)
            # yields Image(file_name= ... ,location= ...,annotations = ...)

            return df

Built-in Dataset
---------------------------

A DatasetRegistry facilitates the construction of built-in datasets. We refer to the API documentation for the available
build configurations of the dataflows.

.. code:: python

   dataset = get_dataset("dataset_name")
   df = dataset.dataflow.build(**kwargs_config)

   for sample in df:
       print(sample)
