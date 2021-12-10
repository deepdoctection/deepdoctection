
Datasets
==========================

Regardless of whether you use data to fine-tune a task, carry out an evaluation or experiment: The creation of a
dataset provides a standardized option to make data available for various tasks in the package.


A dataset can take many forms. We start from the situation where we have individual document pages and ground truth
annotations.

Custom Dataset
--------------------------


|    custom_dataset
|    ├── train
|    │ ├── 01.png
|    │ ├── 02.png
|    ├── gt_train.json



.. code:: python

    _NAME = "dataset name"
    _DESCRIPTION = "a short description"
    _SPLITS = {"train": "/train"}
    _LOCATION = "path/to/custom_dataset"
    _ANNOTATION_FILES = {"train": "gt_train.json"}
    _CATEGORIES = ["label_1","label_2"]

    class CustomDataset(DatasetBase):

        def _info(self):
            return DatasetInfo(name=_NAME, description=_DESCRIPTION, splits=_SPLITS)

        def _categories(self):
            return DatasetCategories(init_categories=_CATEGORIES)

        def _builder(self):
            return CustomDataFlowBuilder(location=_LOCATION,annotation_files=_ANNOTATION_FILES)



Three methods _info, _categories and _builder must be implemented for a dataset, each of which returns an instance
DatasetInfo, DatasetCategories or None and a builder.

DatasetInfo
~~~~~~~~~~~~~~~~~~~~~~~~~~

High level information on the dataset can be stored in the DatasetInfo object. Only the dataset name is mandatory.

DatasetCategories
~~~~~~~~~~~~~~~~~~~~~~~~~~

Information on labels and sub-categories is stored in DatasetCategories. DatasetCategories is useful if you want to
filter certain categories or if you want to exchange categories with subcategories. DatasetCategories takes over the
management of the label sets in such operations.

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
are some functions for different annotation syntaxes in the mapper package.

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

   dataset = DatasetRegistry.get_dataset("dataset_name")
   df = dataset.dataflow.build(**kwargs_config)

   for sample in df:
       print(sample)
