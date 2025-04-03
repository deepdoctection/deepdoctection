
# Datasets


Regardless of whether you use data to fine-tune a task, carry out an evaluation or experiment: The creation of a
dataset provides a way to make data available in a standard format so that it can be processed through components
of the library.

DLA and VDU require image datasets.
This in turn means that they cannot be loaded into memory. Instead, one has a small file with
annotations and links that can be used as a hook to load additional material like images or text data when it is needed.

Let's say you start with a small annotation file containing some ground truth data for some images as well as references
to each image. You load this file into memory having a list of e.g. dicts. You then need something so that you can
iterate over each list element step by step. This is the place where generators come into place.

For users familiar with Pytorch datasets, datasets in **deep**doctection are related to the `IterableDataset`.
In Pytorch you can iterate over samples like:

``` python
    dp = next(iter(MyPytorchDataset))
```

## DatasetInfo, DatasetCategories and DataFlowBuilder


In **deep**doctection, data sets have a `build` method in the `DataFlowBuilder` attribute that
returns a `DataFlow`. The `build` accepts arguments, so that you can change the representation of
datapoints up to some degree or so that you can filter some unwanted samples or reduce the size of the dataset.

A data set consists of three components modelled as attributes: [`DatasetInfo`][deepdoctection.datasets.info.DatasetInfo], 
[`DatasetCategories`][deepdoctection.datasets.info.DatasetCategories] and a
[`DataFlowBuilder`][deepdoctection.datasets.dataflow_builder.DataFlowBaseBuilder] class that have to be implemented
individually. `build` of `DataFlowBuilder` returns a generator, a `DataFlow` instance from which data points can be streamed.


## Custom dataset


As of version 0.18, there is a client [`CustomDataset`][deepdoctection.datasets.base.CustomDataset] that helps you to
create quickly a dataset without implementing a large overhead. Basically, you have to write a
`DataFlowBuilder` and you have to instantiate a `CustomDataset`.

The easiest way is to physically store a dataset in the .cache directory of **deep**doctection (usually this is
~/.cache/deepdoctection/datasets). Create a sub folder *custom_dataset* and store you dataset in this sub folder.

If you set

```python

    my_custom_dataset = CustomDataset(...,location = "custom_dataset",..)
```

then

```python
   my_custom_dataset.dataflow.get_workdir()
```

will point to the sub folder "custom_dataset". Moreover, you have to map every dataset to a  `dataset_type`. This must
be one of the members of the `DatasetType`. The most crucial part is to build a `DataFlowBaseBuilder`.

```python

    class CustomDataflow(DataFlowBaseBuilder):

        def build(**kwargs):

            path =  self.get_workdir() / annotation_file.jsonl
            df = SerializerJsonLines.load(path)                      # will stream every .json linewise
            ...
```

Note, that `build` must yield an [`Image`][deepdoctection.datapoint.image.Image]. It is therefore crucial to map the
data structure of the annotation file into an `Image`. Fortunately, there are already some mappings made available.
For COCO-style annotation, you can simply do:

```python

    class CustomDataflow(DataFlowBaseBuilder):

        def build(**kwargs):

            path =  self.get_workdir() / annotation_file.json
            df = SerializerCoco.load(path)                    # will load a coco style annotation file and combine
                                                              # image and their annotations.


            # a callable with some configuration (mapping category ids and category names/ skipping the image loading)
            coco_mapper = coco_to_image(self.categories.get_categories(init=True),
                                         load_image= False)
            df = MapData(df, coco_mapper)
            return df
```

This dataflow has a very basic behaviour. You can add some more functionalities like filtering some categories.


```python

        class CustomDataflow(DataFlowBaseBuilder):

            def build(**kwargs):
                ...
                df = MapData(df, coco_mapper)

                if self.categories.is_filtered():
                    df = MapData(df, filter_cat(self.categories.get_categories(as_dict=False, filtered=True),
                                                self.categories.get_categories(as_dict=False, filtered=False),
                                 ),
                    )
```
&nbsp;

Having added this to your dataflow, you can now customize your categories:

```python

    my_custom_dataset = CustomDataset("train_data",
                                       DatasetType.object_detection,
                                       "custom_dataset_location",
                                       [LayoutType.text, LayoutType.title, LayoutType.table],
                                       CustomDataflow("custom_dataset_location",{"train": "annotation_file.json"}))

    my_custom_dataset.dataflow.categories.filter_categories(categories="table")

    df = my_custom_dataset.dataflow.build()
    df.reset_state()
    for dp in df:
        ... # dp has now only 'table' labels. 'text' and 'title' has been filtered out.
```


## How to build datasets the long way


We assume that in *custom_dataset* the data set was physically placed in the following the structure:

```
|    custom_dataset
|    ├── train
|    │ ├── 01.png
|    │ ├── 02.png
|    ├── gt_train.json
```


```python

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
            return dd.CustomDataFlowBuilder(location=_LOCATION,annotation_files=_ANNOTATION_FILES)
```


Three methods `_info`, `_categories` and `_builder` must be implemented for a data set, each of which
return an instance `DatasetInfo`, `DatasetCategories` or None and a class derived from `DataFlowBaseBuilder`.

## DatasetInfo


A [`DatasetInfo`][deepdoctection.datasets.info.DatasetInfo] instance must be returned. `DatasetInfo` essentially 
only stores attributes that have informative characters. The instance must be created, but all almost all
arguments can be defaulted.

## DatasetCategories


[`DatasetCategories`][deepdoctection.datasets.info.DatasetCategories] provide a way to manage categories and sub-categories.
This proves to be useful if, for example, you want to filter out certain categories in the dataset. Another application
arises, if you have annotations with categories and sub-categories in the dataset and want to see annotations labeled
with their sub-category name instead of their category name.

`DatasetCategories` takes as argument a list of init_categories, with category names. Category names must be
members of an `Enum` class that must be derived from [`ObjectTypes`][deepdoctection.utils.settings.ObjectTypes]. 
If there are sub-categories, `init_sub_categories` returns a dict with category names as key and a list of subcategory
names as value.

**Example:** In the annotation file there is a category "TABLE_CELL", where "TABLE_CELL" can contain two possible
sub categories "TABLE_HEADER" and "TABLE_BODY". Suppose there are no more categories and sub categories. Then we
define a `ObjectTypes` for new categories and initialize `DatasetCategories`.

```python

    @dd.object_types_registry.register("TableCellType")  # we need to register the ObjectType
    class CellType(ObjectTypes):
        TABLE_CELL = "table_cell"
        TABLE_HEADER = "table_header"
        TABLE_BODY = "table_body"

    dd.DatasetCategories(init_categories=[CellType.TABLE_CELL],
                      init_sub_categories={CellType.TABLE_CELL:[CellType.TABLE_HEADER, CellType.TABLE_BODY]}).

```

When initializing `DatasetCategories` it is important to know the metadata of the dataset annotation file
(available labels etc.) otherwise, logical errors can occur too quickly. That means, if you are in doubt, what
categories might occur, or how sub-categories are related to categories, it is worth the time to perform a quick
analysis on the annotation file.

## DataflowBuilder


The dataflow builder is the tool to create a stream for the dataset. The base class contains an abstract method
`build`. The following has to be implemented:

- Loading a data point (e.g. ground truth data and additional components, such as an image or a path) in raw format.

- Transforming the raw data into the core data model.

Various tools are available for loading and transforming. If the ground truth is in Coco format,
for example, the annotation file can be loaded with SerializerCoco. The instance returns a data flow through which each
sample is streamed individually.
A mapping is required for the transformation, which transfers raw data into the core data model. Here, too, there
are some functions available for different annotation syntax in the mapper package.

```python

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
```


## Built-in Dataset


A `DatasetRegistry` facilitates the construction of built-in datasets. We refer to the API documentation for the available
build configurations of the dataflows.

```python

   dataset = dd.get_dataset("dataset_name")
   df = dataset.dataflow.build(**kwargs_config)

   for sample in df:
       print(sample)
```
