<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>

# Datasets


Regardless of whether we use data to fine-tune a task, carry out an evaluation or experiment: The creation of a
dataset provides a way to make data available in a standard format so that it can be processed through components
of the library.

Let's say we want to start with a small annotation file containing some ground truth data for some images as well as 
references to each image. We load this file into memory having a list of e.g. dicts. We then need something so that 
we can iterate over each list element step by step. This is the place where generators come into place.

!!! info

    For users familiar with Pytorch datasets, datasets in **deep**doctection are related to the `IterableDataset`.
    In Pytorch you can iterate over samples like:

    ``` python
        dp = next(iter(MyPytorchDataset))
    ```

## DatasetInfo, DatasetCategories and DataFlowBuilder


In **deep**doctection, datasets have a `build` method in the `DataFlowBuilder` attribute that
returns a `DataFlow`. The `build` accepts arguments, so that we can change the representation of
datapoints up to some degree or so that we can filter some unwanted samples or reduce the size of the dataset.

A dataset consists of three components modelled as attributes: [`DatasetInfo`][dd_datasets.info.DatasetInfo], 
[`DatasetCategories`][dd_datasets.info.DatasetCategories] and a
[`DataFlowBuilder`][dd_datasets.dataflow_builder.DataFlowBaseBuilder] class that have to be implemented
individually. `build` of `DataFlowBuilder` returns a generator, a `DataFlow` instance from which data points can be 
streamed.


## Custom dataset

There is a client [`CustomDataset`][dd_datasets.base.CustomDataset] that helps us to
create quickly a dataset without implementing a large overhead. Basically, we have to write a
`DataFlowBuilder` and must instantiate a `CustomDataset`.

The easiest way is to physically store a dataset in the `.cache` directory of **deep**doctection (usually this is
`~/.cache/deepdoctection/datasets`). Create a sub folder *custom_dataset* and store you dataset in this sub folder.

We set

```python
my_custom_dataset = CustomDataset(name="some name",
								  dataset_type=dd.DatasetType.object_detection,
								  location = "custom_dataset",
								  ...)
my_custom_dataset.dataflow.get_workdir()
```

will point to the sub folder `custom_dataset`. Moreover, we have to map every dataset to a `dataset_type`. This must
be one of the members of the `DatasetType`. The most crucial part is to build a a sub class of 
the `DataFlowBaseBuilder`.

## Custom `DataflowBuilder`

```python

class CustomDataflow(DataFlowBaseBuilder):

	def build(**kwargs):

		path =  self.get_workdir() / annotation_file.jsonl
		df = SerializerJsonLines.load(path)                      # will stream every .json linewise
		...
		return df
```

Note, that `build` must yield an [`Image`][dd_core.datapoint.image.Image]. It is therefore crucial to map the
data structure of the annotation file into an `Image`. Fortunately, there are already some mappings made available.
For COCO-style annotation, we can simply do:

```python

class CustomDataflow(DataFlowBaseBuilder):

	def build(**kwargs):

		path =  self.get_workdir() / annotation_file.json
		df = SerializerCoco.load(path) # (1) 
		 
		coco_mapper = coco_to_image(self.categories.get_categories(init=True), # (2)
									 load_image= False)
		df = MapData(df, coco_mapper)
		return df
```

1. This will load a coco style annotation file and combine image and their annotations.
2. A callable with some configuration (mapping category ids and category names/ skipping the image loading)


This dataflow has a very basic behaviour. We can add some more functionalities, e.g. filtering some categories.


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
		...
		return df
```


Having added this to our dataflow, we can now customize our categories:

```python

my_custom_dataset = CustomDataset("train_data",
								   DatasetType.object_detection,
								   "custom_dataset_location",
								   [LayoutType.TEXT, LayoutType.TITLE, LayoutType.TABLE],
								   CustomDataflow("custom_dataset_location",{"train": "annotation_file.json"}))

my_custom_dataset.dataflow.categories.filter_categories(categories="table")

df = my_custom_dataset.dataflow.build()
df.reset_state()
for dp in df:
	... # (1) 
```

1. dp has now only has 'table' labels in our samples. 'text' and 'title' has been filtered out.

!!! info "How to build datasets the long way"

    We assume that in *custom_dataset* the data set was physically placed in the following the structure:

    ```
    custom_dataset
    ├── train
    │├── 01.png
    │├── 02.png
    ├── gt_train.json
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
    
    
    Three methods `_info`, `_categories` and `_builder` must be implemented for a dataset, each of which
    return an instance `DatasetInfo`, `DatasetCategories` or `None` and a class derived from `DataFlowBaseBuilder`.
    
    ### `DatasetInfo`

    A [`DatasetInfo`][deepdoctection.datasets.info.DatasetInfo] instance must be returned. `DatasetInfo` essentially 
    only stores attributes that have informative characters. The instance must be created, but almost all
    arguments can be defaulted.
    
    ### `DatasetCategories`
    
    
    [`DatasetCategories`][deepdoctection.datasets.info.DatasetCategories] provide a way to manage categories and 
    sub-categories. This proves to be useful if, for example, we want to filter out certain categories in the 
    dataset. Another application arises, if we have annotations with categories and sub-categories in the dataset 
    and want to see annotations labeled with their sub-category name instead of their category name.
    
    `DatasetCategories` takes as argument a list of `init_categories`, with category names. Category names must be
    members of an `Enum` class that must be derived from [`ObjectTypes`][deepdoctection.utils.settings.ObjectTypes]. 
    If there are sub-categories, `init_sub_categories` returns a dict with category names as key and a list of 
    sub-category names as value.
    
    **Example:** 
    In the annotation file there is a category "TABLE_CELL", where "TABLE_CELL" can contain two possible
    sub categories "TABLE_HEADER" and "TABLE_BODY". Suppose there are no more categories and sub categories. Then we
    define a `ObjectTypes` for new categories and initialize `DatasetCategories`.
    
    ```python
    
	@dd.object_types_registry.register("TableCellType")  # (1)
	class CellType(ObjectTypes):
		TABLE_CELL = "table_cell"
		TABLE_HEADER = "table_header"
		TABLE_BODY = "table_body"

	dd.DatasetCategories(init_categories=[CellType.TABLE_CELL],
					  init_sub_categories={CellType.TABLE_CELL:[CellType.TABLE_HEADER, CellType.TABLE_BODY]}).
    
    ```

    1. Using the `object_types_registry.register` decorator we make the new `ObjectTypes` available in the
	   `ObjectTypes` registry. This allows us to get a member by its string using `get_type`: 
       `get_type('table_cell')=CellType.TABLE_CELL`.
    
    When initializing `DatasetCategories` it is important to know the metadata of the dataset annotation file
    (available labels etc.) otherwise, logical errors can occur quickly. That means, if we are in doubt, what
    categories might occur, or how sub-categories are related to categories, it is worth the time to perform a quick
    analysis on the annotation file.
    
    ### `DataflowBuilder`

    The dataflow builder is the tool to create a stream for the dataset. The base class contains an abstract method
    `build`. The following has to be implemented:
    
    - Loading a data point (e.g. ground truth data and additional components, such as an image or a path) in raw format.
    - Transforming the raw data into the core data model.
    
    Various tools are available for loading and transforming. If the ground truth is in COCO format,
    for example, the annotation file can be loaded with SerializerCoco. The instance returns a data flow through which 
    each sample is streamed individually.
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

We can print a list of all built-in datsets:


```python
dd.print_dataset_infos(add_license=False, add_info=False)
```

??? info "Output"

    ```
    ╒════════════════════╕
    │ dataset            │
    ╞════════════════════╡
    │ doclaynet          │
    ├────────────────────┤
    │ doclaynet-seq      │
    ├────────────────────┤
    │ fintabnet          │
    ├────────────────────┤
    │ funsd              │
    ├────────────────────┤
    │ iiitar13k          │
    ├────────────────────┤
    │ testlayout         │
    ├────────────────────┤
    │ publaynet          │
    ├────────────────────┤
    │ pubtables1m_det    │
    ├────────────────────┤
    │ pubtables1m_struct │
    ├────────────────────┤
    │ pubtabnet          │
    ├────────────────────┤
    │ rvl-cdip           │
    ├────────────────────┤
    │ xfund              │
    ╘════════════════════╛
    ``

With `get_dataset("doclaynet")` we can create an instance of a built-in dataset.


```python
doclaynet = dd.get_dataset("doclaynet")

print(doclaynet.dataset_info.description)
```

??? info "Output"

    DocLayNet is a human-annotated document layout segmentation dataset containing 80863 pages from a broad variety of 
    document sources. DocLayNet provides page-by-page layout segmentation ground-truth using bounding-boxes for 11 
    distinct class labels on 80863 unique pages from 6 document categories. It provides several unique features compared 
    to related work such as PubLayNet or DocBank: Human Annotation: DocLayNet is hand-annotated by well-trained experts, 
    providing a gold-standard in layout segmentation through human recognition and interpretation of each page layout 
    Large layout variability: DocLayNet includes diverse and complex layouts from a large variety of public sources in 
    Finance, Science, Patents, Tenders, Law texts and Manuals Detailed label set: DocLayNet defines 11 class labels to 
    distinguish layout features in high detail. 
    Redundant annotations: A fraction of the pages in DocLayNet are double- or triple-annotated, allowing to estimate 
    annotation uncertainty and an upper-bound of achievable prediction accuracy with ML models Pre-defined train- test- 
    and validation-sets: DocLayNet provides fixed sets for each to ensure proportional representation of the 
    class-labels and avoid leakage of unique layout styles across the sets.


In **deep**doctection there is no function that automatically downloads a dataset from its remote storage.  


To install the dataset, we go to the url below and download the zip-file. We will then have to unzip and place the 
dataset in our local **.cache/deepdoctection/dataset** directory.  


```python
doclaynet.dataset_info.url
```

??? info "Output"

    'https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip'

```python
print(dd.datasets.instances.doclaynet.__doc__)
```

??? info  "Output"  

    Module for DocLayNet dataset. Place the dataset as follows
    
        DocLayNet_core
        ├── COCO
        │ ├── test.json
        │ ├── val.json
        ├── PNG
        │ ├── 0a0d43e301facee9e99cc33b9b16e732dd207135f4027e75f6aea2bf117535a2.png
    
To produce samples, we need to instantiate a dataflow. Most built-in datasets have a split and can stream
datapoint samples from the specified split.


```python
df = doclaynet.dataflow.build(split="train") # (1)
df.reset_state() 

df_iter = iter(df) 

datapoint = next(df_iter)
datapoint_dict = datapoint.as_dict() 
datapoint_dict["file_name"],datapoint_dict["_image_id"], datapoint_dict["annotations"][0]
```

1. Instantiate the dataflow

??? info "Output"

    ```python
        ('c6effb847ae7e4a80431696984fa90c98bb08c266481b9a03842422459c43bdd.png',
     '4385125b-dd1e-3025-880f-3311517cc8d5',
     {'active': True,
      'external_id': 0,
      '_annotation_id': '4385125b-dd1e-3025-880f-3311517cc8d5',
      'service_id': None,
      'model_id': None,
      'session_id': None,
      'category_name': LayoutType.PAGE_HEADER,
      '_category_name': LayoutType.PAGE_HEADER,
      'category_id': 6,
      'score': None,
      'sub_categories': {DatasetType.PUBLAYNET: {'active': True,
        'external_id': None,
        '_annotation_id': '4f10073e-a211-3336-8347-8b34e8a2e59a',
        'service_id': None,
        'model_id': None,
        'session_id': None,
        'category_name': LayoutType.TITLE,
        '_category_name': LayoutType.TITLE,
        'category_id': 11,
        'score': None,
        'sub_categories': {},
        'relationships': {}}},
      'relationships': {},
      'bounding_box': {'absolute_coords': True,
       'ulx': 72,
       'uly': 55,
       'lrx': 444,
       'lry': 75},
      'image': None})
    ```

!!! info "Note on the `build` method"

    Depending on the dataset, different configurations of the `build` method can yield different representations of 
    datapoints. For example, the underlying image is not loaded by default. Passing the parameter `load_image=True` will 
    load the image numpy array.

!!! note "Standard image format in **deep**doctection is 'BGR'"

    Under the hood **deep*doctection uses OpenCV or Pillow for loading images, depending what library is installed and
    what is configured. If you have your own image loader and want to pass a numpy array to an **deep**doctection 
    object, make sure that you pass the numpy array in `BGR` format.


```python
df = doclaynet.dataflow.build(split="train",load_image=True)
df.reset_state()

df_iter = iter(df)
datapoint = next(df_iter)

plt.figure(figsize = (15,12))
plt.axis('off')
plt.imshow(datapoint.image[:,:,::-1])
```


    
![png](../_imgs/datasets_01.png)
    