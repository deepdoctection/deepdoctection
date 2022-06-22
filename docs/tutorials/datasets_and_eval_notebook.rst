Datasets and Evaluation
=======================

In this tutorial we will deal with datasets, which are a separate module
within the package.

Then we will show how to evaluate a predictor in relation to a dataset
and a metric.

Caution! From v0.14 onwards this notebook will run only when the extended package for evaluation, datasets and training
is installed! Check the installation guidelines or the README for further informations.

Dataset concept
---------------

A dataset is a class that consists of a general info part, a category
part and a dataflow. The dataflow can be used to stream the data points
of the dataset.

Since the focus is on document image analysis, data points from datasets
must consist of at least one image. In addition, annotations can be
available for each data point. Generally, these are image annotations,
such as objects, with bouding boxes. However, depending on the data
point, other information can also be contained in the annotations, such
as text, relations between annotations, etc.

Dataflow is an external package that can be used to load data and
transport it through a framework. Dataflows are successive executions of
generators. Each execution can be used to perform specific tasks such as
loading or mapping. The flexibility of a dataflow is that you can
transport any Python object. There are also components that allow
certain tasks to be executed in multiple processes.

Dataflows are used to transport data points of a data set. Each data set
contains a build method, where a dataflow is returned. Over this
successive data points can be streamed. The data points of a dataset are
passed in a normalized data format. Independent of how the annotations
are stored in the source data (whether in Coco format, Pascal-VOC
format,…), they are output in the form of a so-called image data point
via Dataflow. This avoids that different mappings can be executed
independently and datasets can be processed through pipelines without
further adaptation.

By default, all available information is always provided for a data
point via the dataflow. If you want to suppress certain information, you
can configure this accordingly in the data set.

We will now make use of a builtin datasets. We use a factory function to
have immediate access to the built-in dataset. Note, that there is no
automatism to download, extract and save the datasets. We will show you
how to get the required details.

.. code:: ipython3

    from deepdoctection.datasets import print_dataset_infos, get_dataset
    from deepdoctection.datasets.instances import pubtabnet as pt
    from deepdoctection.extern import ModelCatalog
    from deepdoctection.eval import MetricRegistry, Evaluator
    from deepdoctection.extern.tpdetect import TPFrcnnDetector
    from deepdoctection.pipe.layout import ImageLayoutService
    
    from matplotlib import pyplot as plt

.. code:: ipython3

    print_dataset_infos(add_license=False, add_info=False)


.. parsed-literal::

    [36m╒═════════════╕
    │ dataset     │
    ╞═════════════╡
    │ fintabnet   │
    ├─────────────┤
    │ funsd       │
    ├─────────────┤
    │ iiitar13k   │
    ├─────────────┤
    │ testlayout  │
    ├─────────────┤
    │ publaynet   │
    ├─────────────┤
    │ pubtables1m │
    ├─────────────┤
    │ pubtabnet   │
    ├─────────────┤
    │ xfund       │
    ╘═════════════╛[0m


With ``get_dataset("pubtabnet")`` we can generate an instance of this
dataset.

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    pubtabnet.dataset_info.description




.. parsed-literal::

    "PubTabNet is a large dataset for image-based table recognition, containing 568k+ images of \ntabular data annotated with the corresponding HTML representation of the tables. The table images \n are extracted from the scientific publications included in the PubMed Central Open Access Subset \n (commercial use collection). Table regions are identified by matching the PDF format and \n the XML format of the articles in the PubMed Central Open Access Subset. More details are \n available in our paper 'Image-based table recognition: data, model, and evaluation'. \nPubtabnet can be used for training cell detection models as well as for semantic table \nunderstanding algorithms. For detection it has cell bounding box annotations as \nwell as precisely described table semantics like row - and column numbers and row and col spans. \nMoreover, every cell can be classified as header or non-header cell. The dataflow builder can also \nreturn captions of bounding boxes of rows and columns. Moreover, various filter conditions on \nthe table structure are available: maximum cell numbers, maximal row and column numbers and their \nminimum equivalents can be used as filter condition"



To install the dataset, go to the url below and download the zip-file.

.. code:: ipython3

    pubtabnet.dataset_info.url




.. parsed-literal::

    'https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz?_ga=2.267291150.146828643.1629125962-1173244232.1625045842'



You will have to unzip and place the dataset in your local .cache
directory. Once extracted the dataset ought to be in the format the no
further rearraging is required. However, if you are unsure, you can get
some additional information about the physical structure by calling the
dataset modules docstring.

.. code:: ipython3

    pubtabnet.dataflow.get_workdir()

.. code:: ipython3

    print(pt.__doc__)


.. parsed-literal::

    
    Module for Pubtabnet dataset. Place the dataset as follows
    
    |    pubtabnet
    |    ├── test
    |    │ ├── PMC1.png
    |    ├── train
    |    │ ├── PMC2.png
    |    ├── val
    |    │ ├── PMC3.png
    |    ├── PubTabNet_2.0.0.jsonl
    


Dataflows
---------

We now use the build method to obtain data samples.

Let’s display a tiny fraction of annotations that is available for each
datapoint. ``datapoint_dict["annotations"][0]`` displays all
informations that are available for one cell. First of all, there is the
category_name. This represents the main category of the annotation. In
this dataset there are Cells, Rows and Columns.

In addition, there are various sub-categories for this category, which
are grouped under the sub_category heading, such as ROW_NUMBER and
COLUMN_NUMBER.

.. code:: ipython3

    df = pubtabnet.dataflow.build(split=“train”) df.reset_state() df_iter =
    iter(df) datapoint = next(df_iter) datapoint_dict = datapoint.as_dict()
    datapoint_dict[“file_name”],datapoint_dict[“location”],datapoint_dict[“image_id”],
    datapoint_dict[“annotations”][0]

Depending on the dataset, different configurations can be provided via
the build method. For example, the image itself is not loaded by
default. By passing the parameter ``load_image=True`` the image can be
passed in the dataflow.

Note, that all images are loaded with the OpenCV framework, where the
colors are stored as numpy array in BGR order. As matplotlib expects
numpy array in RGB order, we have to swap dimensions.

.. code:: ipython3

    df = pubtabnet.dataflow.build(split="train",load_image=True)
    df.reset_state()
    df_iter = iter(df)
    datapoint = next(df_iter)
    plt.figure(figsize = (15,12))
    plt.axis('off')
    plt.imshow(datapoint.image[:,:,::-1])




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f84737ffbb0>




.. image:: ./pics/output_13_1.png


It is possible to change the representation of a data point in certain
respects. For example, one can replace the category of an annotation
with one of its sub-categories.

Thus, for this dataset, for each cell there is as a sub-category with
the information whether it is a table-header or a table-body cell.
Through the method ``set_cat_to_sub_cat`` the category can be changed.

.. code:: ipython3

    pubtabnet.dataflow.categories.set_cat_to_sub_cat({“CELL”:“HEAD”}) df =
    pubtabnet.dataflow.build(split=“train”) df.reset_state() df_iter =
    iter(df) datapoint = next(df_iter) datapoint_dict = datapoint.as_dict()
    datapoint_dict[“file_name”],datapoint_dict[“location”],datapoint_dict[“image_id”],
    datapoint_dict[“annotations”][0]

This data set was used to train the cell detector of the analyzer. We
will discuss the table detection architecture in more detail later.

In the section that follows now, we will show how to measure the
performance of the detector on the validation split. Afterwards, we want
to measure the performance on another dataset that has documents from a
different domain.

Evaluations
-----------

An evaluator needs a dataset on which to run the evaluation, as well as
a predictor and a metric. The predictor must be wraped into a pipeline
component, which is why we use the ImageLayoutService.

We take the COCO metric for the problem, but define settings that
deviate from the standard. We have to consider the following issues,
which differ from ordinary object detection tasks:

-  The objects to be identified are generally smaller
-  There are many objects to identify.

Therefore, we change the maximum number of detections to consider when
calculating the mean average precision and also choose a different range
scale for segmenting the cells into the categories small, medium and
large.

We then set up the predictor, the pipeline component and the evaluator.

.. code:: ipython3

    config_yaml_path = ModelCatalog.get_full_path_configs("cell/model-1800000.data-00000-of-00001")
    weights_path = ModelCatalog.get_full_path_weights("cell/model-1800000.data-00000-of-00001")

.. code:: ipython3

    coco_metric = MetricRegistry.get_metric("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

A word about the dataset. We have already manipulated the dataset in the
previous part of the notebook by swapping categories with subcategories.
This operation cannot be undone for the dataset instance. Therefore, we
create a new instance with the ``DatasetRegistry`` and adjust the
configuration accordingly:

Since we want to have only cells and no rows and columns as annotations
in the datapoint, we filter them out.

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.filter_categories(categories="CELL")
    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    
    cell_detector = TPFrcnnDetector(config_yaml_path,weights_path,categories)
    layout_service = ImageLayoutService(cell_detector)

We start the evaluation with the ``run``. max_datapoints limits the
number of samples in the evaluation to 100 samples. The val split is
used by default. If this is not available, it must be given as an
argument along with other possible build configurations.

.. code:: ipython3

    evaluator = Evaluator(pubtabnet,layout_service, coco_metric)
    output= evaluator.run(category_names=["CELL"],max_datapoints=100)

As mentioned we are now going to evaluate the cell predictor on tables
from business documents. One difference from the previous evaluation is
the representation of the dataset. Unlike Pubtabnet where tables are
already cropped from their surronding document, the images of Fintabnet
are whole document pages with embedded tables. In order to get tables
only we can change the build mode, which is a specific implementation
for some datasets. In this case we set ``build_mode = "table"``. This
will under the hood crop the table from the image and adjust the
bounding boxes to the sub image, so that the datasets dataflow will look
like the Pubtabnet dataset. For those looking closer at the
configuration, they will also observe a second parameter
``load_image=True``. This setting is particularly necessary for this
dataset as otherwise an AssertionError will be raised, when using this
``build_mode``.

We only need to re-instantiate the evaluator.

Apart from this, the following steps are identical to those of the
previous evaluation.

.. code:: ipython3

    fintabnet = get_dataset("fintabnet")
    fintabnet.dataflow.categories.filter_categories(categories="CELL")
    
    evaluator = Evaluator(fintabnet,layout_service, coco_metric)
    output= evaluator.run(max_datapoints=100,build_mode="table",load_image=True, use_multi_proc=False)

What stands out ?

The mAP for a low IoU drops somewhat. While the mAP for higher IoUs
drops only slightly on the Pubtabnet dataset, it drops much more on the
Fintabnet dataset. This means that the cell detector has much more
problems in its precision. The reason for this is not so much that it is
fundamentally unable to detect the cells (otherwise the 0.5 IoU would be
significantly worse), but that it is more difficult for the predictor to
determine the exact size of the cell.

How to continue (3)
===================

In the last **Fine_Tune** notebook tutorial, we will discuss training a
Tensorpack Predictor on a dataset.
