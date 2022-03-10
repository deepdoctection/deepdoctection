Datasets and Evaluation
=======================

In this tutorial we will deal with datasets, which are a separate module
within the package.

Then we will show how to evaluate a predictor in relation to a dataset
and a metric.

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

We will now make use of a builtin datasets. We use a DatasetRegistry to
be able to access the built-in dataset directly. Note, that there is no
automatism to download, extract and save the datasets. We will show you
how to get the required details.

.. code:: ipython3

    from deepdoctection.datasets import DatasetRegistry
    from deepdoctection.datasets.instances import pubtabnet as pt
    from deepdoctection.extern import ModelCatalog
    from deepdoctection.eval import MetricRegistry, Evaluator
    from deepdoctection.extern.tpdetect import TPFrcnnDetector
    from deepdoctection.pipe.layout import ImageLayoutService
    
    from matplotlib import pyplot as plt

.. code:: ipython3

    DatasetRegistry.print_dataset_names()


.. parsed-literal::

    ['fintabnet', 'funsd', 'iiitar13k', 'testlayout', 'publaynet', 'pubtables1m', 'pubtabnet', 'xfund']


With ``DatasetRegistry.get_dataset("pubtabnet")`` we can generate an
instance of this dataset.

.. code:: ipython3

    pubtabnet = DatasetRegistry.get_dataset("pubtabnet")
    pubtabnet.dataset_info.description




.. parsed-literal::

    "PubTabNet is a large dataset for image-based table recognition, containing 568k+ images of tabular data annotated with the corresponding HTML representation of the tables. The table images are extracted from the scientific publications included in the PubMed Central Open Access Subset (commercial use collection). Table regions are identified by matching the PDF format and the XML format of the articles in the PubMed Central Open Access Subset. More details are available in our paper 'Image-based table recognition: data, model, and evaluation'. Pubtabnet can be used for training cell detection models as well as for semantic table understanding algorithms. For detection it has cell bounding box annotations as well as precisely described table semantics like row - and column numbers and row and col spans. Moreover, every cell can be classified as header or non-header cell. The dataflow builder can also return captions of bounding boxes of rows and columns. Moreover, various filter conditions on the table structure are available: maximum cell numbers, maximal row and column numbers and their minimum equivalents can be used as filter condition"



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

    df = pubtabnet.dataflow.build(split="train")
    df.reset_state()
    df_iter = iter(df)
    datapoint = next(df_iter)
    datapoint_dict = datapoint.as_dict()
    datapoint_dict["file_name"],datapoint_dict["location"],datapoint_dict["image_id"], datapoint_dict["annotations"][0]

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

    <matplotlib.image.AxesImage at 0x7fa0e6252d90>




.. image:: ./pics/output_13_1.png


It is possible to change the representation of a data point in certain
respects. For example, one can replace the category of an annotation
with one of its sub-categories.

Thus, for this dataset, for each cell there is as a sub-category with
the information whether it is a table-header or a table-body cell.
Through the method ``set_cat_to_sub_cat`` the category can be changed.

.. code:: ipython3

    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"CELL":"HEAD"})
    df = pubtabnet.dataflow.build(split="train")
    df.reset_state()
    df_iter = iter(df)
    datapoint = next(df_iter)
    datapoint_dict = datapoint.as_dict()
    datapoint_dict["file_name"],datapoint_dict["location"],datapoint_dict["image_id"], datapoint_dict["annotations"][0]

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

    pubtabnet = DatasetRegistry.get_dataset("pubtabnet")
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


.. parsed-literal::

    [32m[0310 17:14.26 @eval.py:67][0m [32mINF[0m Building multi threading pipeline component to increase prediction throughput. Using 2 threads
    [32m[0310 17:14:26 @varmanip.py:214][0m Checkpoint path /home/janis/.cache/deepdoctection/weights/cell/model-1800000.data-00000-of-00001 is auto-corrected to /home/janis/.cache/deepdoctection/weights/cell/model-1800000.
    [32m[0310 17:14:28 @sessinit.py:86][0m [5m[31mWRN[0m The following variables are in the checkpoint, but not found in the graph: global_step, learning_rate
    [32m[0310 17:14:29 @sessinit.py:114][0m Restoring checkpoint from /home/janis/.cache/deepdoctection/weights/cell/model-1800000 ...
    INFO:tensorflow:Restoring parameters from /home/janis/.cache/deepdoctection/weights/cell/model-1800000
    [32m[0310 17:14.29 @logger.py:193][0m [32mINF[0m Loading annotations for 'val' split from Pubtabnet will take some time...
    [32m[0310 17:15.10 @logger.py:193][0m [32mINF[0m dp: 549232 is malformed, err: IndexError,
                msg: list assignment index out of range in: <frame at 0x6a75050, file '/home/janis/Public/deepdoctection/deepdoctection/mapper/pubstruct.py', line 259, code pub_to_image_uncur> will be filtered
    [32m[0310 17:15.11 @eval.py:116][0m [32mINF[0m Predicting objects...


.. parsed-literal::

    100%|██████████| 99/99 [00:11<00:00,  8.65it/s]

.. parsed-literal::

    [32m[0310 17:15.23 @eval.py:121][0m [32mINF[0m Starting evaluation...


.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=8.23s).
    Accumulating evaluation results...
    DONE (t=0.10s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.950
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.938
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.802
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.845
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.828
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.532
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.850
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.859
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.838
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.876
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.851


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

    fintabnet = DatasetRegistry.get_dataset("fintabnet")
    fintabnet.dataflow.categories.filter_categories(categories="CELL")
    
    evaluator = Evaluator(fintabnet,layout_service, coco_metric)
    output= evaluator.run(category_names=["CELL"],max_datapoints=100,build_mode="table",load_image=True, use_multi_proc=False)


.. parsed-literal::

    [32m[0310 17:16.19 @eval.py:67][0m [32mINF[0m Building multi threading pipeline component to increase prediction throughput. Using 2 threads
    [32m[0310 17:16:19 @varmanip.py:214][0m Checkpoint path /home/janis/.cache/deepdoctection/weights/cell/model-1800000.data-00000-of-00001 is auto-corrected to /home/janis/.cache/deepdoctection/weights/cell/model-1800000.
    [32m[0310 17:16:21 @sessinit.py:86][0m [5m[31mWRN[0m The following variables are in the checkpoint, but not found in the graph: global_step, learning_rate
    [32m[0310 17:16:22 @sessinit.py:114][0m Restoring checkpoint from /home/janis/.cache/deepdoctection/weights/cell/model-1800000 ...
    INFO:tensorflow:Restoring parameters from /home/janis/.cache/deepdoctection/weights/cell/model-1800000
    [32m[0310 17:16.43 @eval.py:116][0m [32mINF[0m Predicting objects...


.. parsed-literal::

    100%|██████████| 100/100 [00:07<00:00, 14.00it/s]

.. parsed-literal::

    [32m[0310 17:16.50 @eval.py:121][0m [32mINF[0m Starting evaluation...


.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=1.69s).
    Accumulating evaluation results...
    DONE (t=0.06s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.902
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.701
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.555
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.559
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.690
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.587
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.648
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.648
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.631
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.625
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.763


What stands out ?

The mAP for a low IoU drops somewhat. While the mAP for higher IoUs
drops only slightly on the Pubtabnet dataset, it drops much more on the
Fintabnet dataset. This means that the cell detector has much more
problems in its precision. The reason for this is not so much that it is
fundamentally unable to detect the cells (otherwise the 0.5 IoU would be
significantly worse), but that it is more difficult for the predictor to
determine the exact size of the cell.

How to continue
===============

In the last **Fine_Tune** notebook tutorial, we will discuss training a
Tensorpack Predictor on a dataset.
