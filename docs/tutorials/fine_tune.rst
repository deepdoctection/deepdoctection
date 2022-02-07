Evaluating and Fine Tuning
==========================

Intro
------------

We show how a model can be fine-tuned for a specific task and how the
performance can be compared with the pre-trained model.

For this purpose, we want to try to improve the table extraction in the
**deep**\ doctection analyzer as an example. To better understand what
we are trying to address, we need to say a little more about processing
table extraction.

.. figure:: ./pics/dd_table.png
   :alt: title

   title

Table extraction is carried out in different stages:

-  Table detection
-  Cell detection
-  Row and column detection
-  Segmentation / cell labeling

Tables, cells and rows / columns are recognized with object detectors
(Cascade-RCNN with FPN). The segmentation is carried out by determining
the coverage of cells to rows and columns and is rule-based.

Cell recognition was carried out on the
`PubTabNet <https://github.com/ibm-aur-nlp/PubTabNet>`__ dataset.
PubTabNet contains approx. 500K tables from the field of medical
research.

The current model was configured in such a way that not only table cells
are recognized, but header cells are also differentiated from body
cells.

We want to train the model further on the same dataset. However, we want
to replace the header of the model and only recognize cells without any
further distinction. Furthermore, want to keep the number of pixels in
the image larger in the hope that relatively small cells can be better
recognized.

Dataset
-------

In order to fine-tune on your own data set, you should create your one
dataset instance based on the example of the existing datasets. We use a
DatasetRegistry to be able to access the built-in dataset directly.
Before we start fine tuning, let’s take a look at the dataset. It will
show you the advantage of the concept within this framework and how it
easily integrates into the training scripts.

.. code:: ipython3

    import os
    
    from deep_doctection.utils import get_weights_dir_path,get_configs_dir_path
    from deep_doctection.datasets import DatasetRegistry
    from deep_doctection.eval import MetricRegistry, Evaluator
    from deep_doctection.extern.tpdetect import TPFrcnnDetector
    from deep_doctection.pipe.layout import ImageLayoutService

.. code:: ipython3

    DatasetRegistry.print_dataset_names()


.. parsed-literal::

    ['fintabnet', 'funsd', 'testlayout', 'publaynet', 'pubtabnet', 'xfund']


.. code:: ipython3

    pubtabnet = DatasetRegistry.get_dataset("pubtabnet")
    pubtabnet.dataset_info.description




.. parsed-literal::

    "PubTabNet is a large dataset for image-based table recognition, containing 568k+ images of tabular data annotated with the corresponding HTML representation of the tables. The table images are extracted from the scientific publications included in the PubMed Central Open Access Subset (commercial use collection). Table regions are identified by matching the PDF format and the XML format of the articles in the PubMed Central Open Access Subset. More details are available in our paper 'Image-based table recognition: data, model, and evaluation'. Pubtabnet can be used for training cell detection models as well as for semantic table understanding algorithms. For detection it has cell bounding box annotations as well as precisely described table semantics like row - and column numbers and row and col spans. Moreover, every cell can be classified as header or non-header cell. The dataflow builder can also return captions of bounding boxes of rows and columns. Moreover, various filter conditions on the table structure are available: maximum cell numbers, maximal row and column numbers and their minimum equivalents can be used as filter condition"



We refer to the in depths tutorial for more details about the
construction of datasets and the architecture of **deep**\ doctection.
Nevertheless, we will briefly go into the individual steps to display a
sample from Pubtabnet. The dataset has a method dataflow.build that
returns a generator where samples can be streamed from.

Let’s display a tiny fraction of annotations that is available for each
datapoint. ``df_dict["annotations"][0]`` displays all informations that
are available for one cell, i.e. sub categories, like row and column
number, header information and bounding boxes.

.. code:: ipython3

    df = pubtabnet.dataflow.build(split="train")
    df.reset_state()
    df_iter = iter(df)
    df_dict = next(df_iter).as_dict()
    df_dict["file_name"],df_dict["location"],df_dict["image_id"], df_dict["annotations"][0]




.. parsed-literal::

    ('PMC4840965_004_00.png',
     '/home/janis/Public/deepdoctection/datasets/pubtabnet/train/PMC4840965_004_00.png',
     'c87ee674-4ddc-3efe-a74e-dfe25da5d7b3',
     {'active': True,
      'annotation_id': '84cbfafb-c878-323a-afcf-6159206f2e49',
      'category_name': 'CELL',
      'category_id': '1',
      'score': None,
      'sub_categories': {'ROW_NUMBER': {'active': True,
        'annotation_id': '37cd395e-a09d-3f73-b7e5-98c0d284c75f',
        'category_name': 'ROW_NUMBER',
        'category_id': '28',
        'score': None,
        'sub_categories': {},
        'relationships': {}},
       'COLUMN_NUMBER': {'active': True,
        'annotation_id': '626c0980-5a45-3223-b7c8-39bc3648722c',
        'category_name': 'COLUMN_NUMBER',
        'category_id': '3',
        'score': None,
        'sub_categories': {},
        'relationships': {}},
       'ROW_SPAN': {'active': True,
        'annotation_id': '02458dd5-e774-3cf6-a299-5546d9c63880',
        'category_name': 'ROW_SPAN',
        'category_id': '1',
        'score': None,
        'sub_categories': {},
        'relationships': {}},
       'COLUMN_SPAN': {'active': True,
        'annotation_id': '87df3823-d8f8-3839-ae67-2690f1ff0379',
        'category_name': 'COLUMN_SPAN',
        'category_id': '1',
        'score': None,
        'sub_categories': {},
        'relationships': {}},
       'HEAD': {'active': True,
        'annotation_id': '446896bf-f176-349b-bd46-d41aa3397dbb',
        'category_name': 'BODY',
        'category_id': '<property object at 0x7f102ab0f770>',
        'score': None,
        'sub_categories': {},
        'relationships': {}}},
      'relationships': {},
      'bounding_box': {'absolute_coords': True,
       'ulx': 336.0,
       'uly': 381.0,
       'lrx': 376.0,
       'lry': 391.0},
      'image': None})



“CELL” label is the main category. It is possible to change the
representation of an annotation by swapping categories with sub
categories.

.. code:: ipython3

    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"CELL":"HEAD"})


.. parsed-literal::

    [32m[1221 17:51.36 @info.py:205][0m [32mINF[0m Will reset all previous updates


.. code:: ipython3

    df = pubtabnet.dataflow.build(split="train")
    df.reset_state()
    df_iter = iter(df)
    df_dict = next(df_iter).as_dict()
    df_dict["annotations"][0]




.. parsed-literal::

    {'active': True,
     'annotation_id': '84cbfafb-c878-323a-afcf-6159206f2e49',
     'category_name': 'BODY',
     'category_id': '2',
     'score': None,
     'sub_categories': {'ROW_NUMBER': {'active': True,
       'annotation_id': '37cd395e-a09d-3f73-b7e5-98c0d284c75f',
       'category_name': 'ROW_NUMBER',
       'category_id': '28',
       'score': None,
       'sub_categories': {},
       'relationships': {}},
      'COLUMN_NUMBER': {'active': True,
       'annotation_id': '626c0980-5a45-3223-b7c8-39bc3648722c',
       'category_name': 'COLUMN_NUMBER',
       'category_id': '3',
       'score': None,
       'sub_categories': {},
       'relationships': {}},
      'ROW_SPAN': {'active': True,
       'annotation_id': '02458dd5-e774-3cf6-a299-5546d9c63880',
       'category_name': 'ROW_SPAN',
       'category_id': '1',
       'score': None,
       'sub_categories': {},
       'relationships': {}},
      'COLUMN_SPAN': {'active': True,
       'annotation_id': '87df3823-d8f8-3839-ae67-2690f1ff0379',
       'category_name': 'COLUMN_SPAN',
       'category_id': '1',
       'score': None,
       'sub_categories': {},
       'relationships': {}},
      'HEAD': {'active': True,
       'annotation_id': 'e9749db2-464c-3144-84f9-f939a4f15a43',
       'category_name': 'BODY',
       'category_id': '<property object at 0x7f63b575f770>',
       'score': None,
       'sub_categories': {},
       'relationships': {}}},
     'relationships': {},
     'bounding_box': {'absolute_coords': True,
      'ulx': 336.0,
      'uly': 381.0,
      'lrx': 376.0,
      'lry': 391.0},
     'image': None}



Evaluating model
-----------------

We want to evaluate the current model and use the evaluator framework
for this. An evaluator needs a dataset on which to run the evaluation,
as well as a predictor and a metric. The predictor must be wraped into a
pipeline component, which is why we use the ImageLayoutService.

We take the COCO metric for the problem, but define settings that
deviate from the standard. We have to consider the following issues,
which differ from ordinary object detection tasks:

-  The objects to be identified are generally smaller
-  There are many objects to identify.

.. code:: ipython3

    coco_metric = MetricRegistry.get_metric("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

.. code:: ipython3

    path_config_yaml=os.path.join(get_configs_dir_path(),"tp/cell/conf_frcnn_cell.yaml")
    path_weights = os.path.join(get_weights_dir_path(),"cell/model-2840000.data-00000-of-00001")
    
    
    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    cell_detector = TPFrcnnDetector(path_config_yaml,path_weights,categories)
    
    layout_service =  ImageLayoutService(cell_detector)
    evaluator = Evaluator(pubtabnet,layout_service, coco_metric)


We start the evaluation with the run method. max_datapoints limits the
number of samples in the evaluation to 100 data records. The val split
is used by default. If this is not available, it must be given as an
argument along with other possible build configurations.

.. code:: ipython3

    output= evaluator.run(category_names=["HEAD","BODY"],max_datapoints=100)


.. parsed-literal::

    [32m[1221 17:52.32 @logger.py:193][0m [32mINF[0m Loading annotations for 'val' split from Pubtabnet will take some time...
    [32m[1221 17:53.14 @logger.py:193][0m [32mINF[0m dp: 549232 is malformed, err: IndexError,
                msg: list assignment index out of range in: <frame at 0x6bdcd40, file '/home/janis/Public/deepdoctection/deep_doctection/mapper/pubstruct.py', line 258, code pub_to_image_uncur> will be filtered
    [32m[1221 17:53.15 @eval.py:132][0m [32mINF[0m Predicting objects...


.. parsed-literal::

    100%|██████████| 99/99 [00:12<00:00,  7.68it/s]

.. parsed-literal::

    [32m[1221 17:53.28 @eval.py:137][0m [32mINF[0m Starting evaluation...

.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=7.36s).
    Accumulating evaluation results...
    DONE (t=0.12s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.930
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.768
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.590
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.689
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.644
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.584
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.708
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.711
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.665
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.741
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.695


Training Tensorpack Predictor
-----------------------------

For the training, we use a training script that stems from the training
of the Faster-RCNN model from Tensorpack. Let’s collect all necessary
inputs:

-  We take the model config of the cell detector. It is important to
   note that the hyperparameter for this detector differs slightly from
   the standard Faster-RCNN config, taking into account that cells are
   generally smaller and have a length/height ratio >=1.

-  We take the pre-trained cell weights.

-  Since we are completely replacing the model head (we are changing the
   number of categories) we have to plan a longer training schedule. We
   use the standard training schedule 1xDetectron, which corresponds to
   a training schedule for a detection task with pre-trained backbone.
   This training schedule takes about 2.5 days on a GPU (RTX 3090) and
   is already included in the configs and therefore does not need to be
   passed explicitly. The most important training configurations, such
   as the learning rate schedule, are also derived from this
   specification.

-  In the configs we overwrite some configurations for callbacks and the
   trainable variables: We train all the variables of the backbone as we
   change the image size. We evaluate and save the model every 20
   epochs. (Attention: An epoch is defined differently here than the
   passage of a dataset).

.. code:: ipython3

    from deep_doctection.train import train_faster_rcnn
    
    
    path_config_yaml=os.path.join(get_configs_dir_path(),"tp/cell/conf_frcnn_cell.yaml")
    path_weights = os.path.join(get_weights_dir_path(),"cell/model-2840000.data-00000-of-00001")
    
    
    config_overwrite=["TRAIN.EVAL_PERIOD=20","PREPROC.TRAIN_SHORT_EDGE_SIZE=[400,600]","TRAIN.CHECKPOINT_PERIOD=20","BACKBONE.FREEZE_AT=0"]

The other configs refer to dataset and metric settings we discussed
before.

.. code:: ipython3

    pubtabnet = DatasetRegistry.get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.filter_categories(categories="CELL")
    dataset_train = pubtabnet
    
    build_train_config=["max_datapoints=500000"]
    
    dataset_val = pubtabnet
    build_val_config = ["max_datapoints=4000"]
    
    coco_metric = MetricRegistry.get_metric("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

We can now start training. Make sure that the log directory is set
correctly. If such a directory already exists, the existing one will be
deleted and created again!

.. code:: ipython3

    train_faster_rcnn(path_config_yaml=path_config_yaml,
                      dataset_train=pubtabnet,
                      path_weights=path_weights,
                      config_overwrite=config_overwrite,
                      log_dir="/path/to/log_dir",
                      build_train_config=build_train_config,
                      dataset_val=dataset_val,
                      build_val_config=build_val_config,
                      metric=coco_metric,
                      pipeline_component_name="ImageLayoutService"
                      )
