Fine Tuning
===========

In this tutorial we will go over how to fine tune a model within the
**deep**\ doctection framework.

Fine-tuning is especially important for our models. This stems from the
fact that datasets on which the models were trained on, were taken from
a specific domain and thus do not cover the variance that documents can
have in layout or table structure. Predictors trained on these datasets
show excellent results on their test splits, but fall off significantly
on datasets from other domains. As long as we have not constructed a
dataset that has a larger variance in layout, we cannot avoid
fine-tuning the predictors for more specific cases for certain
applications.

Here, we will fine-tune the cell predictor that is relevant for table
extraction on the domain of tables from business documents. For this we
use the datasets from the evaluation tutorial.

Table Extraction
----------------

Before we start creating the training script, we want to say something
about how the table recognition process works, since, to the best of my
knowledge, it was not developed according to the specifications of a
scientific paper. The development of a model also depends very much on
the data available and, of course, on the taste of the developer.

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
`PubTabNet <https://github.com/ibm-aur-nlp/PubTabNet>`__ dataset we
already introduces in the previous tutorials. PubTabNet contains approx.
500K tables from the field of medical research.

In addition, we learned about Fintabnet, a dataset that contains tables
from business documents. Furthermore, we have seen that the Cell
Predictor on Fintabnet gives much weaker results than on the validation
split of Pubtabnet. Therefore, we will fine-tune the Cell Predictor on
Fintabnet in the hope that cell detection on these tables will improve.

Training Tensorpack Predictor
-----------------------------

The following steps only work for Tensorpack models and not for
Detectron2 models. We currently do not provide built-in training scripts
for Detectron2. Also note, that for training/fine-tuning an already
pre-trained model we must not use the inference-only weights as these do
not include important checkpoint information for resuming training.

Finally, please note that the following steps require a GPU.

For training, we use a script that stems from the training of the
Faster-RCNN model from Tensorpack. We use the same model as above.

.. code:: ipython3

    import os
    from deepdoctection.datasets import DatasetRegistry
    from deepdoctection.eval import MetricRegistry
    from deepdoctection.extern import ModelCatalog
    from deepdoctection.train import train_faster_rcnn

Fintabnet has a train, val and test split from which we use the first
two. For each split, we need to define the dataflow built configuration.
Even though not necessary, as already set by default within the training
script, we explicitly pass the split.

.. code:: ipython3

    path_weights= ModelCatalog.get_full_path_weights("cell/model-1800000.data-00000-of-00001")
    path_config_yaml=ModelCatalog.get_full_path_configs("cell/model-1800000.data-00000-of-00001")
    
    fintabnet = DatasetRegistry.get_dataset("fintabnet")
    fintabnet.dataflow.categories.filter_categories(categories="CELL")
    
    dataset_train = fintabnet
    build_train_config=["max_datapoints=5000","build_mode='table'","load_image=True", "use_multi_proc_strict=True","split=train"]
    
    dataset_val = fintabnet
    build_val_config = ["max_datapoints=100","build_mode='table'","load_image=True", "use_multi_proc_strict=True","split=val"]
    
    coco_metric = MetricRegistry.get_metric("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

With the following configuration we override the default training
script, which is designed for datasets with 200K data points.

We train with 5K data points as an example (cf. build configuration). As
a rule of thumb, it is reasonable to assume that each data point is run
10 times. So we set ``LR_SCHEDULE=50000`` . We take the learning rate
from comparable fine-tuning tasks and set it to ``TRAIN.BASE_LR=1e-3``.
500 data points pass through in one iteration by definition. We evaluate
and save after every 10th iteration, i.e. after one epoch.

.. code:: ipython3

    config_overwrite=["TRAIN.LR_SCHEDULE=[50000]","TRAIN.EVAL_PERIOD=20","TRAIN.CHECKPOINT_PERIOD=20","BACKBONE.FREEZE_AT=0","TRAIN.BASE_LR=1e-3"]

We can now start training. Make sure that the log directory is set
correctly. If such a directory already exists, the existing one will be
deleted and created again!

.. code:: ipython3

    train_faster_rcnn(path_config_yaml=path_config_yaml,
                      dataset_train= dataset_train,
                      path_weights=path_weights,
                      config_overwrite=config_overwrite,
                      log_dir="/home/janis/Documents/sample_train",
                      build_train_config=build_train_config,
                      dataset_val=dataset_val,
                      build_val_config=build_val_config,
                      metric=coco_metric,
                      pipeline_component_name="ImageLayoutService"
                     )
