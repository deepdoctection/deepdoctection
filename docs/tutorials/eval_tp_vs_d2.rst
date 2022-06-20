Evaluation of Tensorpack models vs Detectron2
=============================================

Summary
-------

Detectron2 is much quicker in when evaluating with two threads,
evaluation results however are worse. Decrease in performance results
from the fact that the model has been trained in Tensorpack framework.
Weights have been then transposed into Detectron2 framework where the
model however has slightly different padding mode. (This summary is
valid for evaluations

Layout
------

The following scripts shows how to evaluate and display mAP and mAR of
Tensorpack and Detectron2 models.

Due to the fact that both models work on different deep learning
libraries, it might be necessary to change and restart the kernel after
finishing the evaluation of the first model.

Detectron2 on Publaynet
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from deepdoctection.utils.fs import is_file_extension
    from deepdoctection.extern import D2FrcnnDetector
    from deepdoctection.pipe import ImageLayoutService
    from deepdoctection.eval import Evaluator, metric_registry
    from deepdoctection.datasets import get_dataset
    from deepdoctection.extern import ModelCatalog

.. code:: ipython3

    publaynet = get_dataset("publaynet")
    coco_metric = metric_registry.get("coco")

.. code:: ipython3

    path_config_yaml = ModelCatalog.get_full_path_configs("layout/d2_model-800000-layout.pkl")
    path_weights = ModelCatalog.get_full_path_weights("layout/d2_model-800000-layout.pkl")

.. code:: ipython3

    categories = publaynet.dataflow.categories.get_categories(filtered=True)
    category_names = publaynet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = D2FrcnnDetector(path_config_yaml,path_weights,categories)
    layout_service =  ImageLayoutService(layout_detector)
    evaluator = Evaluator(publaynet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,category_names=category_names)


.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.72s).
    Accumulating evaluation results...
    DONE (t=0.11s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.873
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.856
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.585
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.431
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.701
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.702
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.606
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.590
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.712


Tensorpack on Publaynet
~~~~~~~~~~~~~~~~~~~~~~~

Maybe, a restart of the kernel is necessary.

.. code:: ipython3

    from deepdoctection.extern import TPFrcnnDetector
    from deepdoctection.utils.fs import is_file_extension
    from deepdoctection.pipe import ImageLayoutService
    from deepdoctection.eval import Evaluator, metric_registry
    from deepdoctection.datasets import get_dataset
    from deepdoctection.extern import ModelCatalog

.. code:: ipython3

    publaynet = get_dataset("publaynet")
    coco_metric = metric_registry.get("coco")

.. code:: ipython3

    path_config_yaml = ModelCatalog.get_full_path_configs("layout/model-800000_inf_only.data-00000-of-00001")
    path_weights = ModelCatalog.get_full_path_weights("layout/model-800000_inf_only.data-00000-of-00001")

.. code:: ipython3

    categories = publaynet.dataflow.categories.get_categories(filtered=True)
    category_names = publaynet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = TPFrcnnDetector(path_config_yaml,path_weights,categories)
    layout_service =  ImageLayoutService(layout_detector)
    evaluator = Evaluator(publaynet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,category_names=category_names)



.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.84s).
    Accumulating evaluation results...
    DONE (t=0.15s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.892
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.928
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.922
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.755
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.744
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.929
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.546
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.907
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.909
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.787
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.774
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.939


Cell and row/column prediction
------------------------------

Detectron2 on Pubtabnet for cell predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe switch kernel again

.. code:: ipython3

    from deepdoctection.utils.fs import is_file_extension
    from deepdoctection.extern import D2FrcnnDetector
    from deepdoctection.pipe import ImageLayoutService
    from deepdoctection.eval import Evaluator, metric_registry
    from deepdoctection.datasets import get_dataset
    from deepdoctection.extern import ModelCatalog


.. parsed-literal::

    /home/janis/Public/deepdoctection_pt/venv/lib/python3.8/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    coco_metric = metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

.. code:: ipython3

    pubtabnet.dataflow.categories.filter_categories("CELL")

.. code:: ipython3

    path_config_yaml = ModelCatalog.get_full_path_configs("cell/d2_model-1800000-cell.pkl")
    path_weights = ModelCatalog.get_full_path_weights("cell/d2_model-1800000-cell.pkl")

.. code:: ipython3

    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = D2FrcnnDetector(path_config_yaml,path_weights,categories)
    layout_service =  ImageLayoutService(layout_detector)
    evaluator = Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,category_names=category_names)



.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=45.76s).
    Accumulating evaluation results...
    DONE (t=0.54s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.979
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.927
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.750
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.780
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.703
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.489
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.781
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.807
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.798
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.827
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.755


Detectron2 on Pubtabnet for row and column predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"ITEM":"row_col"})
    pubtabnet.dataflow.categories.filter_categories(["ROW","COLUMN"])

.. code:: ipython3

    path_config_yaml = ModelCatalog.get_full_path_configs("item/d2_model-1620000-item.pkl")
    path_weights = ModelCatalog.get_full_path_weights("item/d2_model-1620000-item.pkl")
    
    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = D2FrcnnDetector(path_config_yaml,path_weights,categories)
    layout_service =  ImageLayoutService(layout_detector)
    evaluator = Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,category_names=category_names, rows_and_cols=True)


.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=2.80s).
    Accumulating evaluation results...
    DONE (t=0.22s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.934
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.713
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.314
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.493
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.594
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.647
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.647
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.647
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.449
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.579
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.648


Tensorpack on Pubtabnet for cell predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from deepdoctection.extern import TPFrcnnDetector
    from deepdoctection.utils.fs import is_file_extension
    from deepdoctection.pipe import ImageLayoutService
    from deepdoctection.eval import Evaluator, metric_registry
    from deepdoctection.datasets import get_dataset
    from deepdoctection.extern import ModelCatalog

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    coco_metric = metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])
    pubtabnet.dataflow.categories.filter_categories("CELL")

.. code:: ipython3

    path_config_yaml = ModelCatalog.get_full_path_configs("cell/model-1800000_inf_only.data-00000-of-00001")
    path_weights = ModelCatalog.get_full_path_weights("cell/model-1800000_inf_only.data-00000-of-00001")

.. code:: ipython3

    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = TPFrcnnDetector(path_config_yaml,path_weights,categories)
    layout_service =  ImageLayoutService(layout_detector)
    evaluator = Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,category_names=category_names)


.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=44.42s).
    Accumulating evaluation results...
    DONE (t=0.51s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.960
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.936
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.792
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.845
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.836
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.529
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.830
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.858
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.835
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.880
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.866


Detectron2 on Pubtabnet for row and column predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"ITEM":"row_col"})
    pubtabnet.dataflow.categories.filter_categories(["ROW","COLUMN"])
    
    coco_metric = metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

.. code:: ipython3

    path_config_yaml = ModelCatalog.get_full_path_configs("item/model-1620000_inf_only.data-00000-of-00001")
    path_weights = ModelCatalog.get_full_path_weights("item/model-1620000_inf_only.data-00000-of-00001")
    
    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = TPFrcnnDetector(path_config_yaml,path_weights,categories)
    layout_service =  ImageLayoutService(layout_detector)
    evaluator = Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,category_names=category_names, rows_and_cols=True)


.. parsed-literal::

    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=2.86s).
    Accumulating evaluation results...
    DONE (t=0.23s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.953
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.940
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.681
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.714
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.880
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.904
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.904
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.904
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.726
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.769
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.909


