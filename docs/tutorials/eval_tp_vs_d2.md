# Evaluation of Tensorpack models vs Detectron2


## Summary


Detectron2 is much quicker in when evaluating with two threads,
results however are worse. Decrease in performance results
from the fact that weights have been taken from the Tensorpack framework.
They have then been transposed into Detectron2 artefacts. Note that both model
have a slightly different padding mode.

**Update 06/22:** As training scripts are available for Detectron2 we used the
those checkpoints to resume training for a few iterations to adopt weights
to the different padding strategy. The second training
further improved the model performance by a significant amount so that
in summary we can say: Detectron2 is trains faster and performs better
than Tensorpack.

## Layout

The following scripts shows how to determine mAP (mean average precision) and mAR
(mean average recall) for Tensorpack and Detectron2 models.

Due to the fact that both models work on different deep learning
libraries, it might be necessary to stop and switch kernel a couple of times.

### Detectron2 on Publaynet


```python

    import deepdoctection as dd

    publaynet = dd.get_dataset("publaynet")
    coco_metric = dd.metric_registry.get("coco")

    path_config_yaml = dd.ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout_inf_only.pt")
    path_weights = dd.ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout_inf_only.pt")

    categories = publaynet.dataflow.categories.get_categories(filtered=True)
    category_names = publaynet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = dd.D2FrcnnDetector("layout_d2", path_config_yaml,path_weights,categories)
    layout_service =  dd.ImageLayoutService(layout_detector)
    evaluator = dd.Evaluator(publaynet, layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500)
``` 

```python 

    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.72s).
    Accumulating evaluation results...
    DONE (t=0.11s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.919
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.952
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.939
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.809
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.809
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.953
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.550
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.929
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.934
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.835
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.838
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.961
```


### Tensorpack on Publaynet

Maybe, a kernel restart is necessary.

```python

    import deepdoctection as dd

    publaynet = dd.get_dataset("publaynet")
    coco_metric = dd.metric_registry.get("coco")

    path_config_yaml = dd.ModelCatalog.get_full_path_configs("layout/model-800000_inf_only.data-00000-of-00001")
    path_weights = dd.ModelCatalog.get_full_path_weights("layout/model-800000_inf_only.data-00000-of-00001")

    categories = publaynet.dataflow.categories.get_categories(filtered=True)
    category_names = publaynet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = dd.TPFrcnnDetector("layout_tp", path_config_yaml,path_weights,categories)
    layout_service =  dd.ImageLayoutService(layout_detector)
    evaluator = dd.Evaluator(publaynet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500)
```


``` 
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
```

## Cell and row/column prediction


### Detectron2 on Pubtabnet for cell predictions

Maybe switch kernel again

```python

    import deepdoctection as dd

    pubtabnet = dd.get_dataset("pubtabnet")
    coco_metric = dd.metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])


    pubtabnet.dataflow.categories.filter_categories("CELL")


    path_config_yaml = dd.ModelCatalog.get_full_path_configs("cell/d2_model_1849999_cell_inf_only.pt")
    path_weights = dd.ModelCatalog.get_full_path_weights("cell/d2_model_1849999_cell_inf_only.pt")

    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = dd.D2FrcnnDetector("layout_d2", path_config_yaml,path_weights,categories)
    layout_service =  dd.ImageLayoutService(layout_detector)
    evaluator = dd.Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500)
``` 


``` 

    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=45.76s).
    Accumulating evaluation results...
    DONE (t=0.54s).
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=600 ] = 0.989
      Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=600 ] = 0.955
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.813
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.867
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.849
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = 0.536
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = 0.855
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=600 ] = 0.884
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=600 ] = 0.863
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=600 ] = 0.907
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=600 ] = 0.880
``` 

### Detectron2 on Pubtabnet for row and column predictions

```python

    pubtabnet = dd.get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"ITEM":"ITEM"})
    pubtabnet.dataflow.categories.filter_categories(["ROW","COLUMN"])

    path_config_yaml = dd.ModelCatalog.get_full_path_configs("item/d2_model-1620000-item.pkl")
    path_weights = dd.ModelCatalog.get_full_path_weights("item/d2_model-1620000-item.pkl")
    
    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = dd.D2FrcnnDetector("layout_d2", path_config_yaml,path_weights,categories)
    layout_service =  dd.ImageLayoutService(layout_detector)
    evaluator = dd.Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500, rows_and_cols=True)
``` 

```

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
``` 

### Tensorpack on Pubtabnet for cell predictions


```python

    import deepdoctection as dd

    pubtabnet = dd.get_dataset("pubtabnet")
    coco_metric = dd.metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])
    pubtabnet.dataflow.categories.filter_categories("CELL")

    path_config_yaml = dd.ModelCatalog.get_full_path_configs("cell/model-1800000_inf_only.data-00000-of-00001")
    path_weights = dd.ModelCatalog.get_full_path_weights("cell/model-1800000_inf_only.data-00000-of-00001")

    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = dd.TPFrcnnDetector("layout_tp", path_config_yaml,path_weights,categories)
    layout_service =  dd.ImageLayoutService(layout_detector)
    evaluator = dd.Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500)
```

```

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
``` 


### Detectron2 on Pubtabnet for row and column predictions


```python

    pubtabnet = dd.get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"ITEM":"row_col"})
    pubtabnet.dataflow.categories.filter_categories(["ROW","COLUMN"])
    
    coco_metric = dd.metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])

    path_config_yaml = dd.ModelCatalog.get_full_path_configs("item/model-1620000_inf_only.data-00000-of-00001")
    path_weights = dd.ModelCatalog.get_full_path_weights("item/model-1620000_inf_only.data-00000-of-00001")
    
    categories = pubtabnet.dataflow.categories.get_categories(filtered=True)
    category_names = pubtabnet.dataflow.categories.get_categories(filtered=True, as_dict=False)
    
    layout_detector = dd.TPFrcnnDetector("layout_tp", path_config_yaml,path_weights,categories)
    layout_service =  dd.ImageLayoutService(layout_detector)
    evaluator = dd.Evaluator(pubtabnet,layout_service, coco_metric)
    
    output= evaluator.run(max_datapoints=500,rows_and_cols=True)
``` 

```

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
``` 