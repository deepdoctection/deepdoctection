Some Tensorpack trainings scripts
----------------------

Training script for cell detection task. Use TRAIN.LR_SCHEDULE=2x for
training from scratch else TRAIN.LR_SCHEDULE=1x for training from last
checkpoint.

.. code:: ipython3

    import os
    from deepdoctection.datasets import get_dataset
    from deepdoctection.eval import metric_registry
    from deepdoctection.utils import get_configs_dir_path
    from deepdoctection.train import train_faster_rcnn

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.filter_categories(categories="CELL")
    
    path_config_yaml=os.path.join(get_configs_dir_path(),"tp/cell/conf_frcnn_cell.yaml")
    path_weights = "/path/to/dir/model-3540000.data-00000-of-00001"
    
    dataset_train = pubtabnet
    config_overwrite=["TRAIN.STEPS_PER_EPOCH=500","TRAIN.EVAL_PERIOD=20","TRAIN.STARTING_EPOCH=141",
                      "PREPROC.TRAIN_SHORT_EDGE_SIZE=[400,600]","TRAIN.CHECKPOINT_PERIOD=20",
                      "BACKBONE.FREEZE_AT=0"]
    build_train_config=["max_datapoints=500000"]
    dataset_val = pubtabnet
    build_val_config = ["max_datapoints=4000"]
    
    coco_metric = metric_registry.get("coco")
    coco_metric.set_params(max_detections=[50,200,600], area_range=[[0,1000000],[0,200],[200,800],[800,1000000]])
    
    train_faster_rcnn(path_config_yaml=path_config_yaml,
                      dataset_train=dataset_train,
                      path_weights=path_weights,
                      config_overwrite=config_overwrite,
                      log_dir="/path/to/dir/train",
                      build_train_config=build_train_config,
                      dataset_val=dataset_val,
                      build_val_config=build_val_config,
                      metric=coco_metric,
                      pipeline_component_name="ImageLayoutService"
                      )

Training script for row/column detection task. Use TRAIN.LR_SCHEDULE=2x
for training from scratch else TRAIN.LR_SCHEDULE=1x for training from
last checkpoint.

.. code:: ipython3

    pubtabnet = get_dataset("pubtabnet")
    pubtabnet.dataflow.categories.set_cat_to_sub_cat({"ITEM":"row_col"})
    pubtabnet.dataflow.categories.filter_categories(["ROW","COLUMN"])
    
    path_config_yaml=os.path.join(get_configs_dir_path(),"tp/rows/conf_frcnn_rows.yaml")
    path_weights = os.path.join(get_weights_dir_path(),"item/model-1750000.data-00000-of-00001")
    
    dataset_train = pubtabnet
    
    config_overwrite=["TRAIN.STEPS_PER_EPOCH=5000","TRAIN.EVAL_PERIOD=20","TRAIN.STARTING_EPOCH=1",
                       "PREPROC.TRAIN_SHORT_EDGE_SIZE=[400,600]","TRAIN.CHECKPOINT_PERIOD=20",
                       "BACKBONE.FREEZE_AT=0"]
    
    build_train_config=["max_datapoints=500000","rows_and_cols=True"]
    
    dataset_val = pubtabnet
    build_val_config = ["max_datapoints=4000","rows_and_cols=True"]
    
    train_faster_rcnn(path_config_yaml=path_config_yaml,
                       dataset_train=pubtabnet,
                       path_weights=path_weights,
                       config_overwrite=config_overwrite,
                       log_dir="/path/to/dir/train",
                       build_train_config=build_train_config,
                       dataset_val=dataset_val,
                       build_val_config=build_val_config,
                       metric_name="coco",
                       pipeline_component_name="ImageLayoutService"
                       )
