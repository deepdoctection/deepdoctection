<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>



# Fine tuning

We can fine-tune several models to improve accuracy/precision/recall on our data. There are some [training scripts]
[deepdoctection.train] available that we can use straight away. The configuration below gives you a decent (not too 
long) fine tuning training schedule for an object detection model trained on Doclaynet. 

For more information about the config, check the docs for **Detectron2**. 


```python
config_overwrite=["SOLVER.MAX_ITER=100000",    # (1)
                  "TEST.EVAL_PERIOD=20000",                           
                  "SOLVER.CHECKPOINT_PERIOD=20000",                   
                  "MODEL.BACKBONE.FREEZE_AT=0",                       
                  "SOLVER.BASE_LR=1e-3",                              
                  "SOLVER.IMS_PER_BATCH=2"] # (2)  

build_train_config = ["max_datapoints=86000"]  # (3)  

dd.train_d2_faster_rcnn(path_config_yaml=config_yaml_path,
                  dataset_train= doclaynet,
                  path_weights=weights_path,
                  config_overwrite=config_overwrite,
                  log_dir="/path/to/dir",
                  build_train_config=build_train_config,
                  dataset_val=doclaynet,
                  build_val_config=None,
                  metric=coco_metric,
                  pipeline_component_name="ImageLayoutService"
                 )
```

1. Tensorpack equivalent:  TRAIN.LR_SCHEDULE=[100000], TRAIN.EVAL_PERIOD=40 (run a 500 samples * 40), 
   TRAIN.CHECKPOINT_PERIOD=40, BACKBONE.FREEZE_AT=0 (train the every layer of the backbone and do not freeze the bottom 
   layers), TRAIN.BASE_LR=1e-3.
2. If we encounter CUDA out of memory, we can reduce SOLVER.IMS_PER_BATCH to 1.
3. We can also change the setting if you want to train with less samples.


??? info "Output"

    ```
    |  category  | #box   |  category  | #box   |  category  | #box   |
    |:----------:|:-------|:----------:|:-------|:----------:|:-------|
    |   figure   | 39667  |   table    | 30070  |    list    | 161818 |
    |   title    | 171000 |    text    | 538568 |            |        |
    |   total    | 941123 |            |        |            |        |

     CUDNN_BENCHMARK: False
    DATALOADER:
      ASPECT_RATIO_GROUPING: True
      FILTER_EMPTY_ANNOTATIONS: True
      NUM_WORKERS: 4
      REPEAT_SQRT: True
      REPEAT_THRESHOLD: 0.0
      SAMPLER_TRAIN: TrainingSampler
    DATASETS:
      PRECOMPUTED_PROPOSAL_TOPK_TEST: 1000
      PRECOMPUTED_PROPOSAL_TOPK_TRAIN: 2000
      PROPOSAL_FILES_TEST: ()
      PROPOSAL_FILES_TRAIN: ()
      TEST: ('doclaynet',)
      TRAIN: ('doclaynet',)
    GLOBAL:
      HACK: 1.0
    INPUT:
      CROP:
        ENABLED: False
        SIZE: [0.9, 0.9]
        TYPE: relative_range
      FORMAT: BGR
      MASK_FORMAT: polygon
      MAX_SIZE_TEST: 1333
      MAX_SIZE_TRAIN: 1333
      MIN_SIZE_TEST: 800
      MIN_SIZE_TRAIN: (800, 1200)
      MIN_SIZE_TRAIN_SAMPLING: choice
      RANDOM_FLIP: horizontal
    MODEL:
      ANCHOR_GENERATOR:
        ANGLES: [[-90, 0, 90]]
        ASPECT_RATIOS: [[0.5, 1.0, 2.0]]
        NAME: DefaultAnchorGenerator
        OFFSET: 0.0
        SIZES: [[32], [64], [128], [256], [512]]
      BACKBONE:
        FREEZE_AT: 0
        NAME: build_resnet_fpn_backbone
      DEVICE: cuda
      FPN:
        FUSE_TYPE: sum
        IN_FEATURES: ['res2', 'res3', 'res4', 'res5']
        NORM: GN
        OUT_CHANNELS: 256
      KEYPOINT_ON: False
      LOAD_PROPOSALS: False
      MASK_ON: False
      META_ARCHITECTURE: GeneralizedRCNN
      PANOPTIC_FPN:
        COMBINE:
          ENABLED: True
          INSTANCES_CONFIDENCE_THRESH: 0.5
          OVERLAP_THRESH: 0.5
          STUFF_AREA_LIMIT: 4096
        INSTANCE_LOSS_WEIGHT: 1.0
      PIXEL_MEAN: [238.234, 238.14, 238.145]
      PIXEL_STD: [7.961, 7.876, 7.81]
      PROPOSAL_GENERATOR:
        MIN_SIZE: 0
        NAME: RPN
      RESNETS:
        DEFORM_MODULATED: False
        DEFORM_NUM_GROUPS: 1
        DEFORM_ON_PER_STAGE: [False, False, False, False]
        DEPTH: 50
        NORM: GN
        NUM_GROUPS: 32
        OUT_FEATURES: ['res2', 'res3', 'res4', 'res5']
        RES2_OUT_CHANNELS: 256
        RES5_DILATION: 1
        STEM_OUT_CHANNELS: 64
        STRIDE_IN_1X1: False
        WIDTH_PER_GROUP: 4
      RETINANET:
        BBOX_REG_LOSS_TYPE: smooth_l1
        BBOX_REG_WEIGHTS: (1.0, 1.0, 1.0, 1.0)
        FOCAL_LOSS_ALPHA: 0.25
        FOCAL_LOSS_GAMMA: 2.0
        IN_FEATURES: ['p3', 'p4', 'p5', 'p6', 'p7']
        IOU_LABELS: [0, -1, 1]
        IOU_THRESHOLDS: [0.4, 0.5]
        NMS_THRESH_TEST: 0.5
        NORM: 
        NUM_CLASSES: 80
        NUM_CONVS: 4
        PRIOR_PROB: 0.01
        SCORE_THRESH_TEST: 0.05
        SMOOTH_L1_LOSS_BETA: 0.1
        TOPK_CANDIDATES_TEST: 1000
      ROI_BOX_CASCADE_HEAD:
        BBOX_REG_WEIGHTS: ((10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 10.0, 10.0), (30.0, 30.0, 15.0, 15.0))
        IOUS: (0.5, 0.6, 0.7)
      ROI_BOX_HEAD:
        BBOX_REG_LOSS_TYPE: smooth_l1
        BBOX_REG_LOSS_WEIGHT: 1.0
        BBOX_REG_WEIGHTS: (10.0, 10.0, 5.0, 5.0)
        CLS_AGNOSTIC_BBOX_REG: True
        CONV_DIM: 256
        FC_DIM: 1024
        FED_LOSS_FREQ_WEIGHT_POWER: 0.5
        FED_LOSS_NUM_CLASSES: 50
        NAME: FastRCNNConvFCHead
        NORM: 
        NUM_CONV: 0
        NUM_FC: 2
        POOLER_RESOLUTION: 7
        POOLER_SAMPLING_RATIO: 0
        POOLER_TYPE: ROIAlignV2
        SMOOTH_L1_BETA: 0.0
        TRAIN_ON_PRED_BOXES: False
        USE_FED_LOSS: False
        USE_SIGMOID_CE: False
      ROI_HEADS:
        BATCH_SIZE_PER_IMAGE: 512
        IN_FEATURES: ['p2', 'p3', 'p4', 'p5']
        IOU_LABELS: [0, 1]
        IOU_THRESHOLDS: [0.5]
        NAME: CascadeROIHeads
        NMS_THRESH_TEST: 0.001
        NUM_CLASSES: 5
        POSITIVE_FRACTION: 0.25
        PROPOSAL_APPEND_GT: True
        SCORE_THRESH_TEST: 0.1
      ROI_KEYPOINT_HEAD:
        CONV_DIMS: (512, 512, 512, 512, 512, 512, 512, 512)
        LOSS_WEIGHT: 1.0
        MIN_KEYPOINTS_PER_IMAGE: 1
        NAME: KRCNNConvDeconvUpsampleHead
        NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS: True
        NUM_KEYPOINTS: 17
        POOLER_RESOLUTION: 14
        POOLER_SAMPLING_RATIO: 0
        POOLER_TYPE: ROIAlignV2
      ROI_MASK_HEAD:
        CLS_AGNOSTIC_MASK: False
        CONV_DIM: 256
        NAME: MaskRCNNConvUpsampleHead
        NORM: 
        NUM_CONV: 4
        POOLER_RESOLUTION: 14
        POOLER_SAMPLING_RATIO: 0
        POOLER_TYPE: ROIAlignV2
      RPN:
        BATCH_SIZE_PER_IMAGE: 256
        BBOX_REG_LOSS_TYPE: smooth_l1
        BBOX_REG_LOSS_WEIGHT: 1.0
        BBOX_REG_WEIGHTS: (1.0, 1.0, 1.0, 1.0)
        BOUNDARY_THRESH: -1
        CONV_DIMS: [-1]
        HEAD_NAME: StandardRPNHead
        IN_FEATURES: ['p2', 'p3', 'p4', 'p5', 'p6']
        IOU_LABELS: [0, -1, 1]
        IOU_THRESHOLDS: [0.3, 0.7]
        LOSS_WEIGHT: 1.0
        NMS_THRESH: 0.7
        POSITIVE_FRACTION: 0.5
        POST_NMS_TOPK_TEST: 1000
        POST_NMS_TOPK_TRAIN: 1000
        PRE_NMS_TOPK_TEST: 1000
        PRE_NMS_TOPK_TRAIN: 2000
        SMOOTH_L1_BETA: 0.0
      SEM_SEG_HEAD:
        COMMON_STRIDE: 4
        CONVS_DIM: 128
        IGNORE_VALUE: 255
        IN_FEATURES: ['p2', 'p3', 'p4', 'p5']
        LOSS_WEIGHT: 1.0
        NAME: SemSegFPNHead
        NORM: GN
        NUM_CLASSES: 54
      WEIGHTS: /media/janis/Elements/.cache/deepdoctection/weights/layout/d2_model_0829999_layout_inf_only.pt
    NMS_THRESH_CLASS_AGNOSTIC: 0.001
    OUTPUT_DIR: /home/janis/Documents/Experiments/Tests/
    SEED: -1
    SOLVER:
      AMP:
        ENABLED: False
      BASE_LR: 0.001
      BASE_LR_END: 0.0
      BIAS_LR_FACTOR: 1.0
      CHECKPOINT_PERIOD: 20000
      CLIP_GRADIENTS:
        CLIP_TYPE: value
        CLIP_VALUE: 1.0
        ENABLED: False
        NORM_TYPE: 2.0
      GAMMA: 0.1
      IMS_PER_BATCH: 2
      LR_SCHEDULER_NAME: WarmupMultiStepLR
      MAX_ITER: 100000
      MOMENTUM: 0.9
      NESTEROV: False
      NUM_DECAYS: 3
      REFERENCE_WORLD_SIZE: 0
      RESCALE_INTERVAL: False
      STEPS: (60000, 80000)
      WARMUP_FACTOR: 0.001
      WARMUP_ITERS: 1000
      WARMUP_METHOD: linear
      WEIGHT_DECAY: 0.0001
      WEIGHT_DECAY_BIAS: None
      WEIGHT_DECAY_NORM: 0.0
    TEST:
      AUG:
        ENABLED: False
        FLIP: True
        MAX_SIZE: 4000
        MIN_SIZES: (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
      DETECTIONS_PER_IMAGE: 100
      DO_EVAL: True
      EVAL_PERIOD: 20000
      EXPECTED_RESULTS: []
      KEYPOINT_OKS_SIGMAS: []
      PRECISE_BN:
        ENABLED: False
        NUM_ITER: 200
    VERSION: 2
    VIS_PERIOD: 0
    WANDB:
      PROJECT: None
      REPO: deepdoctection
      USE_WANDB: False
    ```
