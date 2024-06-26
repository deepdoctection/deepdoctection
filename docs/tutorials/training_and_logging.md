# Train a model on a union of datasets and log training with Weights & Biases

One problem with most Document Layout Analysis datasets is that they have very limited layout variability. This causes models to generalize poorly to other domains. 

We show how to easily merge datasets with deepdoctection and monitor the training with Weights & Biases. 

We use a union of subsets from Doclaynet and Publaynet and resume training of a pre-trained layout model.


```python
from collections import defaultdict
import numpy as np

from fvcore.transforms import ScaleTransform
import wandb

import deepdoctection as dd
```

## Step 1: Rescaling Doclaynet images

Doclaynet images are scaled to 1000x1000 pixels, wheres Publaynet images have a 4/3 height/width ratio. We therefore re-scale Doclaynet images in order to have the same width/height ratio.  


```python
@dd.curry
def scale_transform(dp, scaler, apply_image=False):
    dp._bbox = None
    dp.set_width_height(1654,2339)
    if apply_image:
        dp.image = scaler.apply_image(dp.image)
    anns = dp.get_annotation()
    boxes = np.array([ann.bounding_box.to_list(mode="xyxy") for ann in anns])
    scaled_boxes = scaler.apply_box(boxes)
    for box, ann in zip(scaled_boxes[:,],anns):
        ann.bounding_box= dd.BoundingBox(ulx=box[0],uly=box[1],lrx=box[2],lry=box[3],absolute_coords=True)
    return dp


def filter_list_images(dp):
    if dp.summary.get_sub_category(dd.LayoutType.LIST).category_id != 0:
        return None
    return dp
```

We save all resized images in DocLayNet_core/PNG_resized. We run the code for the splits `train`,`val` and `test`. You will need around 50GB of space. If this is too much, you can also play with smaller `new_h`, `new_w` values.


```python
doclaynet = dd.get_dataset("doclaynet")

df = doclaynet.dataflow.build(split="train", load_image=True)

scaler = ScaleTransform(h=1025,w=1025,new_h=2339,new_w=1654,interp="bilinear")

df = dd.MapData(df, scale_transform(scaler,True))
df.reset_state()

for idx, dp in enumerate(df):
    print(f"processing: {idx}")
    path = dd.get_dataset_dir_path() / "DocLayNet_core/PNG_resized" / dp.file_name
    cv2.imwrite(str(path),dp.image)
```

## Step 2: Merging datasets

Doclaynet and Publaynet have different labels. Therefore we re-label some Doclaynet categories in order so that we
get "text","title","list","table","figure". 

We have to re-scale coordinates to align with the new image size, we therefore pass all datapoints on the fly 
through `scale_transform`. One other thing we observed is that the style how `list` have been annotated is different to
how the annotation style in Publaynet: Doclaynet labels each list item separately in one bounding box wheres in Publaynet a `list` bounding box encloses all list items.    
    
After buffering annotations (no images (!)) we shuffle all datapoints and build `train`, `val` and `test` split.

```python
doclaynet = dd.get_dataset("doclaynet")

doclaynet.dataflow.categories.set_cat_to_sub_cat({
    dd.LayoutType.CAPTION: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.FOOTNOTE: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.FORMULA: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.LIST: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.PAGE_FOOTER: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.PAGE_HEADER: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.FIGURE: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.SECTION_HEADER: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.TABLE: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.TEXT: dd.DatasetType.PUBLAYNET,
    dd.LayoutType.TITLE: dd.DatasetType.PUBLAYNET})

doclaynet.dataflow.categories._categories_update = [dd.LayoutType.TEXT,
                                                    dd.LayoutType.TITLE,
                                                    dd.LayoutType.LIST,
                                                    dd.LayoutType.TABLE,
                                                    dd.LayoutType.FIGURE]

df_doc = doclaynet.dataflow.build(split="train", resized=True)  # requires to create resized images
df_doc_val = doclaynet.dataflow.build(split="val", resized=True)  # requires to create resized images
df_doc_test = doclaynet.dataflow.build(split="test", resized=True)  # requires to create resized images

scaler = ScaleTransform(h=1025,
                        w=1025,
                        new_h=2339,
                        new_w=1654,
                        interp="bilinear")

df_doc = dd.MapData(df_doc, filter_list_images)
df_doc = dd.MapData(df_doc, scale_transform(scaler))

df_doc_val = dd.MapData(df_doc_val, filter_list_images)
df_doc_val = dd.MapData(df_doc_val, scale_transform(scaler))

df_doc_test = dd.MapData(df_doc_test, filter_list_images)
df_doc_test = dd.MapData(df_doc_test, scale_transform(scaler))

publaynet = dd.get_dataset("publaynet")
df_pub = publaynet.dataflow.build(split="train", max_datapoints=25000)

merge = dd.MergeDataset(publaynet)
merge.explicit_dataflows(df_doc, df_doc_val, df_doc_test, df_pub)
merge.buffer_datasets()
merge.split_datasets(ratio=0.01) 
```

## Step 3: Saving the split as Artifact

To reproduce results and know what datapoint belongs to what split, we create and log an artifact that saves the mapping between `image_id` of each datapoint and its split class. 


```python
out = merge.get_ids_by_split()

table_rows=[]
for split, split_list in out.items():
    for ann_id in split_list:
        table_rows.append([split,ann_id])
table = wandb.Table(columns=["split","annotation_id"], data=table_rows)

wandb.init(project="layout_detection")

artifact = wandb.Artifact(merge.dataset_info.name, type='dataset')
artifact.add(table, "split")

wandb.log_artifact(artifact)
wandb.finish()
```

### Step 4: Setup training and monitoring training process

Next we setup our configuration and start our training. To monitor the training process with W&B dashboard 
we set `WANDB.USE_WANDB=True` as well as `WANDB.PROJECT=layout_detection`. We can stop the training process at each step we
want. To resume training, please follow Step 5.


```python
path_config = dd.ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout.pth")
path_weights = dd.ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout.pth")
categories = dd.ModelCatalog.get_profile("layout/d2_model_0829999_layout.pth").categories
coco = dd.metric_registry.get("coco")
```


```python
dd.train_d2_faster_rcnn(path_config,
                        merge,
                        path_weights,
                        ["SOLVER.IMS_PER_BATCH=2", 
                         "SOLVER.MAX_ITER=240000", 
                         "SOLVER.CHECKPOINT_PERIOD=4000",
                         "SOLVER.STEPS=(160000, 190000)", 
                         "TEST.EVAL_PERIOD=4000", 
                         "MODEL.BACKBONE.FREEZE_AT=0",
                         "WANDB.USE_WANDB=True", 
                         "WANDB.PROJECT=layout_detection"],
                         "/path/to/dir",
                         None,
                         merge,
                         None,
                         None,
                         coco,
                         "ImageLayoutService")
```

## Step 5: Resume training

If we want to resume training maybe after starting a new jupyter kernel we need to re-create our dataset split.
So we need to execute Step 2 to re-load annotations and the artifact with splits information from W&B. With 
`create_split_by_id` we can easily reconstruct our split and resume training. If we do not change the logging dir
Detectron2 will use the last checkpoint we saved and resume training from there. 


```python
wandb.init(project="layout_detection", resume=True)
artifact = wandb.use_artifact('jm76/layout_detection/merge_publaynet:v0', type='dataset')
table = artifact.get("split")

split_dict = defaultdict(list)
for row in table.data:
    split_dict[row[0]].append(row[1])
    
merge.create_split_by_id(split_dict)
```


```python
dd.train_d2_faster_rcnn(path_config,
                        merge,
                        path_weights,
                        ["SOLVER.IMS_PER_BATCH=2", "SOLVER.MAX_ITER=240000", 
                         "SOLVER.CHECKPOINT_PERIOD=4000",
                         "SOLVER.STEPS=(160000, 190000)", 
                         "TEST.EVAL_PERIOD=4000", 
                         "MODEL.BACKBONE.FREEZE_AT=0",
                         "WANDB.USE_WANDB=True", 
                         "WANDB.PROJECT=layout_detection"],
                         "/path/to/dir",
                         None,
                         merge,
                         None,
                         None,
                         coco,
                         "ImageLayoutService")
```
