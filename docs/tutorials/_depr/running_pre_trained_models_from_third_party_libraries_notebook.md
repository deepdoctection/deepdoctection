## Running pre-trained models from Layout-Parser

[**Layout-Parser**](https://layout-parser.github.io/) provides several models trained on datasets like Publaynet but from many different areas as well.

In this tutorial we will show you how to configure and run a model from this library within a **deep**doctection pipeline.

Layout-Parser provides some pre-trained Detectron2 models for various document layout analysis tasks. Models from other libraries are available as well, but running Detectron2 models with **deep**doctection is particularly easy because model wrappers are already available.

You can find the Layout-Parser catalog [here](https://github.com/Layout-Parser/layoutparser/blob/main/src/layoutparser/models/detectron2/catalog.py) .

Let's download `faster_rcnn_R_50_FPN_3x` trained on Publaynet. Enter the URL into your browser and the download starts. We need the model weights and the config file. 

To make it easy we suggest to save config file in **deep**doctection's `.cache` directory.

Assume you have saved the model at: 

`~/.cache/deepdoctection/weights/layoutparser/publaynet/model_final.pth`

and the config file at:

`~/.cache/deepdoctection/configs/layoutparser/publaynet/config.yml`


```python
import deepdoctection as dd
from matplotlib import pyplot as plt
```

    /home/janis/Documents/Repos/deepdoctection_pt/.venv/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    [32m[0715 10:48.10 @file_utils.py:36][0m  [32mINF[0m  [97mPyTorch version 2.1.2+cu121 available.[0m
    [32m[0715 10:48.10 @file_utils.py:74][0m  [32mINF[0m  [97mDisabling Tensorflow because USE_TORCH is set[0m


## Adding model to the `ModelCatalog`

Next we will be adding the model to the `ModelCatalog`. This is not required but it simplifies a lot of things. 
You need to pass a key, e.g. `layoutparser/publaynet/model_final.pth` and a `ModelProfile`. A `ModelProfile` stores meta data about the model. Make sure that `ModelProfile.name` and `ModelProfile.config` is equal to the relative path of the model weights and the config file. 

It is convenient to add the categories. You can find the categories of the model [here](https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/models/detectron2/catalog.py) as well.  Note however, that, unlike in Layout-Parser all categories in deepdoctection start with 1. You therefore have to increment all category ids by 1. Also please ensure that all category ids are strings.  


```python
dd.ModelCatalog.register("layoutparser/publaynet/model_final.pth",dd.ModelProfile(
            name="layoutparser/publaynet/model_final.pth",
            description="Add some infos regarding the model",
            config="layoutparser/publaynet/config.yml",
            tp_model=False,
            size = [],
            categories={
                1: dd.LayoutType.TEXT,
                2: dd.LayoutType.TITLE,
                3: dd.LayoutType.LIST,
                4: dd.LayoutType.TABLE,
                5: dd.LayoutType.FIGURE,
            },
            model_wrapper="D2FrcnnDetector",
        ))
```


```python
path_weights = dd.ModelCatalog.get_full_path_weights("layoutparser/publaynet/model_final.pth")
path_config = dd.ModelCatalog.get_full_path_configs("layoutparser/publaynet/model_final.pth")
categories = dd.ModelCatalog.get_profile("layoutparser/publaynet/model_final.pth").categories

d2_detector = dd.D2FrcnnDetector(path_config,path_weights,categories)
image_layout = dd.ImageLayoutService(d2_detector)

pipe = dd.DoctectionPipe([image_layout])

path = "/home/janis/Documents/Repos/notebooks/pics/samples"
df = pipe.analyze(path="/path/to/publaynet_dir")
df.reset_state()

df_iter = iter(df)
dp = next(df_iter)

image = dp.viz()

plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
```

![layoutparser_1.png](./_imgs/layoutparser_1.png)

You can use some additional config parameters that are only available in **deep**doctection like `NMS_THRESH_CLASS_AGNOSTIC`. This can reduce the number of overlapping layout segments.


```python
path_weights = dd.ModelCatalog.get_full_path_weights("layoutparser/publaynet/model_final.pth")
path_config = dd.ModelCatalog.get_full_path_configs("layoutparser/publaynet/model_final.pth")
categories = dd.ModelCatalog.get_profile("layoutparser/publaynet/model_final.pth").categories

d2_detector = dd.D2FrcnnDetector(path_config,path_weights,categories,config_overwrite=["NMS_THRESH_CLASS_AGNOSTIC=0.001"])
image_layout = dd.ImageLayoutService(d2_detector)

pipe = dd.DoctectionPipe([image_layout])

df = pipe.analyze(path="/path/to/publaynet")
df.reset_state()

df_iter = iter(df)
dp = next(df_iter)

image = dp.viz()

plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
```

![layoutparser_2.png](./_imgs/layoutparser_2.png)

## A second example

Let's have a look at a second example. We will be using the model trained on the NewspaperNavigator dataset. This dataset detects labels that have not been used in **deep**doctection before. As all categories are registered in a sub-class of `ObjectTypes` which in turn, is a sub-class of `Enum` we need to define a derived `ObjectTypes` class listing all new layout sections.   


```python
@dd.object_types_registry.register("NewspaperType")
class NewspaperExtension(dd.ObjectTypes):
    """Additional Newspaper labels not registered yet"""

    PHOTOGRAPH ="Photograph",
    ILLUSTRATION = "Illustration",
    MAP = "Map",
    COMIC = "Comics/Cartoon",
    EDITORIAL_CARTOON = "Editorial Cartoon",
    HEADLINE = "Headline",
    ADVERTISEMENT =  "Advertisement"
```

There are two more things one needs to take care of.

1.) If we want to display the layout result we need to characterize the type of detected sections. There are several classes. Most of them are barely bounded layout components that only contain text and we call them `Layout` structures. There are other components as well, like `Table` which inherit a more complex structure like having `Cell`s. There is a dictionary `IMAGE_ANNOTATION_TO_LAYOUTS` available that maps the `Enum` members to the specific classes.

2.) When parsing the detected components into the `Page` format we need to add the components to the top level layout sections. While this not really important for the very small pipeline we have been creating, not adding them would also prevent the layout sections to be visualized.


```python
from deepdoctection.datapoint.view import IMAGE_ANNOTATION_TO_LAYOUTS, Layout

IMAGE_ANNOTATION_TO_LAYOUTS.update({i: Layout for i in NewspaperExtension})
```

Everything else is pretty much straight forward.


```python
dd.ModelCatalog.register("layoutparser/newspaper/model_final.pth",dd.ModelProfile(
            name="layoutparser/newspaper/model_final.pth",
            description="layout detection ",
            config="layoutparser/newspaper/config.yml",
            size=[],
            tp_model=False,
            categories={1: NewspaperExtension.PHOTOGRAPH,
                        2: NewspaperExtension.ILLUSTRATION,
                        3: NewspaperExtension.MAP,
                        4: NewspaperExtension.COMIC,
                        5: NewspaperExtension.EDITORIAL_CARTOON,
                        6: NewspaperExtension.HEADLINE,
                        7: NewspaperExtension.ADVERTISEMENT},
            model_wrapper="D2FrcnnDetector",
        ))
```


```python
path_weights = dd.ModelCatalog.get_full_path_weights("layoutparser/newspaper/model_final.pth")
path_config = dd.ModelCatalog.get_full_path_configs("layoutparser/newspaper/model_final.pth")
categories = dd.ModelCatalog.get_profile("layoutparser/newspaper/model_final.pth").categories

d2_detector = dd.D2FrcnnDetector(path_config,path_weights,categories,config_overwrite=["NMS_THRESH_CLASS_AGNOSTIC=0.8","MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.1"])
image_layout = dd.ImageLayoutService(d2_detector)

page_parser = dd.PageParsingService(text_container = dd.LayoutType.WORD, # this argument is required but will not have any effect
                                    floating_text_block_categories=[layout_item for layout_item in NewspaperExtension])
pipe = dd.DoctectionPipe([image_layout],page_parsing_service = page_parser)
```

    [32m[0715 11:07.14 @detection_checkpoint.py:38][0m  [32mINF[0m  [97m[DetectionCheckpointer] Loading from /media/janis/Elements/.cache/deepdoctection/weights/layoutparser/newspaper/model_final.pth ...[0m
    [32m[0715 11:07.14 @checkpoint.py:150][0m  [32mINF[0m  [97m[Checkpointer] Loading from /media/janis/Elements/.cache/deepdoctection/weights/layoutparser/newspaper/model_final.pth ...[0m



```python
df = pipe.analyze(path="/path/to/dir/newspaper_layout")
df.reset_state()

df_iter = iter(df)
dp = next(df_iter)

image = dp.viz()

plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
```

![layoutparser_3.png](./_imgs/layoutparser_3.png)


```python
dp.layouts
```

[Layout(active=True, _annotation_id='5b0bd0dd-300c-3303-ad45-ffcb50ba5af8', category_name=<NewspaperExtension.headline>, _category_name=<NewspaperExtension.headline>, category_id='6', score=0.9875668287277222, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=14.930139541625977, uly=194.06497192382812, lrx=518.3706665039062, lry=270.4627685546875, height=76.39779663085938, width=503.4405269622803)),\n",
Layout(active=True, _annotation_id='b6bf8f1a-a62b-3958-8f9d-7c0dc7c79354', category_name=<NewspaperExtension.photograph>, _category_name=<NewspaperExtension.photograph>, category_id='1', score=0.9749446511268616, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=275.99072265625, uly=522.1495971679688, lrx=454.6565246582031, lry=775.3734741210938, height=253.223876953125, width=178.66580200195312)),
       " Layout(active=True, _annotation_id='7fd1431f-4e48-3ff5-9fe9-be7c11c11bab', category_name=<NewspaperExtension.photograph>, _category_name=<NewspaperExtension.photograph>, category_id='1', score=0.8084900379180908, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=12.471419334411621, uly=268.1596984863281, lrx=273.9352722167969, lry=506.079833984375, height=237.92013549804688, width=261.46385288238525)),
       " Layout(active=True, _annotation_id='94c10163-eb20-3694-afa0-1b6cac9efda4', category_name=<NewspaperExtension.advertisement>, _category_name=<NewspaperExtension.advertisement>, category_id='7', score=0.44110408425331116, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=8.226184844970703, uly=493.3304138183594, lrx=276.7277526855469, lry=779.7467041015625, height=286.4162902832031, width=268.5015678405762)),
       " Layout(active=True, _annotation_id='98a12ec2-4b2f-36a6-ab49-10accaac5912', category_name=<NewspaperExtension.illustration>, _category_name=<NewspaperExtension.illustration>, category_id='2', score=0.37421464920043945, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=0.0, uly=24.19247817993164, lrx=539.7670288085938, lry=261.5598449707031, height=237.36736679077148, width=539.7670288085938)),
       " Layout(active=True, _annotation_id='700c979a-4c6a-3218-ae14-fd8009768590', category_name=<NewspaperExtension.illustration>, _category_name=<NewspaperExtension.illustration>, category_id='2', score=0.23696725070476532, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=239.40988159179688, uly=81.54991149902344, lrx=298.70074462890625, lry=152.0034942626953, height=70.45358276367188, width=59.290863037109375)),
       " Layout(active=True, _annotation_id='aa7d72ce-46c5-3304-88a8-1403251ce0e3', category_name=<NewspaperExtension.advertisement>, _category_name=<NewspaperExtension.advertisement>, category_id='7', score=0.14747683703899384, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=260.8577575683594, uly=500.2983093261719, lrx=520.8097534179688, lry=781.0679321289062, height=280.7696228027344, width=259.9519958496094))]

