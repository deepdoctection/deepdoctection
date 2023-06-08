# Pipelines

Let's now take a closer look at pipelines. We will be using the **deep**doctection analyzer as an example. 

A pipeline is built as a sequence of tasks. These tasks are called pipeline components, pipeline backbones or services.

![pipelines](./_imgs/dd_overview_pipeline.png)

Once a pipeline is defined, images or documents can be processed. These are either pure image files (like JPG, PNG, TIFF) or PDF files. PDF files are read and processed page by page. Each PDF page is converted into a numpy array. 

We do not want to go into detail about the data structure at this point. If you want more information, please refer to the notebook **Diving_deeper_into_the_data_structure.ipynb**. 


```python
import deepdoctection as dd
```


```python
analyzer = dd.get_dd_analyzer()
```

Let's take a closer look at the **deep**doctection analyzer. 

![pipeline](./_imgs/dd_pipeline.png)

The default construction consists of several pipeline components: 

- Layout analysis (object detection)
- Cell analysis in table regions (object detection)
- Row and column analysis in table regions (object detection)
- Table segmentation 
- Table segmentation refinement
- OCR with Tesseract
- Assignment of words to layout segments
- Reading order of words within layout segments and reading order of layout segments.

Therefore in total, three object detectors and one OCR are loaded.

## Configuration

![config](./_imgs/conf_dd_one_yaml.png)

We see while initializing a configuration of the analyzer. The configuration is recorded in a `.yaml` file. You can find this file in the .cache dir of **deep**doctection.

You can use the `.yaml` file to replace one model with e.g. a model trained on your own data. The easiest way is to add your model to the `ModelCatalog` and change the model in the `.yaml` file. Adding a model to the `ModelCatalog` is something that is covered in the section **Running_pre_trained_models_from_third_party_libraries.ipynb**.

## Pipeline components

Having a pipeline, you can list the components with `get_pipeline_info()`.


```python
analyzer.get_pipeline_info()
```




    {0: 'image_weights_layout_d2_model_0829999_layout_inf_only.pt',
     1: 'sub_image_weights_cell_d2_model_1849999_cell_inf_only.pt',
     2: 'sub_image_weights_item_d2_model_1639999_item_inf_only.pt',
     3: 'table_segment',
     4: 'table_segment_refine',
     5: 'text_extract_tesseract',
     6: 'matching',
     7: 'text_order'}




```python
analyzer.get_pipeline_info(position=3)
```




    'table_segment'



If you do not want to process any text extraction you can set `ocr=False` which gives you a shorter pipeline with fewer backbones.


```python
analyzer = dd.get_dd_analyzer(ocr=False)
```


```python
analyzer.get_pipeline_info()
```

    {0: 'image_weights_layout_d2_model_0829999_layout_inf_only.pt',
     1: 'sub_image_weights_cell_d2_model_1849999_cell_inf_only.pt',
     2: 'sub_image_weights_item_d2_model_1639999_item_inf_only.pt',
     3: 'table_segment',
     4: 'table_segment_refine'}

You have access to pipeline components via `pipe_component_list`.


```python
analyzer.pipe_component_list
```

    [<deepdoctection.pipe.layout.ImageLayoutService at 0x7fa9661000a0>,
     <deepdoctection.pipe.cell.SubImageLayoutService at 0x7fa965f708e0>,
     <deepdoctection.pipe.cell.SubImageLayoutService at 0x7fa965ff6df0>,
     <deepdoctection.pipe.segment.TableSegmentationService at 0x7fa965ff6c40>,
     <deepdoctection.pipe.refine.TableSegmentationRefinementService at 0x7fa965ff6a60>]

## Layout detection models

The `ImageLayoutService` is responsible to detect the coarse layout structure over a full image. It has an object
detector, which can be either a Tensorpack or a Detectron2 model.


```python
image_layout_service = analyzer.pipe_component_list[0]
```


```python
image_layout_service.predictor
```

    <deepdoctection.extern.d2detect.D2FrcnnDetector at 0x7faa61388d30>

You can get a list of all categories that a model is able to detect. Moreover, you will find a unique description of each model in your pipeline.


```python
image_layout_service.predictor.possible_categories()
```

    [<LayoutType.text>,
     <LayoutType.title>,
     <LayoutType.list>,
     <LayoutType.table>,
     <LayoutType.figure>]

```python
image_layout_service.predictor.name
```

    'weights_layout_d2_model_0829999_layout_inf_only.pt'

```python
cell_service = analyzer.pipe_component_list[1]
```

```python
cell_service.predictor.possible_categories()
```

    [<LayoutType.cell>]

```python
cell_service.predictor.name
```

    'weights_cell_d2_model_1849999_cell_inf_only.pt'

## OCR, matching and reading order

Let's re-load the analyzer again, now with OCR.


```python
analyzer = dd.get_dd_analyzer()
```

The matching services maps words the layout segments by overlapping.  In order to do so, we need to specify what layout segments we want to consider. 

In this situation we do not consider `figure` as valid layout section and neglect any overlapping of a word with a `figure` segment. Of course, this can be changed by adding `figure` to the list of `parent_categories`.

Orphan words with no overlapping with any layout segment will be set aside. There are customizations that describe how to deal with orphan words. 


```python
match_service = analyzer.pipe_component_list[6]
```


```python
print(f"parent_categories: {match_service.parent_categories}, child_categories: {match_service.child_categories}")
```

    parent_categories: ['text', 'title', 'cell', 'list'], child_categories: word


There is a matching rule and a threshold to specifiy. We also need to choose whether we want to assign a word to 
multiple layout sections. When setting `max_parent_only=True` we consider only the largest overlapping. 


```python
print(f"matching_rule: {match_service.matching_rule} \n match_threshold: {match_service.threshold} \n max_parent_only: {match_service.max_parent_only}")
```

    matching_rule: ioa 
     match_service: 0.6 
     max_parent_only: False


Customizing the reading order requires some additional terminology which goes beyond the introduction. 
We refer to [this page](https://deepdoctection.readthedocs.io/en/latest/tutorials/layout_parsing_structure) .


```python
text_order_service = analyzer.pipe_component_list[7]
```


```python
text_order_service.text_container
```

    <LayoutType.word>

```python
text_order_service.floating_text_block_names
```


    [<LayoutType.title>, <LayoutType.text>, <LayoutType.list>]

```python
text_order_service.text_block_names
```

    [<LayoutType.title>,
     <LayoutType.text>,
     <LayoutType.list>,
     <LayoutType.cell>,
     <CellType.header>,
     <CellType.body>]


