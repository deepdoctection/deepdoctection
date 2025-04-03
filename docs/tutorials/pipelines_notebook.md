# Pipelines

Let's now take a closer look at pipelines. We will be using the **deep**doctection analyzer as an example. 

A pipeline is built as a sequence of tasks. These tasks are called pipeline components, pipeline backbones or services.

![pipelines](./_imgs/dd_overview_pipeline.png)

Once a pipeline is defined, images or documents can be processed. These are either pure image files (like JPG, PNG, TIFF) or PDF files. PDF files are read and processed page by page. Each PDF page is converted into a numpy array under the hood. 

We do not want to go into detail about the data structure at this point. If you want more information, please refer to the [data structure notebook](data_structure_notebook.md).


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

We see while initializing a configuration in the logs of the analyzer. The configuration is saved in a `.yaml` file. You can find this file in the .cache dir of **deep**doctection.

You can use the `.yaml` file to replace one model with e.g. a model trained on your own data. The tutorial [**Analyzer_Configuration.ipynb**](./analyzer_configuration_notebook.md) will show you where you need to pay attention when changing the `.yaml` file.

In [this tutorial](./running_pre_trained_models_from_third_party_libraries_notebook.md) we will show you how to add a model to the `ModelCatalog` and change the model in the `.yaml` file so that you can use model from third party libraries, that run layout detection models with Detectron2.

## Pipeline components

Having a pipeline, you can list the components with `get_pipeline_info()`.


```python
analyzer.get_pipeline_info()
```




    {'c5a80ae0': 'image_detectron2_GeneralizedRCNNlayout_d2_model_0829999_layout_inf_only.pt',
     '4e905a12': 'sub_image_detectron2_GeneralizedRCNNitem_d2_model_1639999_item_inf_only.pt',
     '1c13beff': 'sub_image_detectron2_GeneralizedRCNNcell_d2_model_1849999_cell_inf_only.pt',
     'dbf4f87c': 'table_segment',
     '19c9a57c': 'table_segment_refine',
     'a3192d15': 'text_extract_Tesseract_4.1.1',
     'd6219eba': 'matching',
     'f10aa678': 'text_order'}




```python
analyzer.get_pipeline_info(service_id="c5a80ae0")
```




    'image_detectron2_GeneralizedRCNNlayout_d2_model_0829999_layout_inf_only.pt'



If you do not want to process any text extraction you can set `config_overwrite=["USE_OCR=False"]` which gives you a shorter pipeline with fewer backbones.


```python
analyzer = dd.get_dd_analyzer(config_overwrite=["USE_OCR=False"])
```

The `config_overwrite` option allows to overwrite every argument specified in the `.yaml` file. E.g. you can overwrite `SEGMENTATION.ASSIGNMENT_RULE` simply by `config_overwrite=["SEGMENTATION.ASSIGNMENT_RULE=True"]`.


```python
analyzer.get_pipeline_info()
```




    {'c5a80ae0': 'image_detectron2_GeneralizedRCNNlayout_d2_model_0829999_layout_inf_only.pt',
     '4e905a12': 'sub_image_detectron2_GeneralizedRCNNitem_d2_model_1639999_item_inf_only.pt',
     '1c13beff': 'sub_image_detectron2_GeneralizedRCNNcell_d2_model_1849999_cell_inf_only.pt',
     'dbf4f87c': 'table_segment',
     '19c9a57c': 'table_segment_refine'}



You have access to pipeline components via `pipe_component_list`.


```python
analyzer.pipe_component_list
```




    [<deepdoctection.pipe.layout.ImageLayoutService at 0x7fb28034cdf0>,
     <deepdoctection.pipe.sub_layout.SubImageLayoutService at 0x7fb28006df10>,
     <deepdoctection.pipe.sub_layout.SubImageLayoutService at 0x7fb27df1ffd0>,
     <deepdoctection.pipe.segment.TableSegmentationService at 0x7fb27df1ffa0>,
     <deepdoctection.pipe.refine.TableSegmentationRefinementService at 0x7fb27df1fe80>]



## Layout detection models

The `ImageLayoutService` is responsible to detect the coarse layout structure over a full image. It has an object
detector, which can be either a Tensorpack or a Detectron2 model.


```python
image_layout_service = analyzer.pipe_component_list[0]
```


```python
image_layout_service.predictor
```




    <deepdoctection.extern.d2detect.D2FrcnnDetector at 0x7fb28027a490>



You can get a list of all categories that a model is able to detect. Moreover, you will find a unique description of each model in your pipeline.

```python
image_layout_service.predictor.get_category_names()
```




    (<LayoutType.TEXT>,
     <LayoutType.TITLE>,
     <LayoutType.LIST>,
     <LayoutType.TABLE>,
     <LayoutType.FIGURE>)




```python
image_layout_service.predictor.name
```




    'detectron2_GeneralizedRCNNlayout_d2_model_0829999_layout_inf_only.pt'




```python
cell_service = analyzer.pipe_component_list[1]
```

```python
cell_service.predictor.get_category_names()
```




    (<LayoutType.ROW>, <LayoutType.COLUMN>)




```python
cell_service.predictor.name
```




    'detectron2_GeneralizedRCNNitem_d2_model_1639999_item_inf_only.pt'



## OCR and matching words to layout segments

Let's re-load the analyzer again, now with OCR.


```python
analyzer = dd.get_dd_analyzer()
```

The matching services maps words to layout segments by overlapping rules.  In order to do so, we need to specify what layout segments we want to consider.

```yaml
WORD_MATCHING:
  PARENTAL_CATEGORIES:
    - text
    - title
    - list
    - cell
    - column_header
    - projected_row_header
    - spanning
    - row_header
  RULE:  ioa
  THRESHOLD:  0.6
```

In this situation we do not consider `figure` as valid layout section and neglect any overlapping of a word with a `figure` segment. Of course, this can be changed by adding `figure` to the list of `parent_categories` or in `WORD_MATCHING.PARENTAL_CATEGORIES` in the `.yaml` file.

What is going to happen with so called orphan words, e.g. words with no overlapping with any layout segment? They simply have no anchor and will be ignored unless we force to process them as well. We will come to this point later. 


```python
match_service = analyzer.pipe_component_list[6]
```


```python
print(f"parent_categories: {match_service.parent_categories}, child_categories: {match_service.child_categories}")
```

    parent_categories: (<LayoutType.TEXT>, <LayoutType.TITLE>, <LayoutType.LIST>, <LayoutType.CELL>, <CellType.COLUMN_HEADER>, <CellType.PROJECTED_ROW_HEADER>, <CellType.SPANNING>, <CellType.ROW_HEADER>), child_categories: (<LayoutType.WORD>,)


There is a matching rule and a threshold to specifiy. We also need to choose whether we want to assign a word to 
multiple layout sections. When setting `max_parent_only=True` we assign the word to the layout section with the largest overlapping. Otherwise note, that the word might be considered twice. Changing `max_parent_only` from the `.yaml` is not provided.


```python
print(f"matching_rule: {match_service.matching_rule} \n match_threshold: {match_service.threshold} \n max_parent_only: {match_service.max_parent_only}")
```

    matching_rule: ioa 
     match_threshold: 0.6 
     max_parent_only: True


## Reading order

In the last step, words and layout segments must be arranged to create continuous text. This all takes place in 
the component `TextOrderService`.

Words that are assigned to layout segments are grouped into lines. Lines are read from top to bottom. 
Auxiliary columns are formed to sort the layout segments. These auxiliary columns are then grouped into contiguous blocks that span vertically across the page. Then the blocks are arranged so that adjacent columns in the contiguous blocks are read from left to right, and the contiguous blocks are read from top to bottom. 

![pipelines](./_imgs/dd_connected_blocks.png)


This order is, of course, completely arbitrary and will not result in the expected reading order for many layout compositions. 

An additional difficulty may be that the layout detection is not sufficiently precise and the algorithm returns a questionable reading order. This should always be kept in mind!

`TextOrderService` has four important parameters: `text_container`, `text_block_categories`, `floating_text_block_categories` and `include_residual_text_container`. 

`text_container` must contain the category that contains characters, e.g. `word`. 

`text_block_categories` contains all layout segments to which words have been added and which must be ordered.

`floating_text_block_categories` contains the text blocks to be included in the floating text. For example, it can be discussed whether tables should be included in the body text. In this configuration they are not included in the text. 

Let's get back to the orphan words: If we set `include_residual_text_container = False`, these words will not receive a `reading_order` and will be ignored in text output.

If, on the other hand, we set `include_residual_text_container = True`, they will be grouped and combined into lines and included to the text output. Thus no words are lost. This is an important configuration and you'll likely need to change it.

We refer to [this page](layout_parsing_structure.md) for more detailed information about layout parsing and text ordering.

Let's have a look how the text order configs are reflected in the `.yaml`.

```yaml
TEXT_ORDERING:
  TEXT_BLOCK_CATEGORIES:
    - title
    - text
    - list
    - cell
    - column_header
    - projected_row_header
    - spanning
    - row_header
  FLOATING_TEXT_BLOCK_CATEGORIES:
    - title
    - text
    - list
  INCLUDE_RESIDUAL_TEXT_CONTAINER: False
  STARTING_POINT_TOLERANCE: 0.005
  BROKEN_LINE_TOLERANCE: 0.003
  HEIGHT_TOLERANCE: 2.0
  PARAGRAPH_BREAK: 0.035
```


```python
text_order_service = analyzer.pipe_component_list[7]
```


```python
print(f"text_container: {text_order_service.text_container} \n floating_text_block_categories: {text_order_service.floating_text_block_categories} \n text_block_categories: {text_order_service.text_block_categories} \n include_residual_text_container: {text_order_service.include_residual_text_container}")
```

    text_container: word 
     floating_text_block_categories: (<LayoutType.TITLE>, <LayoutType.TEXT>, <LayoutType.LIST>) 
     text_block_categories: (<LayoutType.TITLE>, <LayoutType.TEXT>, <LayoutType.LIST>, <LayoutType.CELL>, <CellType.COLUMN_HEADER>, <CellType.PROJECTED_ROW_HEADER>, <CellType.SPANNING>, <CellType.ROW_HEADER>) 
     include_residual_text_container: False


## Output structure

There is a last step in a pipeline that prepares all information gathered from the different into a consumable class, the `Page` class. The `PageParsingService` is optional and should only be processed if you want to analyze the output.

For a deeper understanding of the connection between `Page` and `Image`, we refer to the [data structure notebook](data_structure_notebook.md).


```python
df = analyzer.analyze(path="path/to/doc.pdf", output="image") # output = "image" will skip PageParsingService. But default value is "page"
```

Note, that the `PageParsingService` is saved in a separate attribute and not part of `analyzer.pipe_component_list`. The `PageParsingService` shares some common parameters with the `TextOrderService`
and it is recommended to use the same configurations. 


```python
page_parser = analyzer.page_parser
```


```python
print(f"text_container: {page_parser.text_container} \n floating_text_block_categories: {page_parser.floating_text_block_categories} \n include_residual_text_container: {page_parser.include_residual_text_container}")
```

    text_container: word 
     floating_text_block_categories: (<LayoutType.TITLE>, <LayoutType.TEXT>, <LayoutType.LIST>) 
     include_residual_text_container: False



```python

```
