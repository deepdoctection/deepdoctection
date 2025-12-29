<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>


# More on Parsing

In our [first example](Analyzer_Get_Started.md), we parsed a page and looked at the individual components that were 
returned. Now, we will look at another example and demonstrate what other kinds of results can also be displayed.


We begin by instantiating the analyzer, which we configure slightly differently from the default settings.

## First config changes

```python
import deepdoctection as dd
from matplotlib import pyplot as plt

analyzer = dd.get_dd_analyzer(config_overwrite=['USE_LAYOUT_LINK=True']) # (1)

df = analyzer.analyze(path="/path/to/dir/2312.13560.pdf")
df.reset_state()
doc=iter(df)
pages = list(doc) # (2)
page = pages[0]

plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(page.viz())
```

1. We set the `USE_LAYOUT_LINK` parameter to `True`. This enables the analyzer to link captions to figures and tables.
2. Will process the whole document.
    
![png](../_imgs/analyzer_more_on_parsing_01.png)


!!! info "Note"
	
    You may notice that some `line`s are labeled with the category line. This layout section is artificial and generated
    by the `analyzer`. Every word recognized by the OCR must be assigned to a layout section. If this is not possible
    for certain `word`s, they are grouped together and merged into a `line`.


The watermark on the left is noticeable — it is not displayed. These are `residual_layouts` like `page_header` and 
`page_footer`. These special layout sections can be displayed if needed.


```python
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(page.viz(page_header="category_name", 
                    page_footer="category_name")) # (1) 
```

1. Pass the layout section`s category_name as argument. It`s value is the value we want to display, in this case it`s 
   `category_name`. You can also display other attributes, e.g. `annotation_id`.

![png](../_imgs/analyzer_more_on_parsing_02.png)
    

!!! info "Note"
    The **deep**doctection reading order algorithm is rule-based but can handle various layout types, such as 
    multi-column layouts. However, there are also page layouts where determining the correct reading order fails.


```python
print(page.text)
```

??? info "Output"

    KNN-CTC: ENHANCING ASR VIA RETRIEVAL OF CTC PSEUDO LABELS
    Jiaming Zhou', Shiwan Zhao*, Yaqi Liu, Wenjia Zengs, Yong Chen, Yong Qin't
    Nankai University, Tianjin, China 2 Beijing University of Technology, Beijing, China 3 Lingxi (Beijing) Technology 
    Co., Ltd.
    ABSTRACT
    The success of retrieval-augmented language models in var- ious natural language processing (NLP) tasks has been 
    con- strained in automatic speech recognition (ASR) applications due to challenges in constructing fine-grained 
    audio-text datastores. This paper presents KNN-CTC, a novel approach that overcomes these challenges by leveraging 
    Connection- ist Temporal Classification (CTC) pseudo labels to establish frame-level audio-text key-value pairs, 
    circumventing the need for precise ground truth alignments. We further in- troduce a "skip-blank" strategy, which 
    strategically ignores CTC blank frames, to reduce datastore size. By incorpo- rating a k-nearest neighbors retrieval 
    mechanism into pre- trained CTC ASR systems and leveraging a fine-grained, pruned datastore, KNN-CTC consistently 
    achieves substan- tial improvements in performance under various experimental settings. Our code is available at 
    htps/gihuhcomNKU: HLT/KNN-CTC.
    Index Terms- speech recognition, CTC, retrieval- augmented method, datastore construction
    1. INTRODUCTION
    In recent years, retrieval-augmented language models [1,2,3, 4, 5], which refine a pre-trained language model by 
    linearly interpolating the output word distribution with a k-nearest neighbors (KNN) model, have achieved remarkable 
    success across a broad spectrum of NLP tasks, encompassing lan- guage modeling, question answering, and machine 
    transla- tion. Central to the success of KNN language models is the construction of a high-quality key-value datastore.
    Despite these advancements in NLP tasks, applications in speech tasks, particularly in automatic speech recognition 
    (ASR), remain constrained due to the challenges associ- ated with constructing a fine-grained datastore for the 
    audio modality. Early exemplar-based ASR [6, 7] utilized KNN to improve the conventional GMM-HMM or DNN-HMM based 
    approaches. Recently, Yusuf et al. [8] proposed enhancing
    Independent researcher.
    TCorresponding author. This work was supported in part by NSF China (Grant No. 62271270).
    a transducer-based ASR model by incorporating a retrieval mechanism that searches an external text corpus for poten- 
    tial completions of partial ASR hypotheses. However, this method still falls under the KNN language model category, 
    which only enhances the text modality of RNN-T [9]. Chan etal. [10] employed Text To Speech (TTS) to generate audio 
    and used the audio embeddings and semantic text embed- dings as key-value pairs to construct a datastore, and then 
    augmented the Conformer [11] with KNN fusion layers to en- hance contextual ASR. However, this approach is 
    restricted to contextual ASR, and the key-value pairs are coarse-grained, with both key and value at the phrase 
    level. Consequently, the challenge of constructing a fine-grained key-value datastore remains a substantial obstacle 
    in the ASR domain.
    In addition to ASR, there have been a few works in other speech-related tasks. For instance, RAMP [12] incorporated 
    KNN into mean opinion score (MOS) prediction by merg- ing the parametric model and the KNN-based non-parametric model. 
    Similarly, Wang et al. [13] proposed a speech emotion recognition (SER) framework that utilizes contrastive learn- 
    ing to separate different classes and employs KNN during in- ference to harness improved distances. However, both ap- 
    proaches still rely on the utterance-level audio embeddings as the key, with ground truth labels as the value.
    The construction of a fine-grained datastore for the audio modality in ASR tasks is challenged by two significant 
    obsta- cles: (i) the absence of precise alignment knowledge between audio and transcript characters, which poses a 
    substantial dif- ficulty in acquiring the ground-truth labels (i.e., the values) necessary for the creation of 
    key-value pairs, and (i) the im- mense volume of entries generated when processing audio at the frame level. In this 
    study, we present KNN-CTC, a novel approach that overcomes these challenges by utilizing CTC (Connectionist 
    Temporal Classification) [14] pseudo labels. This innovative method results in significant improvements in ASR task 
    performance. By utilizing CTC pseudo labels, W6 are able to establish frame-level key-value pairs, eliminating the 
    need for precise ground-truth alignments. Additionally we introduce a 'skip-blank' strategy that exploits the inher 
    ent characteristics of CTC to strategically omit blank frames thereby reducing the size of the datastore. KNN-CTC 
    attains comparable performance on the pruned datastore, and ever



```python
figure = page.figures[0]
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(figure.image.viz()) # (1) 
```

1. `figure.image.viz()` returns a NumPy array containing the image segment enclosed by the bounding box.
 

![png](../_imgs/analyzer_more_on_parsing_04.png)
    
We can save the figure as a single `.png`. 

```python
dd.viz_handler.write_image(f"/path/to/dir/{figure.annotation_id}.png", figure.image.image)
```

## Layout Linking

By setting `USE_LAYOUT_LINK=True`, we enabled a component that links `caption`s to `table`s or `figure`s. The linking 
is rule-based: if a `table` or `figure` is present, a `caption` is associated with the nearest one in terms of spatial 
proximity.


```python
caption = figure.layout_link[0]
print(f"annotation_id: {caption.annotation_id}, text: {caption.text}")
```

??? info "Output"

    text: Fig. 1. Overview of our KNN-CTC framework, which com- bines CTC and KNN models. The KNN model consists of two
    stages: datastore construction (blue dashed lines) and candi- date retrieval (orange lines)., annotation_id: 
    46bd4e42-8d50-30fb-883a-6c4d82b236af


We conclude  with some special features. Suppose you have a specific layout segment. Using get_layout_context, we can 
retrieve the surrounding layout segments within a given context_size, i.e., the `k` layout segments that appear before
and after it in the reading order.


```python
for layout in page.get_layout_context(annotation_id="13a5f0ea-19e5-3317-b50c-e4c829a73d09", context_size=1):
    print(f"annotation_id: {layout.annotation_id}, text: {layout.text}")
```

??? info "Output"

    annotation_id: 40d63bea-9815-3e97-906f-76b501c67667, text: (2)
    annotation_id: 13a5f0ea-19e5-3317-b50c-e4c829a73d09, text: Candidate retrieval: During the decoding phase, our 
    process commences by generating the intermediate represen-
    annotation_id: 13cb9477-f605-324a-b497-9f42335c747d, text: tation f(Xi) alongside the CTC output PCTC(YIx). Pro- 
    ceeding further, we leverage the intermediate representations f(Xi) as queries, facilitating the retrieval of the 
    k-nearest neighbors N. We then compute a softmax probability distri- bution over the neighbors, aggregating the 
    probability mass for each vocabulary item by:

## Analyzer metadata 

What does the analyzer predict? 

We can use the meta annotations to find out which attributes are determined for which object types. The attribute 
`image_annotations` represent all layout segments constructed by the analyzer. Ultimately, `ImageAnnotation`s are 
everything that can be enclosed by a bounding box. 


```python
meta_annotations = analyzer.get_meta_annotation()
meta_annotations.image_annotations
```

??? info "Output"
    
    <pre>
     (DefaultType.DEFAULT_TYPE,
     LayoutType.CAPTION,
     LayoutType.TEXT,
     LayoutType.TITLE,
     LayoutType.FOOTNOTE,
     LayoutType.FORMULA,
     LayoutType.LIST_ITEM,
     LayoutType.PAGE_FOOTER,
     LayoutType.PAGE_HEADER,
     LayoutType.FIGURE,
     LayoutType.SECTION_HEADER,
     LayoutType.TABLE,
     LayoutType.COLUMN,
     LayoutType.ROW,
     CellType.COLUMN_HEADER,
     CellType.PROJECTED_ROW_HEADER,
     CellType.SPANNING,
     LayoutType.WORD,
     LayoutType.LINE)
     </pre>


The `sub_categories` represent attributes associated with specific `ImageAnnotations`. For a table cell, for example,
these include: `<CellType.COLUMN_NUMBER>, <CellType.COLUMN_SPAN>, <CellType.ROW_NUMBER> and <CellType.ROW_SPAN>`. 


```python
meta_annotations.sub_categories
```


??? info "Output"

    <pre>
    {LayoutType.CELL: {CellType.COLUMN_NUMBER,
                      CellType.COLUMN_SPAN,
                      CellType.ROW_NUMBER,
                      CellType.ROW_SPAN}, 
    CellType.SPANNING: {CellType.COLUMN_NUMBER,
                      CellType.COLUMN_SPAN,
                      CellType.ROW_NUMBER,
                      CellType.ROW_SPAN},
    CellType.ROW_HEADER: {CellType.COLUMN_NUMBER,
                      CellType.COLUMN_SPAN,
                      CellType.ROW_NUMBER,
                      CellType.ROW_SPAN},
    CellType.COLUMN_HEADER: {CellType.COLUMN_NUMBER,
                      CellType.COLUMN_SPAN,
                      CellType.ROW_NUMBER,
                      CellType.ROW_SPAN},
    CellType.PROJECTED_ROW_HEADER: {CellType.COLUMN_NUMBER,
                      CellType.COLUMN_SPAN,
                      CellType.ROW_NUMBER,
                      CellType.ROW_SPAN},
    LayoutType.ROW: {CellType.ROW_NUMBER},
    LayoutType.COLUMN: {CellType.COLUMN_NUMBER},
    LayoutType.WORD: {WordType.CHARACTERS, Relationships.READING_ORDER},
    LayoutType.TEXT: {Relationships.READING_ORDER},
    LayoutType.TITLE: {Relationships.READING_ORDER},
    LayoutType.LIST: {Relationships.READING_ORDER},
    LayoutType.KEY_VALUE_AREA: {Relationships.READING_ORDER},
    LayoutType.LINE: {Relationships.READING_ORDER}}
    </pre>


The relationships represent one or more relations between different `ImageAnnotation`s. 


```python
meta_annotations.relationships
```


??? info "Output"

    <pre>
    {LayoutType.TABLE: {Relationships.CHILD, 
                        Relationships.LAYOUT_LINK},
    LayoutType.TABLE_ROTATED: {Relationships.CHILD},
    LayoutType.TEXT: {Relationships.CHILD},
    LayoutType.TITLE: {Relationships.CHILD},
    LayoutType.LIST_ITEM: {Relationships.CHILD},
    LayoutType.LIST: {Relationships.CHILD},
    LayoutType.CAPTION: {Relationships.CHILD},
    LayoutType.PAGE_HEADER: {Relationships.CHILD},
    LayoutType.PAGE_FOOTER: {Relationships.CHILD},
    LayoutType.PAGE_NUMBER: {Relationships.CHILD},
    LayoutType.MARK: {Relationships.CHILD},
    LayoutType.KEY_VALUE_AREA: {Relationships.CHILD},
    LayoutType.FIGURE: {Relationships.CHILD, 
                        Relationships.LAYOUT_LINK},
    CellType.SPANNING: {Relationships.CHILD},
    LayoutType.CELL: {Relationships.CHILD}}
    </pre>


The summaries describe facts presented at the page level — for instance, a `document_type`. This pipeline does not have
a document type classifier.


```python
meta_annotations.summaries
```

??? info "Output"

    ()


By the way, don’t be confused by the obscure way the different categories are displayed. The categories are specific 
enum members. Each enum member can be converted into a string type, and vice versa — a string type can be converted 
back into an enum member:


```python
dd.LayoutType.CELL, dd.LayoutType.CELL.value, dd.get_type('cell')
```

??? info "Output"

    (LayoutType.CELL, 'cell', LayoutType.CELL)


