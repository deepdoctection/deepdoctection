# Various topics around LayoutLM part 1


These records form a disjointed collection of topics that may be
helpful for fine-tuning LayoutLM. In addition,
the code provided should serve as an aid on how to deal with other
topics with the **deep**doctection.

## How does LayoutLM work on newer date documents?

LayoutLM was pre-trained on the CDIP (Complex Document Information
Processing) dataset, a corpus of various types of documents disclosed by
the tobacco industry. Most documents date from the 1990s.

We investigate to what extent document classification works for
documents of recent date. For this we use Doclaynet, a more recent
dataset released for Document Layout Analysis, but which can also be
used for classification problems.

The corpus consists of single pages classified as financial reports,
laws and regulations, scientific publications, government tenders,
manuals and patents and is delivered in scaled image files as well as
native PDF documents. As in the previous sequence classification
tutorial, we first extract the text to create a training and evaluation
set.

We use PDF Plumber to extract the text from the native PDF documents.

## Generating raw corpus from PDF documents

```python

    import deepdoctection as dd
    import os
    from matplotlib import pyplot as plt
    from transformers import LayoutLMTokenizerFast
```

```python

    def get_pdf_miner():
        pdf_miner = dd.PdfPlumberTextDetector()
        text_service = dd.TextExtractionService(pdf_miner)
        return dd.DoctectionPipe(pipeline_component_list=[text_service])
    
    def map_to_pdf_file(dp):
        pdf_base_path = Path("/path/to/dir/DocLayNet_extra/PDF")
    
        dp.file_name = dp.file_name.replace(".png",".pdf")
        dp.location = pdf_base_path / dp.file_name
        pdf_bytes =  dd.load_bytes_from_pdf_file(dp.location)
        dp._bbox = None
        dp.embeddings.pop(dp.image_id)
        dp.image = dd.convert_pdf_bytes_to_np_array_v2(pdf_bytes, dpi=100)
        dp.pdf_bytes = pdf_bytes
        return dp
```

```python

    # This step requires Doclaynet to be downloaded. Similar to the LayoutLM primer we select four categories financial_report,
    # government_tenders, manuals, laws_and_regulations and generate a raw dataset corpus with extrated text and bounding boxes.
    
    doclaynet_seq = dd.get_dataset("doclaynet-seq")
    doclaynet_seq.dataflow.categories.filter_categories({"financial_reports"})
    pdf_miner_pipe = get_pdf_miner()
    
    df = doclaynet_seq.dataflow.build()
    
    df = dd.MapData(df, map_to_pdf_file) # some manipulation necessary like passing the pdf document in bytes so that we 
                                         # can run the text extraction with pdfplumber. We also create np array with dpi=100 
                                         # instead of default dpi = 300.
    
    df = pdf_miner_pipe.analyze(dataset_dataflow= df,output="image")
    dd.dataflow_to_json(df, "/path/to/dir/doclaynet_img", single_files=True, max_datapoints=1000, save_image=True,
                        save_image_in_json=False, highest_hierarchy_only=True)
```

## Defining deepdoctection dataset


```python

    @dd.dataset_registry.register("doclaynet-seq")
    class Doclaynet(dd.DatasetBase):
    
        @classmethod
        def _info(cls) -> dd.DatasetInfo:
            return dd.DatasetInfo(name="doclaynet-seq", description="", license="", url="", splits={}, type="SEQUENCE_CLASSIFICATION")
    
        def _categories(self) -> dd.DatasetCategories:
            return dd.DatasetCategories(init_categories=["financial_report",
                                                         "government_tenders",
                                                         "manuals",
                                                         "laws_and_regulations"])
    
        def _builder(self) -> "DocBuilder":
            return DocBuilder(location="doclaynet_img")
    
    
    class DocBuilder(dd.DataFlowBaseBuilder):
    
        def build(self, **kwargs) -> dd.DataFlow:
            load_image = kwargs.get("load_image", False)
    
            ann_files_dir = self.get_workdir()
            image_dir = self.get_workdir() / "image"
    
            df = dd.SerializerFiles.load(ann_files_dir,".json")
            df = dd.MapData(df, dd.load_json)
            categories = self.categories.get_categories(name_as_key=True)
    
            @dd.curry
            def map_to_img(dp, cats):
                dp = dd.Image.from_dict(**dp) # no heavy conversion necessary.
                dp.file_name= dp.file_name.replace(".pdf",".png")
                dp.location = image_dir / dp.file_name
                if not os.path.isfile(dp.location): # when creating the dataset some image could not be generated and we have to skip these
                    return None
                if not len(dp.annotations): # Some samples were rotated where OCR was not able to recognize text. No text -> no features
                    return None
                for ann in dp.get_annotation():
                    try:
                        ann.get_sub_category("characters") # Sometime pdfplumber hangs and generates boxes without text. Will the filter the
                                                               # complete sample
                    except KeyError:
                        return None
                sub_cat = dp.summary.get_sub_category("document_type")
                sub_cat.category_id = cats[sub_cat.category_name]
                return dp
            df = dd.MapData(df, map_to_img(categories))
    
            def _maybe_load_image(dp):
                if load_image:
                    dp.image = dd.load_image_from_file(dp.location)
                return dp
    
            df = dd.MapData(df, _maybe_load_image)
    
            return df
```

```python

    doclaynet = dd.get_dataset("doclaynet-seq")
    
    df = doclaynet.dataflow.build(load_image=True)
    df.reset_state()
    df_iter = iter(df)
```


## Displaying some samples

```python

    dp = next(df_iter)
    page = dd.Page.from_image(dp,text_container="word")
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(page.viz())
```

![](./_imgs/layoutlm_various_topics_1.png)


```python

    page.document_type
```



```

    'laws_and_regulations'
```


```python

    dp = next(df_iter)
    page = dd.Page.from_image(dp,text_container="word")
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(page.viz())
```

![](./_imgs/layoutlm_various_topics_2.png)


```python

    page.document_type
```



```

    'manuals'
```


```python

    dp = next(df_iter)
    page = dd.Page.from_image(dp,text_container="WORD")
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(page.viz())
```

![](./_imgs/layoutlm_various_topics_3.png)


```python

    page.document_type
```


```
    'financial_report'
```


## Fine tuning


```python

    path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")
    path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlm-base-uncased/pytorch_model.bin")
```

```python

    doclaynet = dd.get_dataset("doclaynet-seq")
    
    merge = dd.MergeDataset(doclaynet)
    merge.buffer_datasets()
    merge.split_datasets(ratio=0.1)
```

```

    [0906 15:16.46 @base.py:270] ___________________ Number of datapoints per split ___________________
    [0906 15:16.46 @base.py:271] {'test': 161, 'train': 2989, 'val': 161}
```

```python

    dataset_train = merge
    dataset_val = merge
    
    metric = dd.get_metric("confusion")
    metric.set_categories(summary_sub_category_names="document_type")
    
    dd.train_hf_layoutlm(path_config_json,
                         dataset_train,
                         path_weights,
                         log_dir="/path/to/dir/Seq_Doclaynet",
                         dataset_val= dataset_val,
                         metric=metric,
                         pipeline_component_name="LMSequenceClassifierService")
```

```

    [0906 15:16.46 @maputils.py:205]Ground-Truth category distribution:
    |       category       | #box   |      category      | #box   |  category  | #box   |
    |:--------------------:|:-------|:------------------:|:-------|:----------:|:-------|
    |  FINANCIAL_REPORTS   | 877    | GOVERNMENT_TENDERS | 301    |  MANUALS   | 901    |
    | LAWS_AND_REGULATIONS | 910    |                    |        |            |        |
    |        total         | 2989   |                    |        |            |        |
    [0906 15:16.46 @custom.py:133]Make sure to call .reset_state() for the dataflow
    otherwise an error will be raised

    
    Saving model checkpoint to /path/to/dir/Seq_Doclaynet/checkpoint-2000
    Configuration saved in /path/to/dir/Seq_Doclaynet/checkpoint-2000/config.json
    Model weights saved in /path/to/dir/Seq_Doclaynet/checkpoint-2000/pytorch_model.bin

    [0906 15:27.46 @eval.py:157]Starting evaluation...
    [0906 15:27.46 @accmetric.py:404]Confusion matrix: 
    |    predictions ->  |   1 |   2 |   3 |   4 |
    |     ground truth | |     |     |     |     |
    |                  v |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|
    |                  1 |  63 |   0 |   0 |   0 |
    |                  2 |   0 |  15 |   0 |   0 |
    |                  3 |   0 |   0 |  43 |   0 |
    |                  4 |   0 |   0 |   1 |  39 |
```

```python

    path_config_json = "/path/to/dir/Seq_Doclaynet/checkpoint-2000/config.json"
    path_weights = "/path/to/dir/Seq_Doclaynet/checkpoint-2000/pytorch_model.bin"
    
    layoutlm_classifier = dd.HFLayoutLmSequenceClassifier("layoutlmv1", path_config_json,
                                                          path_weights,
                                                          merge.dataflow.categories.get_categories(as_dict=True))
    
    tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    
    pipe_component = dd.LMSequenceClassifierService(tokenizer_fast,layoutlm_classifier,dd.image_to_layoutlm_features)
    
    evaluator = dd.Evaluator(merge,pipe_component,metric)
    evaluator.run(split="test")

    
    loading weights file /path/to/dir/Seq_Doclaynet/checkpoint-2000/pytorch_model.bin
    All model checkpoint weights were used when initializing LayoutLMForSequenceClassification.
    
    All the weights of LayoutLMForSequenceClassification were initialized from the model checkpoint at /home/janis/Tests/Seq_Doclaynet/checkpoint-2000/pytorch_model.bin.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use LayoutLMForSequenceClassification for predictions without further training.
```

```

    [0906 15:29.45 @eval.py:157]Starting evaluation...
    [0906 15:29.45 @accmetric.py:404]Confusion matrix: 
    |    predictions ->  |   1 |   2 |   3 |   4 |
    |     ground truth | |     |     |     |     |
    |                  v |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|
    |                  1 |  45 |   0 |   0 |   0 |
    |                  2 |   0 |  17 |   0 |   0 |
    |                  3 |   0 |   0 |  56 |   0 |
    |                  4 |   0 |   0 |   0 |  43 |
```

## Conclusion


LayoutLM delivers excellent results when classifying documents from
other domains.

## Follow up:

Can you reduce the training set?

It may be due to the fact that the documents in this dataset are well
distinguished. However, with the almost perfect prediction, the question
arises as to whether significantly less training data can be used. This
question is important because labeling always involves a certain amount
of effort.

We choose a distribution where training data is only about 10% of the
total data set.

```python

    doclaynet = dd.get_dataset("doclaynet-seq")
    
    merge = dd.MergeDataset(doclaynet)
    merge.buffer_datasets()
    merge.split_datasets(ratio=0.9)
```


    [0906 15:43.46 @base.py:270][0m ___________________ Number of datapoints per split ___________________
    [0906 15:43.46 @base.py:271][0m {'test': 1494, 'train': 322, 'val': 1495}


```python

    dataset_train = merge
    dataset_val = merge
    
    metric = dd.get_metric("confusion")
    metric.set_categories(summary_sub_category_names="document_type")
    
    path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")
    path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlm-base-uncased/pytorch_model.bin")
    
    dd.train_hf_layoutlm(path_config_json,
                         dataset_train,
                         path_weights,
                         log_dir="/path/to/dir/Seq_Doclaynet",
                         dataset_val= dataset_val,
                         metric=metric,
                         pipeline_component_name="LMSequenceClassifierService")
```

```

    [0906 15:43.59 @maputils.py:205]Ground-Truth category distribution:
    |       category       | #box   |      category      | #box   |  category  | #box   |
    |:--------------------:|:-------|:------------------:|:-------|:----------:|:-------|
    |  FINANCIAL_REPORTS   | 108    | GOVERNMENT_TENDERS | 32     |  MANUALS   | 99     |
    | LAWS_AND_REGULATIONS | 83     |                    |        |            |        |
    |        total         | 322    |                    |        |            |        |
    [0906 15:43.59 @custom.py:133][0m Make sure to call .reset_state() for the dataflow otherwise an error will be raised

    [0906 15:45.13 @accmetric.py:404]Confusion matrix: 
    |    predictions ->  |   1 |   2 |   3 |   4 |
    |     ground truth | |     |     |     |     |
    |                  v |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|
    |                  1 | 436 |   0 |   3 |  11 |
    |                  2 |   4 | 139 |   0 |   4 |
    |                  3 |   0 |   0 | 436 |   4 |
    |                  4 |   2 |   0 |   0 | 456 |
```


## Conclusion:


We stop the training after 100 iterations because the first evaluation
with Confusion Matrix already shows that the results are excellent. We want to emphasize that we have not looked at
examples, therefore we cannot rule out that there might be a trivial reason why the score is that high.
