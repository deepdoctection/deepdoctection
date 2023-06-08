# LayoutLMv3 for financial report NER

We now cover the latest model in the LayoutLM family. 

An essential difference to other models is that bounding box coordinates do not have to be passed per word not on word level but on segment level. Using this grouping procedure (because segments are coarser than words), one expects that for entities consisting of multiple tokens, predictions will be pushed towards giving equal labels to words from equal segments. As our labels `fund_name` or `umbrella` consist of many tokens, it is interesting to explore whether this leads to further improvement.

Where do we get the segment information from? One possibility is to use a textline detector and use the results for segments. 

FRFPE was labeled so that we used a layout detector fine-tuned on fund documents. The segment results are available as `ImageAnnotation` in ground truth. With that, relations to the segments and words were created using the `MatchingService`. 

During training (as well as in the evaluation or pipelines) it is possible to use the segments that one wants to use as replacement for the `Word` bounding boxes. 

We will now use these procedures to fine-tune LayoutLMv3 correctly. 


```python
import deepdoctection as dd
from collections import defaultdict
import wandb
from transformers import RobertaTokenizerFast
```

    /home/janis/Public/deepdoctection_pt/venv/lib/python3.8/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
@dd.object_types_registry.register("ner_first_page")
class FundsFirstPage(dd.ObjectTypes):

    report_date = "report_date"
    umbrella = "umbrella"
    report_type = "report_type"
    fund_name = "fund_name"

dd.update_all_types_dict()

@dd.curry
def overwrite_location_and_load(dp, image_dir, load_image):
    image_file = image_dir / dp.file_name.replace("pdf","png")
    dp.location = image_file.as_posix()
    if load_image:
        dp.image = dd.load_image_from_file(image_file)
    return dp

class NerBuilder(dd.DataFlowBaseBuilder):

    def build(self, **kwargs) -> dd.DataFlow:
        load_image = kwargs.get("load_image", False)
        filter_languages = kwargs.get("filter_languages")

        ann_files_dir = self.get_workdir()
        image_dir = self.get_workdir() / "image"

        df = dd.SerializerFiles.load(ann_files_dir,".json")   # get a stream of .json files
        df = dd.MapData(df, dd.Image.from_file)   # load .json file

        df = dd.MapData(df, overwrite_location_and_load(image_dir, load_image))

        if self.categories.is_filtered():
            df = dd.MapData(
                df,
                dd.filter_cat(
                    self.categories.get_categories(as_dict=False, filtered=True),
                    self.categories.get_categories(as_dict=False, filtered=False),
                ),
            )
        df = dd.MapData(df,dd.re_assign_cat_ids(cat_to_sub_cat_mapping=self.categories.get_sub_categories(
                                                 categories=dd.LayoutType.word,
                                                 sub_categories={dd.LayoutType.word: dd.WordType.token_class},
                                                 keys = False,
                                                 values_as_dict=True,
                                                 name_as_key=True)))
        
        if filter_languages:
            df = dd.MapData(df, dd.filter_summary({"language": [dd.get_type(lang) for lang in filter_languages]},
                                                 mode="value"))

        return df
    
ner = dd.CustomDataset(name = "FRFPE",
                 dataset_type=dd.DatasetType.token_classification,
                 location="FRFPE",
                 init_categories=[dd.LayoutType.text, dd.LayoutType.title, dd.LayoutType.list, dd.LayoutType.table,
                                  dd.LayoutType.figure, dd.LayoutType.line, dd.LayoutType.word],
                 init_sub_categories={dd.LayoutType.word: {dd.WordType.token_class: [FundsFirstPage.report_date,
                                                                                     FundsFirstPage.report_type,
                                                                                     FundsFirstPage.umbrella,
                                                                                     FundsFirstPage.fund_name,
                                                                                     dd.TokenClasses.other],
                                                           dd.WordType.tag: []}},
                 dataflow_builder=NerBuilder)
```

```python
df = ner.dataflow.build(load_image=True)

merge = dd.MergeDataset(ner)
merge.explicit_dataflows(df)
merge.buffer_datasets()
```


```python
wandb.init(project="FRFPE_layoutlmv1", resume=True)
artifact = wandb.use_artifact('jm76/FRFPE_layoutlmv1/merge_FRFPE:v0', type='dataset')
table = artifact.get("split")
```

wandb version 0.15.4 is available!  To upgrade, please run:
 $ pip install wandb --upgrade



Syncing run <strong><a href='https://wandb.ai/jm76/FRFPE_layoutlmv1/runs/1kca6a9r' target="_blank">avid-plant-11</a></strong> to <a href='https://wandb.ai/jm76/FRFPE_layoutlmv1' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/run' target="_blank">docs</a>)<br/>


View project at <a href='https://wandb.ai/jm76/FRFPE_layoutlmv1' target="_blank">https://wandb.ai/jm76/FRFPE_layoutlmv1</a>


View run at <a href='https://wandb.ai/jm76/FRFPE_layoutlmv1/runs/1kca6a9r' target="_blank">https://wandb.ai/jm76/FRFPE_layoutlmv1/runs/1kca6a9r</a>



```python
split_dict = defaultdict(list)
for row in table.data:
    split_dict[row[0]].append(row[1])

merge.create_split_by_id(split_dict)
```


```python
wandb.finish()
```


So not forget to download the model if it is not in you .cache yet.


```python
path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlmv3-base/pytorch_model.bin")
path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlmv3-base/pytorch_model.bin")

metric = dd.get_metric("f1")
metric.set_categories(sub_category_names={"word": ["token_class"]})

dd.train_hf_layoutlm(path_config_json,
                     merge,
                     path_weights,
                     config_overwrite=["max_steps=2000",
                                       "per_device_train_batch_size=8",
                                       "eval_steps=100",
                                       "save_steps=400",
                                       "use_wandb=True",
                                       "wandb_project=FRFPE_layoutlmv3"],
                     log_dir="/path/to/dir/Experiments/FRFPE/layoutlmv3",
                     dataset_val=merge,
                     metric=metric,
                     use_token_tag=False,
                     pipeline_component_name="LMTokenClassifierService",
                     segment_positions=[dd.LayoutType.title, 
                                        dd.LayoutType.text, 
                                        dd.LayoutType.table, 
                                        dd.LayoutType.list])
```

    [32m[0608 19:16.48 @adapter.py:77][0m  [32mINF[0m  [97mYielding dataflow into memory and create torch dataset[0m
    |                                                                                                                                                                                                           |0/?[00:00<?,?it/s][32m[0608 19:16.48 @logger.py:253][0m  [5m[35mWRN[0m  [97mDatapoint have images as np arrays stored and they will be loaded into memory. To avoid OOM set 'load_image'=False in dataflow build config. This will load images when needed and reduce memory costs!!![0m
    |                                                                                                                                                                                               |305/?[00:00<00:00,5006.70it/s]
    [32m[0608 19:16.48 @maputils.py:222][0m  [32mINF[0m  [97mGround-Truth category distribution:
     [36m|  category   | #box   |  category   | #box   |  category  | #box   |
    |:-----------:|:-------|:-----------:|:-------|:----------:|:-------|
    | report_date | 1017   | report_type | 682    |  umbrella  | 843    |
    |  fund_name  | 1721   |    other    | 10692  |            |        |
    |    total    | 14955  |             |        |            |        |[0m[0m
    |                                                                                                                                                                                             |305/?[00:00<00:00,911155.78it/s]



Syncing run <strong><a href='https://wandb.ai/jm76/FRFPE_layoutlmv3/runs/l78v5xln' target="_blank">feasible-hill-6</a></strong> to <a href='https://wandb.ai/jm76/FRFPE_layoutlmv3' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/run' target="_blank">docs</a>)<br/>



View project at <a href='https://wandb.ai/jm76/FRFPE_layoutlmv3' target="_blank">https://wandb.ai/jm76/FRFPE_layoutlmv3</a>



View run at <a href='https://wandb.ai/jm76/FRFPE_layoutlmv3/runs/l78v5xln' target="_blank">https://wandb.ai/jm76/FRFPE_layoutlmv3/runs/l78v5xln</a>


     {0: <FundsFirstPage.report_date>,
     1: <FundsFirstPage.report_type>,
     2: <FundsFirstPage.umbrella>,
     3: <FundsFirstPage.fund_name>,
     4: <TokenClasses.other>}[0m
    
     [36m|     key     | category_id   | val      | num_samples   |
    |:-----------:|:--------------|:---------|:--------------|
    |    word     | 1             | 1        | 2538          |
    | token_class | 1             | 0.980645 | 79            |
    | token_class | 2             | 0.87234  | 48            |
    | token_class | 3             | 0.722689 | 71            |
    | token_class | 4             | 0.851485 | 95            |
    | token_class | 5             | 0.991567 | 2245          |[0m[0m



```python
wandb.finish()
```


View run <strong style="color:#cdcd00">feasible-hill-6</strong> at: <a href='https://wandb.ai/jm76/FRFPE_layoutlmv3/runs/l78v5xln' target="_blank">https://wandb.ai/jm76/FRFPE_layoutlmv3/runs/l78v5xln</a><br/>Synced 5 W&B file(s), 21 media file(s), 560 artifact file(s) and 0 other file(s)


Find logs at: <code>./wandb/run-20230608_191648-l78v5xln/logs</code>


## Evaluation

Evaluation on the test split drops significantly. This is quite surprising as we haven't seen a F1-score drop of this size before. 
Especially `fund_name` and `other` have a significant drop. As there are much more `fund_name` labels in at least one sample
it looks like the model gets confused due to the segment bounding boxes.


```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/FRFPE/layoutlmv3/checkpoint-2000/config.json"
path_weights = "/path/to/dir/FRFPE/layoutlmv3/checkpoint-1600/pytorch_model.bin"

layoutlm_classifier = dd.HFLayoutLmv3TokenClassifier(path_config_json,
                                                     path_weights,
                                                     categories=categories)

tokenizer_fast = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

pipe_component = dd.LMTokenClassifierService(tokenizer_fast,
                                             layoutlm_classifier,
                                             use_other_as_default_category=True)

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0608 19:56.22 @eval.py:113][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0608 19:56.23 @eval.py:225][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:01<00:00, 20.03it/s]
    [32m[0608 19:56.24 @eval.py:207][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0608 19:56.24 @accmetric.py:373][0m  [32mINF[0m  [97mF1 results:
     [36m|     key     | category_id   | val      | num_samples   |
    |:-----------:|:--------------|:---------|:--------------|
    |    word     | 1             | 1        | 1505          |
    | token_class | 1             | 0.962162 | 89            |
    | token_class | 2             | 0.931298 | 69            |
    | token_class | 3             | 0.728571 | 86            |
    | token_class | 4             | 0.565341 | 490           |
    | token_class | 5             | 0.822703 | 771           |[0m[0m


Many `fund_name` token have been mis-classified as `other`. And this happens particularly with segments that are rather large. 


```python
metric = dd.get_metric("confusion")
metric.set_categories(sub_category_names={"word": ["token_class"]})

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0608 20:06.23 @eval.py:113][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0608 20:06.24 @eval.py:225][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:01<00:00, 20.13it/s]
    [32m[0608 20:06.26 @eval.py:207][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0608 20:06.26 @accmetric.py:431][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  89 |   0 |   0 |   0 |   0 |
    |                  2 |   0 |  61 |   0 |   0 |   8 |
    |                  3 |   0 |   0 |  51 |  13 |  22 |
    |                  4 |   0 |   0 |   3 | 199 | 288 |
    |                  5 |   7 |   1 |   0 |   2 | 761 |[0m[0m



```python
evaluator.compare(interactive=True, split="test", show_words=True)
```

## Conclusion

The results show that LayoutLMv3 is not the best choice for this dataset and it is being outperformed by LayoutXLM.

It is likely to get better results with text line segment bounding boxes. This assumption is backed by the fact that 
the model has difficulties to deliver consistent results especially when the segment bounding box is too large. 
To confirm this assumption, however, one would have to adjust the dataset.
