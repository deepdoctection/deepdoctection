# LayoutLMv1 for financial report NER

This notebook is the beginning of a series of training and evaluation scripts for the LayoutLM family of models.

The goal is to train the models LayoutLMv1, LayoutLMv2, LayoutXLM and LayoutLMv3 for token classification on a custom dataset. The goal is to give a realistic setting and to document the findings.

We use the self-labeled dataset Funds Report Front Page Entities (FRFPE), which can be downloaded from [Huggingface](https://huggingface.co/datasets/deepdoctection/fund_ar_front_page). 

For experimentation, we use the W&B framework, which is integrated into the training and evaluation. 


```python
import deepdoctection as dd
from collections import defaultdict
import wandb
from transformers import LayoutLMTokenizerFast
```

    /home/janis/Documents/Repos/deepdoctection_pt/.venv/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    [32m[0712 18:07.41 @file_utils.py:36][0m  [32mINF[0m  [97mPyTorch version 2.1.2+cu121 available.[0m
    [32m[0712 18:07.41 @file_utils.py:74][0m  [32mINF[0m  [97mDisabling Tensorflow because USE_TORCH is set[0m


## Defining object types

**FRFPE** contains categories that have not been defined in **deep**doctection. These must first be made known to the framework. `TokenClasses.other` has already been defined.


```python
@dd.object_types_registry.register("ner_first_page")
class FundsFirstPage(dd.ObjectTypes):

    REPORT_DATE = "report_date"
    UMBRELLA = "umbrella"
    REPORT_TYPE = "report_type"
    FUND_NAME = "fund_name"

dd.update_all_types_dict()
```

Download the dataset and save it to 


```python
dd.get_dataset_dir_path() / "FRFPE"
```




    PosixPath('/media/janis/Elements/.cache/deepdoctection/datasets/FRFPE')



## Visualization and display of ground truth


```python
path = dd.get_dataset_dir_path() / "FRFPE" / "40952248ba13ae8bfdd39f56af22f7d9_0.json"

page = dd.Page.from_file(path)
page.image =  dd.load_image_from_file(path.parents[0]  / "image" / page.file_name.replace("pdf","png"))
#page.viz(interactive=True,show_words=True)  # close interactive window with q
```


```python
for word in page.words:
    print(f"word: {word.characters}, category: {word.token_class}, bio: {word.tag}")
```

    word: GFG, category: umbrella, bio: B
    word: Funds, category: umbrella, bio: I
    word: Société, category: other, bio: O
    word: d, category: other, bio: O
    word: ', category: other, bio: O
    word: Investissement, category: other, bio: O
    word: à, category: other, bio: O
    word: Capital, category: other, bio: O
    word: Variable, category: other, bio: O
    word: incorporated, category: other, bio: O
    word: in, category: other, bio: O
    word: Luxembourg, category: other, bio: O
    word: Luxembourg, category: other, bio: O
    word: R, category: other, bio: O
    word: ., category: other, bio: O
    word: C, category: other, bio: O
    word: ., category: other, bio: O
    word: S, category: other, bio: O
    word: ., category: other, bio: O
    word: B60668, category: other, bio: O
    word: Unaudited, category: other, bio: O
    word: Semi-Annual, category: report_type, bio: B
    word: Report, category: report_type, bio: I
    word: as, category: other, bio: O
    word: at, category: other, bio: O
    word: 30.06.2021, category: report_date, bio: B


We will not use the `word.tag`.

## Defining Dataflow and Dataset

We define dataflow and use the interface for the `CustomDataset`.


```python
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
                                                 categories=dd.LayoutType.WORD,
                                                 sub_categories={dd.LayoutType.WORD: dd.WordType.TOKEN_CLASS},
                                                 keys = False,
                                                 values_as_dict=True,
                                                 name_as_key=True)))

        return df
```


```python
ner = dd.CustomDataset(name = "FRFPE",
                 dataset_type=dd.DatasetType.TOKEN_CLASSIFICATION,
                 location="FRFPE",
                 init_categories=[dd.LayoutType.TEXT, dd.LayoutType.TITLE, dd.LayoutType.LIST, dd.LayoutType.TABLE,
                                  dd.LayoutType.FIGURE, dd.LayoutType.LINE, dd.LayoutType.WORD],
                 init_sub_categories={dd.LayoutType.WORD: {dd.WordType.TOKEN_CLASS: [FundsFirstPage.REPORT_DATE,
                                                                                     FundsFirstPage.REPORT_TYPE,
                                                                                     FundsFirstPage.UMBRELLA,
                                                                                     FundsFirstPage.FUND_NAME,
                                                                                     dd.TokenClasses.OTHER],
                                                           dd.WordType.TAG: []}},
                 dataflow_builder=NerBuilder)
```

## Defining a split and saving the split distribution as W&B artifact 

The ground truth contains some layout sections `ImageAnnotation` that we need to explicitly filter out.

We define a split with ~90% train, ~5% validation and ~5% test samples.

To reproduce the split later we save the split as a W&B artifact.


```python
ner.dataflow.categories.filter_categories(categories=dd.LayoutType.WORD)

merge = dd.MergeDataset(ner)
merge.buffer_datasets()
merge.split_datasets(ratio=0.1)
```

    [32m[0712 18:08.06 @base.py:259][0m  [32mINF[0m  [97mWill use the same build setting for all dataflows[0m
    |                                                                                                                                                                                                                                                                                                   |357/?[00:00<00:00,26702.93it/s]
    |                                                                                                                                                                                                                                                                                                      |357/?[00:05<00:00,70.27it/s]
    [32m[0712 18:08.12 @base.py:314][0m  [32mINF[0m  [97m___________________ Number of datapoints per split ___________________[0m
    [32m[0712 18:08.12 @base.py:315][0m  [32mINF[0m  [97m{'test': 15, 'train': 327, 'val': 15}[0m



```python
out = merge.get_ids_by_split()

table_rows=[]
for split, split_list in out.items():
    for ann_id in split_list:
        table_rows.append([split,ann_id])
table = wandb.Table(columns=["split","annotation_id"], data=table_rows)

wandb.init(project="FRFPE_layoutlmv1")

artifact = wandb.Artifact(merge.dataset_info.name, type='dataset')
artifact.add(table, "split")

wandb.log_artifact(artifact)
wandb.finish()
```




    ArtifactManifestEntry(path='split.table.json', digest='Y4OPSqZ/Z3PlQYSWBBDbxw==', size=18543, local_path='/home/janis/.local/share/wandb/artifacts/staging/tmpkptwl7lb', skip_cache=False)



## LayoutLMv1 training

Next we setup training with LayoutLMv1.


```python
path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")
path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlm-base-uncased/pytorch_model.bin")
```


```python
metric = dd.get_metric("f1")
metric.set_categories(sub_category_names={"word": ["token_class"]})
```

Remember id to label mapping:

``` 
0: <FundsFirstPage.REPORT_DATE>,
1: <FundsFirstPage.REPORT_TYPE>,
2: <FundsFirstPage.UMBRELLA>,
3: <FundsFirstPage.FUND_NAME>,
4: <TokenClasses.OTHER>
```



```python
dd.train_hf_layoutlm(path_config_json,
                     merge,
                     path_weights,
                     config_overwrite=["max_steps=2000",
                                       "per_device_train_batch_size=8",
                                       "eval_steps=100",
                                       "save_steps=400",
                                       "use_wandb=True",
                                       "wandb_project=FRFPE_layoutlmv1",
                                      ],
                     log_dir="/path/to/dir/Experiments/FRFPE/layoutlmv1",
                     dataset_val=merge,
                     metric=metric,
                     use_token_tag=False,
                     pipeline_component_name="LMTokenClassifierService")
```

## Further exploration of evaluation

### Evaluation with confusion matrix



```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/Experiments/FRFPE/layoutlmv1/checkpoint-1600/config.json"
path_weights = "/path/to/dir/Experiments/FRFPE/layoutlmv1/checkpoint-1600/model.safetensors"

layoutlm_classifier = dd.HFLayoutLmTokenClassifier(path_config_json,
                                                   path_weights,
                                                   categories=categories)

tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
pipe_component = dd.LMTokenClassifierService(tokenizer_fast,
                                             layoutlm_classifier,
                                             use_other_as_default_category=True)
metric = dd.get_metric("confusion")
metric.set_categories(sub_category_names={"word": ["token_class"]})
evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="val")
```

    [32m[0712 18:20.04 @eval.py:112][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0712 18:20.12 @eval.py:226][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:05<00:00,  2.95it/s]
    [32m[0712 18:20.17 @eval.py:208][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0712 18:20.17 @accmetric.py:429][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  41 |   0 |   0 |   0 |   4 |
    |                  2 |   0 |  19 |   0 |   0 |   8 |
    |                  3 |   0 |   0 |  20 |   4 |  14 |
    |                  4 |   0 |   0 |   0 |  25 |   5 |
    |                  5 |   0 |   0 |   1 |   1 | 657 |[0m[0m


###  Visualizing predictions and ground truth


```python
result = evaluator.compare(interactive=True, split="val", show_words=True)
sample = next(iter(result))
sample.viz()
```

## Evaluation on test set

Comparing the evaluation results of eval and test split we see a deterioration of `fund_name`  F1-score (too many erroneous as `umbrella`). All remaining labels are slightly worse.


```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/FRFPE/layoutlmv1/checkpoint-1600/config.json"
path_weights = "/path/to/dir/FRFPE/layoutlmv1/checkpoint-1600/model.safetensors"

layoutlm_classifier = dd.HFLayoutLmTokenClassifier(path_config_json,
                                                   path_weights,
                                                   categories=categories)

tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
pipe_component = dd.LMTokenClassifierService(tokenizer_fast,
                                             layoutlm_classifier,
                                             use_other_as_default_category=True)


evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0712 18:24.09 @eval.py:112][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0712 18:24.17 @eval.py:226][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:05<00:00,  2.92it/s]
    [32m[0712 18:24.22 @eval.py:208][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0712 18:24.22 @accmetric.py:429][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  53 |   0 |   0 |   0 |   2 |
    |                  2 |   0 |  20 |   0 |   0 |  12 |
    |                  3 |   0 |   0 |  25 |   6 |  10 |
    |                  4 |   0 |   0 |   4 | 292 |  33 |
    |                  5 |   0 |   0 |   4 |   8 | 482 |[0m[0m



```python
metric = dd.get_metric("confusion")
metric.set_categories(sub_category_names={"word": ["token_class"]})

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0712 18:24.26 @eval.py:112][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0712 18:24.34 @eval.py:226][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:05<00:00,  2.91it/s]
    [32m[0712 18:24.39 @eval.py:208][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0712 18:24.39 @accmetric.py:429][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  53 |   0 |   0 |   0 |   2 |
    |                  2 |   0 |  20 |   0 |   0 |  12 |
    |                  3 |   0 |   0 |  25 |   6 |  10 |
    |                  4 |   0 |   0 |   4 | 292 |  33 |
    |                  5 |   0 |   0 |   4 |   8 | 482 |[0m[0m



```python
result = evaluator.compare(interactive=True, split="test", show_words=True)
sample = next(iter(result))
sample.viz()
```


```python
wandb.finish()
```

## Changing training parameters and settings

Already the first training run delivers satisfactory results. The following parameters could still be changed:

**Training specific** (Check the Transformers doc to know more about these parameters): 
`per_device_train_batch_size`
`max_steps` 

**General**: 
`sliding_window_stride` - LayoutLMv1 accepts a maximum of 512 tokens. For samples containing more tokens a sliding window can be used: Assume a training sample contains 600 tokens. Without sliding window the last 88 tokens are not considered in the training. If a sliding window of 16 is set, 88 % 16 +1= 9 samples are generated. 

Caution: 
- The number `per_device_train_batch_size` can thereby turn out to increase very fast and lead to Cuda OOM.
- If a sample occurs that generates multiple training points due to the sliding windows setting, all other samples in the batch will be ignored and only the one data point with all its windows will be considered in this step. If you train with a dataset where the number of tokens is high for many samples, you should choose `per_device_train_batch_size` to be rather small to ensure that you train with the whole dataset. 

To avoid the situation that due to the `sliding_window_stride` the batch size becomes arbitrarily large one can select the additional `max_batch_size`:
Setting this parameter causes a selection of max_batch_size samples to be randomly sent to the GPU from the generated sliding window samples.

E.g. another training configuration might look like this:


```
dd.train_hf_layoutlm(path_config_json,
                     merge,
                     path_weights,
                     config_overwrite=["max_steps=4000",
                                       "per_device_train_batch_size=8",
                                       "eval_steps=100",
                                       "save_steps=400",
                                       "sliding_window_stride=16",
                                       "max_batch_size=8",
                                       "use_wandb=True",
                                       "wandb_project=funds_layoutlmv1"],
                         log_dir="/path/to/dir/Experiments/ner_first_page_v1_2",
                         dataset_val=merge,
                         metric=metric,
                         use_token_tag=False,
                         pipeline_component_name="LMTokenClassifierService")
``` 

