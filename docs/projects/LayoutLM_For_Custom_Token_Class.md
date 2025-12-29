<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>

# Project: LayoutLM for financial report NER

The goal is to fine-tune the LayoutLM model for token classification on a custom dataset. The goal is to give a 
realistic setting and to document the findings.

We use the self-labeled dataset Funds Report Front Page Entities (FRFPE), which can be downloaded from the 
[Huggingface Hub](https://huggingface.co/datasets/deepdoctection/fund_ar_front_page). 

For experimentation, we use the W&B framework that can be activated when starting the training script. 


## Step 1: Defining custom object types

**FRFPE** contains categories that have not been defined in **deep**doctection. These must first be added to the 
framework. 


```python
import deepdoctection as dd
from collections import defaultdict
import wandb
from transformers import LayoutLMTokenizerFast

@dd.object_types_registry.register("ner_first_page")
class FundsFirstPage(dd.ObjectTypes):

    REPORT_DATE = "report_date"
    UMBRELLA = "umbrella"
    REPORT_TYPE = "report_type"
    FUND_NAME = "fund_name"
```

## Step 2: Downloading the dataset

Download the dataset and save it to 


```python
import os
from pathlib import Path

frfpe_path = Path(os.environ["DATASET_DIR"]) / "FRFPE"
```


## Step 3: Visualization and display of ground truth


```python
path = frfpe_path / "40952248ba13ae8bfdd39f56af22f7d9_0.json"

page = dd.Page.from_file(path)
page.image =  dd.load_image_from_file(path.parents[0]  / 
									  "image" / page.file_name.replace("pdf","png"))

page.viz(interactive=True,
		 show_words=True)  # (1) 
```

1. Close interactive window with 'q'


```python
for word in page.words:
    print(f"word: {word.characters}, category: {word.token_class}, bio: {word.tag}")
```

??? info "Output"

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
    ```


## Step 4: Defining Dataflow and Dataset

We define a dataflow and use the `CustomDataset` class.


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

        df = dd.SerializerFiles.load(ann_files_dir,".json")   # (1) 
        df = dd.MapData(df, dd.Image.from_file)   # (2) 

        df = dd.MapData(df, overwrite_location_and_load(image_dir, load_image))

        if self.categories.is_filtered():
            df = dd.MapData(
                df,
                dd.filter_cat(
                    self.categories.get_categories(as_dict=False, 
												   filtered=True),
                    self.categories.get_categories(as_dict=False, 
												   filtered=False),
                ),
            )
        df = dd.MapData(df,
			 dd.re_assign_cat_ids(cat_to_sub_cat_mapping=
			 self.categories.get_sub_categories(
                                  categories=dd.LayoutType.WORD,
                                  sub_categories={dd.LayoutType.WORD: 
												  dd.WordType.TOKEN_CLASS},
                                  keys = False,
                                  values_as_dict=True,
                                  name_as_key=True)))

        return df
```

1. Get a stream of `.json` files
2. Load `.json` file


```python
ner = dd.CustomDataset(name = "FRFPE",
                 dataset_type=dd.DatasetType.TOKEN_CLASSIFICATION,
                 location="FRFPE",
                 init_categories=[dd.LayoutType.TEXT, 
								  dd.LayoutType.TITLE, 
								  dd.LayoutType.LIST, 
								  dd.LayoutType.TABLE,
                                  dd.LayoutType.FIGURE, 
								  dd.LayoutType.LINE, 
								  dd.LayoutType.WORD],
                 init_sub_categories={dd.LayoutType.WORD: 
									 {dd.WordType.TOKEN_CLASS: 
									  [FundsFirstPage.REPORT_DATE,
									   FundsFirstPage.REPORT_TYPE,
									   FundsFirstPage.UMBRELLA,
									   FundsFirstPage.FUND_NAME,
									   dd.TokenClasses.OTHER],
                                      dd.WordType.TAG: []}},
                 dataflow_builder=NerBuilder)
```

## Step 5: Defining a split and saving the split distribution as W&B artifact 

- The ground truth contains some layout sections `ImageAnnotation` that we need to explicitly filter out.
- We define a split with ~90% train, ~5% validation and ~5% test samples.
- To reproduce the split later we save the split as a W&B artifact.


```python
ner.dataflow.categories.filter_categories(categories=dd.LayoutType.WORD)

merge = dd.MergeDataset(ner)
merge.buffer_datasets()
merge.split_datasets(ratio=0.1)
```

!!! info "Output"

     ___________________ Number of datapoints per split ___________________
                   {'test': 15, 'train': 327, 'val': 15}



```python
out = merge.get_ids_by_split()

table_rows=[]
for split, split_list in out.items():
    for ann_id in split_list:
        table_rows.append([split,ann_id])
table = wandb.Table(columns=["split","annotation_id"], 
					data=table_rows)

wandb.init(project="FRFPE_layoutlmv1")

artifact = wandb.Artifact(merge.dataset_info.name, type='dataset')
artifact.add(table, "split")

wandb.log_artifact(artifact)
wandb.finish()
```


## Step 6: LayoutLM fine-tuning

```python
path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")
path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlm-base-uncased/pytorch_model.bin")

metric = dd.get_metric("f1")
metric.set_categories(sub_category_names={"word": ["token_class"]})
```

!!! info

    Remember id to label mapping:

    ``` 
    0: FundsFirstPage.REPORT_DATE,
    1: FundsFirstPage.REPORT_TYPE,
    2: FundsFirstPage.UMBRELLA,
    3: FundsFirstPage.FUND_NAME,
    4: TokenClasses.OTHER
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
                     pipeline_component_name="LMTokenClassifierService")
```


## Step 7: Evaluation with confusion matrix



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

??? info "Output"

    Confusion matrix: 

    |    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  41 |   0 |   0 |   0 |   4 |
    |                  2 |   0 |  19 |   0 |   0 |   8 |
    |                  3 |   0 |   0 |  20 |   4 |  14 |
    |                  4 |   0 |   0 |   0 |  25 |   5 |
    |                  5 |   0 |   0 |   1 |   1 | 657 |


###  Step 8: Visualizing predictions and ground truth


```python
result = evaluator.compare(interactive=True, split="val", show_words=True)
sample = next(iter(result))
sample.viz()
```

## Step 9: Evaluation on test set

Comparing the evaluation results of eval and test split we see a deterioration of `fund_name`  F1-score (too many 
erroneous as `umbrella`). All remaining labels are slightly worse.


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

???? info "Output"

    Confusion matrix:

    |    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  53 |   0 |   0 |   0 |   2 |
    |                  2 |   0 |  20 |   0 |   0 |  12 |
    |                  3 |   0 |   0 |  25 |   6 |  10 |
    |                  4 |   0 |   0 |   4 | 292 |  33 |
    |                  5 |   0 |   0 |   4 |   8 | 482 |


```python
wandb.finish()
```

!!! note "Changing training parameters and settings"

    Already the first training run delivers satisfactory results. The following parameters could still be changed:

    `per_device_train_batch_size`
    `max_steps` 

!!! note "Sliding Window"

     `sliding_window_stride` - LayoutLM accepts a maximum of 512 tokens. For samples containing more tokens a sliding 
     window can be used: Assume a training sample contains 600 tokens. Without sliding window the last 88 tokens are 
     not considered in the training. If a sliding window of 16 is set, 88 % 16 +1= 9 samples are generated. 

    *Caution:* 
    - The number `per_device_train_batch_size` can increase very fast and lead to Cuda OOM.
    - If a sample occurs that generates multiple training points due to the sliding windows setting, all other samples 
      in the batch will be ignored and only the one data point with all its windows will be considered in this step. 
      If you train with a dataset where the number of tokens is high for many samples, you should choose 
      `per_device_train_batch_size` to be rather small to ensure that you train with the whole dataset. 

    To avoid the situation to have a very large batch size becomes due to the sliding windos, we can add a 
    `max_batch_size`. Setting this parameter causes a selection of `max_batch_size` samples to be randomly sent to the 
    GPU from the generated sliding window samples.

    The training script will be looking like this:

    ```python
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
