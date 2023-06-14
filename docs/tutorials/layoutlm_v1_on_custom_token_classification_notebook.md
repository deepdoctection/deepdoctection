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

## Defining object types

**FRFPE** contains categories that have not been defined in **deep**doctection. These must first be made known to the framework. `TokenClasses.other` has already been defined.


```python
@dd.object_types_registry.register("ner_first_page")
class FundsFirstPage(dd.ObjectTypes):

    report_date = "report_date"
    umbrella = "umbrella"
    report_type = "report_type"
    fund_name = "fund_name"

dd.update_all_types_dict()
```

Download the dataset and save it to 


```python
dd.get_dataset_dir_path() / "FRFPE"
```

## Visualization and display of ground truth


```python
path = dd.get_dataset_dir_path() / "FRFPE" / "40952248ba13ae8bfdd39f56af22f7d9_0.json"

page = dd.Page.from_file(path)
page.image =  dd.load_image_from_file(path.parents[0]  / "image" / page.file_name.replace("pdf","png"))
page.viz(interactive=True,show_words=True)  # close interactive window with q
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
                                                 categories=dd.LayoutType.word,
                                                 sub_categories={dd.LayoutType.word: dd.WordType.token_class},
                                                 keys = False,
                                                 values_as_dict=True,
                                                 name_as_key=True)))

        return df
```

    [32m[0608 11:38.49 @file_utils.py:33][0m  [32mINF[0m  [97mPyTorch version 1.9.0+cu111 available.[0m



```python
ner = dd.CustomDataset(name = "FRFPE",
                 dataset_type=dd.DatasetType.token_classification,
                 location="FRFPE",
                 init_categories=[dd.Layout.text, dd.LayoutType.title, dd.LayoutType.list, dd.LayoutType.table,
                                  dd.LayoutType.figure, dd.LayoutType.line, dd.LayoutType.word],
                 init_sub_categories={dd.LayoutType.word: {dd.WordType.token_class: [FundsFirstPage.report_date,
                                                                                     FundsFirstPage.report_type,
                                                                                     FundsFirstPage.umbrella,
                                                                                     FundsFirstPage.fund_name,
                                                                                     dd.TokenClasses.other],
                                                           dd.WordType.tag: []}},
                 dataflow_builder=NerBuilder)
```

## Defining a split and saving the split distribution as W&B artifact 

The ground truth contains some layout sections `ImageAnnotation` that we need to explicitly filter out.

We define a split with ~90% train, ~5% validation and ~5% test samples.

To reproduce the split later we save the split as a W&B artifact.


```python
ner.dataflow.categories.filter_categories(categories=dd.LayoutType.word)

merge = dd.MergeDataset(ner)
merge.buffer_datasets()
merge.split_datasets(ratio=0.1)
```

    [32m[0608 11:39.02 @base.py:308][0m  [32mINF[0m  [97m___________________ Number of datapoints per split ___________________[0m
    [32m[0608 11:39.02 @base.py:309][0m  [32mINF[0m  [97m{'test': 26, 'train': 305, 'val': 26}[0m



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

    [32m[0608 11:39.10 @jupyter.py:224][0m  [4m[5m[31mERR[0m  [97mFailed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.[0m
    [34m[1mwandb[0m: Currently logged in as: [33mjm76[0m. Use [1m`wandb login --relogin`[0m to force relogin



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

0: <FundsFirstPage.report_date>,
1: <FundsFirstPage.report_type>,
2: <FundsFirstPage.umbrella>,
3: <FundsFirstPage.fund_name>,
4: <TokenClasses.other>


```python
dd.train_hf_layoutlm(path_config_json,
                     merge,
                     path_weights,
                     config_overwrite=["max_steps=2000",
                                       "per_device_train_batch_size=8",
                                       "eval_steps=100",
                                       "save_steps=400",
                                       "use_wandb=True",
                                       "wandb_project=FRFPE_layoutlmv1"],
                     log_dir="/home/janis/Experiments/FRFPE/layoutlmv1",
                     dataset_val=merge,
                     metric=metric,
                     use_token_tag=False,
                     pipeline_component_name="LMTokenClassifierService")
```

    [32m[0608 11:39.36 @maputils.py:222][0m  [32mINF[0m  [97mGround-Truth category distribution:
     [36m|  category   | #box   |  category   | #box   |  category  | #box   |
    |:-----------:|:-------|:-----------:|:-------|:----------:|:-------|
    | report_date | 1017   | report_type | 682    |  umbrella  | 843    |
    |  fund_name  | 1721   |    other    | 10692  |            |        |
    |    total    | 14955  |             |        |            |        |[0m[0m
    |                                                                                                                                                                                             |305/?[00:00<00:00,358537.76it/s]



    [32m[0608 11:39.37 @hf_layoutlm_train.py:425][0m  [5m[35mWRN[0m  [97mAfter 305 dataloader will log warning at every iteration about unexpected samples[0m
    [32m[0608 11:39.37 @hf_layoutlm_train.py:430][0m  [32mINF[0m  [97mConfig: 
     {'output_dir': '/path/to/dir/Experiments/FRFPE/layoutlmv1', 'overwrite_output_dir': False, 'do_train': False, 'do_eval': True, 'do_predict': False, 'evaluation_strategy': 'steps', 'prediction_loss_only': False, 'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 8, 'per_gpu_train_batch_size': None, 'per_gpu_eval_batch_size': None, 'gradient_accumulation_steps': 1, 'eval_accumulation_steps': None, 'eval_delay': 0, 'learning_rate': 5e-05, 'weight_decay': 0.0, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-08, 'max_grad_norm': 1.0, 'num_train_epochs': 3.0, 'max_steps': 2000, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.0, 'warmup_steps': 0, 'log_level': 'passive', 'log_level_replica': 'warning', 'log_on_each_node': True, 'logging_dir': '/home/janis/Experiments/FRFPE/layoutlmv1/runs/Jun08_11-39-37_janis-x299-ud4-pro-local', 'logging_strategy': 'steps', 'logging_first_step': False, 'logging_steps': 500, 'logging_nan_inf_filter': True, 'save_strategy': 'steps', 'save_steps': 400, 'save_total_limit': None, 'save_safetensors': False, 'save_on_each_node': False, 'no_cuda': False, 'use_mps_device': False, 'seed': 42, 'data_seed': None, 'jit_mode_eval': False, 'use_ipex': False, 'bf16': False, 'fp16': False, 'fp16_opt_level': 'O1', 'half_precision_backend': 'auto', 'bf16_full_eval': False, 'fp16_full_eval': False, 'tf32': None, 'local_rank': -1, 'xpu_backend': None, 'tpu_num_cores': None, 'tpu_metrics_debug': False, 'debug': [], 'dataloader_drop_last': False, 'eval_steps': 100, 'dataloader_num_workers': 0, 'past_index': -1, 'run_name': '/home/janis/Experiments/FRFPE/layoutlmv1', 'disable_tqdm': False, 'remove_unused_columns': False, 'label_names': None, 'load_best_model_at_end': False, 'metric_for_best_model': None, 'greater_is_better': None, 'ignore_data_skip': False, 'sharded_ddp': [], 'fsdp': [], 'fsdp_min_num_params': 0, 'fsdp_config': {'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False}, 'fsdp_transformer_layer_cls_to_wrap': None, 'deepspeed': None, 'label_smoothing_factor': 0.0, 'optim': 'adamw_hf', 'optim_args': None, 'adafactor': False, 'group_by_length': False, 'length_column_name': 'length', 'report_to': ['tensorboard', 'wandb'], 'ddp_find_unused_parameters': None, 'ddp_bucket_cap_mb': None, 'dataloader_pin_memory': True, 'skip_memory_metrics': True, 'use_legacy_prediction_loop': False, 'push_to_hub': False, 'resume_from_checkpoint': None, 'hub_model_id': None, 'hub_strategy': 'every_save', 'hub_token': '<HUB_TOKEN>', 'hub_private_repo': False, 'gradient_checkpointing': False, 'include_inputs_for_metrics': False, 'fp16_backend': 'auto', 'push_to_hub_model_id': None, 'push_to_hub_organization': None, 'push_to_hub_token': '<PUSH_TO_HUB_TOKEN>', 'mp_parameters': '', 'auto_find_batch_size': False, 'full_determinism': False, 'torchdynamo': None, 'ray_scope': 'last', 'ddp_timeout': 1800, 'torch_compile': False, 'torch_compile_backend': None, 'torch_compile_mode': None}[0m
    [32m[0608 11:39.37 @hf_layoutlm_train.py:434][0m  [32mINF[0m  [97mWill setup a head with the following classes
     {0: <FundsFirstPage.report_date>,
     1: <FundsFirstPage.report_type>,
     2: <FundsFirstPage.umbrella>,
     3: <FundsFirstPage.fund_name>,
     4: <TokenClasses.other>}
    
     [36m|     key     | category_id   | val      | num_samples   |
    |:-----------:|:--------------|:---------|:--------------|
    |    word     | 1             | 1        | 2538          |
    | token_class | 1             | 0.96732  | 79            |
    | token_class | 2             | 0.710526 | 48            |
    | token_class | 3             | 0.880597 | 71            |
    | token_class | 4             | 0.882682 | 95            |
    | token_class | 5             | 0.987208 | 2245          |


## Further exploration of evaluation

### Evaluation with confusion matrix

```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/Experiments/FRFPE/layoutlmv1/checkpoint-1600/config.json"
path_weights = "/path/to/dir/Experiments/FRFPE/layoutlmv1/checkpoint-1600/pytorch_model.bin"

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

    [32m[0608 12:48.27 @accmetric.py:431][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |    5 |
    |     ground truth | |     |     |     |     |      |
    |                  v |     |     |     |     |      |
    |-------------------:|----:|----:|----:|----:|-----:|
    |                  1 |  73 |   0 |   0 |   0 |    6 |
    |                  2 |   0 |  28 |   0 |   0 |   20 |
    |                  3 |   0 |   0 |  59 |   3 |    9 |
    |                  4 |   0 |   0 |   1 |  80 |   14 |
    |                  5 |   0 |   1 |   5 |   1 | 2238 |[0m[0m


###  Visualizing predictions and ground truth


```python
evaluator.compare(interactive=True, split="val", show_words=True)
```

## Evaluation on test set

Comparing the evaluation results of eval and test split we see a deterioration of `fund_name`  F1-score (too many erroneous as `umbrella`). All remaining labels are slightly worse.


```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/Experiments/FRFPE/layoutlmv1/checkpoint-1600/config.json"
path_weights = "/path/to/dir/Experiments/FRFPE/layoutlmv1/checkpoint-1600/pytorch_model.bin"

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

    [32m[0608 12:15.45 @accmetric.py:373][0m  [32mINF[0m  [97mF1 results:
     [36m|     key     | category_id   | val      | num_samples   |
    |:-----------:|:--------------|:---------|:--------------|
    |    word     | 1             | 1        | 1505          |
    | token_class | 1             | 0.95082  | 89            |
    | token_class | 2             | 0.809524 | 69            |
    | token_class | 3             | 0.666667 | 86            |
    | token_class | 4             | 0.857464 | 490           |
    | token_class | 5             | 0.900782 | 771           |[0m[0m
    [34m[1mwandb[0m: [33mWARNING[0m `log` ignored (called from pid=35652, `init` called from pid=None). See: http://wandb.me/init-multiprocess



```python
metric = dd.get_metric("confusion")
metric.set_categories(sub_category_names={"word": ["token_class"]})

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0608 12:17.46 @accmetric.py:431][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  87 |   0 |   0 |   0 |   2 |
    |                  2 |   0 |  51 |   0 |   0 |  18 |
    |                  3 |   0 |   0 |  49 |  13 |  24 |
    |                  4 |   0 |   0 |   9 | 382 |  99 |
    |                  5 |   7 |   6 |   3 |   6 | 749 |[0m[0m



```python
evaluator.compare(interactive=True, split="test", show_words=True)
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
                         log_dir="/home/janis/Experiments/ner_first_page_v1_2",
                         dataset_val=merge,
                         metric=metric,
                         use_token_tag=False,
                         pipeline_component_name="LMTokenClassifierService")
``` 

