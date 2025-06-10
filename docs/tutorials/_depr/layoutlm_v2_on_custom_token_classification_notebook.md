# LayoutLMv2 for financial report NER

We now come to the training and evaluation of LayoutLMv2 and LayoutXLM.

We use the same split that we used for training LayoutLMv1 and load the artifact from W&B for this.

Needless to say, that we need to define dataset and `ObjectType`s again.


```python
import deepdoctection as dd
from collections import defaultdict
import wandb
from transformers import LayoutLMTokenizerFast, XLMRobertaTokenizerFast
```

## Defining `ObjectTypes`, Dataset and Dataflow


```python
@dd.object_types_registry.register("ner_first_page")
class FundsFirstPage(dd.ObjectTypes):

    REPORT_DATE = "report_date"
    UMBRELLA = "umbrella"
    REPORT_TYPE = "report_type"
    FUND_NAME = "fund_name"

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
                                                 categories=dd.LayoutType.WORD,
                                                 sub_categories={dd.LayoutType.WORD: dd.WordType.TOKEN_CLASS},
                                                 keys = False,
                                                 values_as_dict=True,
                                                 name_as_key=True)))
        
        if filter_languages:
            df = dd.MapData(df, dd.filter_summary({"language": [dd.get_type(lang) for lang in filter_languages]},
                                                 mode="value"))

        return df
    
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

ner.dataflow.categories.filter_categories(categories=dd.LayoutType.word)
df = ner.dataflow.build(load_image=True)

merge = dd.MergeDataset(ner)
merge.explicit_dataflows(df)
merge.buffer_datasets()
```

## Loading W&B artifact and building dataset split


```python
wandb.init(project="FRFPE_layoutlmv1", resume=True)
artifact = wandb.use_artifact('jm76/FRFPE_layoutlmv1/merge_FRFPE:v0', type='dataset')
table = artifact.get("split")
```


```python
split_dict = defaultdict(list)
for row in table.data:
    split_dict[row[0]].append(row[1])

merge.create_split_by_id(split_dict)
```

    [32m[0608 14:46.11 @base.py:250][0m  [32mINF[0m  [97mWill used dataflow from previously explicitly passed configuration[0m
    |                                                                                                                                                                                                 |357/?[00:28<00:00,12.63it/s]



```python
wandb.finish()
```

## Exporing the language distribustion across the split


```python
categories={1: dd.Languages.english, 2: dd.Languages.german, 3: dd.Languages.french}
categories_name_as_key = {val: key for key, val in categories.items()}

# train
summarizer_train = dd.LabelSummarizer(categories)
langs_train = []
for dp in merge._dataflow_builder.split_cache["train"]:
    langs_train.append(categories_name_as_key[dp.summary.get_sub_category("language").value])
summarizer_train.dump(langs_train)
   
# val
summarizer_val = dd.LabelSummarizer(categories)
langs_val = []
for dp in merge._dataflow_builder.split_cache["val"]:
    langs_val.append(categories_name_as_key[dp.summary.get_sub_category("language").value])
summarizer_val.dump(langs_val)

# test
summarizer_test = dd.LabelSummarizer(categories)
langs_test = []
for dp in merge._dataflow_builder.split_cache["test"]:
    langs_test.append(categories_name_as_key[dp.summary.get_sub_category("language").value])
summarizer_test.dump(langs_test)

train_summary = {categories[key]:val for key, val in summarizer_train.get_summary().items()}
val_summary= {categories[key]:val for key, val in summarizer_val.get_summary().items()}
test_summary = {categories[key]:val for key, val in summarizer_test.get_summary().items()}

print(f"train split: {train_summary}")
print(f"val split: {val_summary}")
print(f"test split: {test_summary}")
```

    train split: {<Languages.english>: 152, <Languages.german>: 145, <Languages.french>: 8}
    val split: {<Languages.english>: 11, <Languages.german>: 14, <Languages.french>: 1}
    test split: {<Languages.english>: 17, <Languages.german>: 9, <Languages.french>: 0}


Language is well balanced across the splits.


```python
dd.ModelCatalog.get_model_list() # find the model

# If you haven't downloaded the base model make sure to have it in your .cache
# dd.ModelDownloadManager.maybe_download_weights_and_configs("microsoft/layoutlmv2-base-uncased/pytorch_model.bin")
```


```python
path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlmv2-base-uncased/pytorch_model.bin")
path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlmv2-base-uncased/pytorch_model.bin")
```


```python
metric = dd.get_metric("f1")
metric.set_categories(sub_category_names={"word": ["token_class"]})
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
                                       "wandb_project=FRFPE_layoutlmv2"],
                     log_dir="/path/to/dir/Experiments/FRFPE/layoutlmv2",
                     dataset_val=merge,
                     metric=metric,
                     use_token_tag=False,
                     pipeline_component_name="LMTokenClassifierService")
```


```python
wandb.finish()
```

## Evaluation

Evaluation metrics show that the first checkpoint already delivers one of the best results.


```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/Experiments/FRFPE/layoutlmv2/checkpoint-400/config.json"
path_weights = "/path/to/dir/Experiments/FRFPE/layoutlmv2/checkpoint-400/model.safetensors"

layoutlm_classifier = dd.HFLayoutLmv2TokenClassifier(path_config_json,
                                                   path_weights,
                                                   categories=categories)

tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")  # tokenizer is the same as for LayoutLMv1
pipe_component = dd.LMTokenClassifierService(tokenizer_fast,
                                             layoutlm_classifier,
                                             use_other_as_default_category=True)

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0608 15:54.11 @eval.py:113][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0608 15:54.14 @eval.py:225][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:01<00:00, 17.09it/s]
    [32m[0608 15:54.15 @eval.py:207][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0608 15:54.15 @accmetric.py:373][0m  [32mINF[0m  [97mF1 results:
     [36m|     key     | category_id   | val      | num_samples   |
    |:-----------:|:--------------|:---------|:--------------|
    |    word     | 1             | 1        | 1505          |
    | token_class | 1             | 0.950276 | 89            |
    | token_class | 2             | 0.790323 | 69            |
    | token_class | 3             | 0.688312 | 86            |
    | token_class | 4             | 0.858974 | 490           |
    | token_class | 5             | 0.90031  | 771           |[0m[0m


There is little to no improvement compared with LayouLMv1.


```python
metric = dd.get_metric("confusion")
metric.set_categories(sub_category_names={"word": ["token_class"]})

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    [32m[0608 16:09.51 @eval.py:113][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0608 16:09.54 @eval.py:225][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:01<00:00, 17.40it/s]
    [32m[0608 16:09.55 @eval.py:207][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0608 16:09.55 @accmetric.py:431][0m  [32mINF[0m  [97mConfusion matrix: 
     [36m|    predictions ->  |   1 |   2 |   3 |   4 |   5 |
    |     ground truth | |     |     |     |     |     |
    |                  v |     |     |     |     |     |
    |-------------------:|----:|----:|----:|----:|----:|
    |                  1 |  86 |   0 |   0 |   0 |   3 |
    |                  2 |   0 |  49 |   0 |   0 |  20 |
    |                  3 |   0 |   0 |  53 |  12 |  21 |
    |                  4 |   0 |   0 |  15 | 402 |  73 |
    |                  5 |   6 |   6 |   0 |  32 | 727 |[0m[0m


# LayoutXLM for financial report NER

Next, we turn our attention to LayoutXLM which is a multi-language model. The training setting in the first experiment will be unchanged.


```python
path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutxlm-base/pytorch_model.bin")
path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutxlm-base/pytorch_model.bin")

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
                                       "wandb_project=FRFPE_layoutxlm"],
                         log_dir="/path/to/dir/Experiments/FRFPE/layoutxlm",
                         dataset_val=merge,
                         metric=metric,
                         use_xlm_tokenizer=True, # layoutv2 are layoutlm are from layer perspective identical. However, they do not share the same tokenizer. We therefore need to provide the information to the training script.
                         use_token_tag=False,
                         pipeline_component_name="LMTokenClassifierService")
```

Evalutation result look a lot more promising. We get an F1-score close to 0.9 along all labels.  

This is backed by the very impressive F1-results on the test split.


```python
categories = ner.dataflow.categories.get_sub_categories(categories="word",
                                                        sub_categories={"word": ["token_class"]},
                                                        keys=False)["word"]["token_class"]

path_config_json = "/path/to/dir/Experiments/FRFPE/layoutxlm/checkpoint-1600/config.json"
path_weights = "/path/to/dir/Experiments/FRFPE/layoutxlm/checkpoint-1600/model.safetensors"

layoutlm_classifier = dd.HFLayoutLmv2TokenClassifier(path_config_json,
                                                     path_weights,
                                                     categories=categories,
                                                     use_xlm_tokenizer=True)

tokenizer_fast = XLMRobertaTokenizerFast.from_pretrained("microsoft/layoutxlm-base")
tokenizer_fast.model_max_length=512 # Instantiating the tokenizer the way we do above seems to be problematic as 
                                    # no max_length is provided and the tokenizer therefore does not truncate the
                                    # sequence. We therefore have to set this value manually.

pipe_component = dd.LMTokenClassifierService(tokenizer_fast,
                                             layoutlm_classifier,
                                             use_other_as_default_category=True)

evaluator = dd.Evaluator(merge, pipe_component, metric)
_ = evaluator.run(split="test")
```

    The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
    The tokenizer class you load from this checkpoint is 'LayoutXLMTokenizer'. 
    The class this function is called from is 'XLMRobertaTokenizerFast'.
    [32m[0608 16:58.17 @eval.py:113][0m  [32mINF[0m  [97mBuilding multi threading pipeline component to increase prediction throughput. Using 2 threads[0m
    [32m[0608 16:58.20 @eval.py:225][0m  [32mINF[0m  [97mPredicting objects...[0m
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:01<00:00, 18.60it/s]
    [32m[0608 16:58.22 @eval.py:207][0m  [32mINF[0m  [97mStarting evaluation...[0m
    [32m[0608 16:58.22 @accmetric.py:373][0m  [32mINF[0m  [97mF1 results:
     [36m|     key     | category_id   | val      | num_samples   |
    |:-----------:|:--------------|:---------|:--------------|
    |    word     | 1             | 1        | 1505          |
    | token_class | 1             | 0.949721 | 89            |
    | token_class | 2             | 0.930556 | 69            |
    | token_class | 3             | 0.911111 | 86            |
    | token_class | 4             | 0.960825 | 490           |
    | token_class | 5             | 0.970722 | 771           |[0m[0m



```python
wandb.finish()
```


```python
evaluator.compare(interactive=True, split="test", show_words=True)
```

## Training XLM models on separate languages

Of course, there are various experimentation options here as well. For example, one could investigate whether one gets better results when training XLM models for each language separately. In our case, one could train one each on English and German data (there are too few data points for a French model). 

For this, one would have to filter the data set once for English and German data points. E.g. training a german model would look like this:

```
df = ner.dataflow.build(load_image=True, filter_languages=[dd.Languages.german])

merge = dd.MergeDataset(ner)
merge.explicit_dataflows(df)
merge.buffer_datasets()
merge.create_split_by_id(split_dict)

path_config_json = dd.ModelCatalog.get_full_path_configs("microsoft/layoutxlm-base/pytorch_model.bin")
path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutxlm-base/pytorch_model.bin")

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
                                       "wandb_project=FRFPE_layoutxlm"],
                         log_dir="/path/to/dir/Experiments/FRFPE/layoutxlm",
                         dataset_val=merge,
                         metric=metric,
                         use_xlm_tokenizer=True, 
                         use_token_tag=False,
                         pipeline_component_name="LMTokenClassifierService")
```
