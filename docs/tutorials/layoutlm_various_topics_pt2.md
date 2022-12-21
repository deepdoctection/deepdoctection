# Various topics around LayoutLM part 2

## Adding a CNN backbone for fine-tuning

In a fine-tuning experiment of the original [LayoutLM](https://arxiv.org/pdf/1912.13318.pdf), a CNN backbone was
added for additional features. The ResNet-101 was pre-trained on the
Visual-Genome Dataset, a dataset with real-world images.

The aim here is to examine the extent to which F1 results differ when a
backbone that is pre-trained on a document layout tasks.

Here we use the Detectron2 CNN backbone from the **deep**doctection
cell detector. Compared to the paper, it is a Resnext-50 backbone with
FPN features.

We also refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb)
in which the design specification is fully included.

```python

    from typing import Optional, Union, Tuple
    
    import torch
    from torch import nn
    
    from transformers import (
        LayoutLMPreTrainedModel, 
        PretrainedConfig, 
        LayoutLMModel, 
        LayoutLMTokenizerFast, 
        TrainingArguments,
    )
    from transformers.modeling_outputs import TokenClassifierOutput
    
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.structures import Boxes
    
    import deepdoctection as dd
    from deepdoctection.datasets.adapter import DatasetAdapter
    from deepdoctection.train.hf_layoutlm_train import LayoutLMTrainer
```


## Defining the LayoutLMv1 model with visual backbone


```python

    class LayoutLMWithImageFeaturesForTokenClassification(LayoutLMPreTrainedModel):
    
        def __init__(self, config, d2_config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.layoutlm = LayoutLMModel(config)
            self.d2_backbone = build_model(d2_config)
    
            # When instantiating the model for the first time we need to use pre-trained weights from 
            # different arefacts. Once checkpoints for fine-tuning has been generated all weights are in the same 
            # artefact.
            # Note however, that the Cascade-Head will be loaded as well even though it will not be used.
            if d2_config.MODEL.WEIGHTS:
                self._instantiate_d2_predictor(d2_config)
            self.projection = nn.Linear(in_features=d2_config.MODEL.FPN.OUT_CHANNELS
                                                    * d2_config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
                                                    * d2_config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
                                        out_features=config.hidden_size)
    
            # Dropout and final linear layer for classification
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
            # Initialize weights and apply final processing
            self.post_init()
    
        def get_input_embeddings(self):
            return self.layoutlm.embeddings.word_embeddings
    
        def _instantiate_d2_predictor(self, d2_config) -> None:
            checkpointer = DetectionCheckpointer(self.d2_backbone)
            checkpointer.load(d2_config.MODEL.WEIGHTS)
    
        def forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                bbox: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                images:  Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            outputs = self.layoutlm(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
            last_hidden_states = outputs.last_hidden_state
    
            # Generating image features and rois for each box
            d2_images_input = [{"image": image} for image in images]
            d2_bbox_input = [Boxes(boxes) for boxes in bbox]
    
            out = self.d2_backbone.preprocess_image(d2_images_input)
            
            # Feature map
            features = self.d2_backbone.backbone(out.tensor)
            features = [features[f] for f in self.d2_backbone.roi_heads.box_in_features]
            batch_size, seq_length = bbox.shape[0], bbox.shape[1]
            
            # RoiAlignv2
            box_features = self.d2_backbone.roi_heads.box_pooler(features, d2_bbox_input)
            box_features = box_features.view(batch_size, seq_length, -1)
            
            projected_box_features = self.projection(box_features)
            
            # Adding hidden states from layoutlm and backbone
            last_layer_input = last_hidden_states + projected_box_features
    
            last_layer_input = self.dropout(last_layer_input)
            logits = self.classifier(last_layer_input)
    
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
    
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    
        def resize_position_embeddings(self, new_num_position_embeddings: int):
            pass
    
        def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
            pass
    
        def _reorder_cache(self, past, beam_idx):
            pass
```

## Setting up training script


```python

    # Config and weights for LayoutLM
    config_path = dd.ModelCatalog.get_full_path_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")
    path_weights = dd.ModelCatalog.get_full_path_weights("microsoft/layoutlm-base-uncased/pytorch_model.bin")
        
    # Config and weights for Resnext-50 with FPN
    path_yaml = dd.ModelCatalog.get_full_path_configs("cell/d2_model-1800000-cell.pkl")
    path_d2_weights = dd.ModelCatalog.get_full_path_weights("cell/d2_model-1800000-cell.pkl")
    
    log_dir = "/path/to/dir/Vis_backbone"
    
    # Setting up dataset for fine tuning
    funsd = dd.get_dataset("funsd")
    dataset_type = funsd.dataset_info.type
    categories_dict_name_as_key = funsd.dataflow.categories.get_sub_categories(
        categories="word",
        sub_categories={"word": ["token_tag"]},
        keys=False,
        values_as_dict=True,
        name_as_key=True,
        )["word"]["token_tag"]
    
    id2label = {int(k) - 1: v for v, k in categories_dict_name_as_key.items()}
    
    
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=config_path, id2label=id2label)
    
    # additional attribute with default value, so that the true value can be loaded from the configs
    cfg = get_cfg()
    cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.1
    cfg.merge_from_file(path_yaml)
    cfg.merge_from_list(["MODEL.WEIGHTS", path_d2_weights])
    
    # Setup model
    model = LayoutLMWithImageFeaturesForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config, d2_config=cfg)
    device = torch.device("cuda")
    model.to(device)
    
    # Adapter for training PyTorch models
    dataset = DatasetAdapter(
        funsd,
        True,
        dd.image_to_raw_layoutlm_features(categories_dict_name_as_key, dataset_type),
        **{"split": "train", "load_image": True},
        )
    number_samples = len(dataset)
    
    # Training config
    conf_dict = {
        "output_dir": log_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": 2,
        "max_steps": 6000,
        "save_steps": 200,
        "evaluation_strategy": "no",
        "eval_steps": 100,
        }
    
    arguments = TrainingArguments(**conf_dict)
    tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    data_collator = dd.LayoutLMDataCollator(tokenizer_fast, return_tensors="pt")
    trainer = LayoutLMTrainer(model, arguments, data_collator, dataset)
    
    trainer.train()
```

```

    You are using a model of type layoutlm to instantiate a model of type . This is not supported for all configurations of models and can yield errors.
    Model config PretrainedConfig {
      "_name_or_path": "microsoft/layoutlm-base-uncased",
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "id2label": {
        "0": "B-answer",
        "1": "B-header",
        "2": "B-question",
        "3": "I-answer",
        "4": "I-header",
        "5": "I-question",
        "6": "O"
      },
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_2d_position_embeddings": 1024,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "output_past": true,
      "pad_token_id": 0,
      "position_embedding_type": "absolute",
      "transformers_version": "4.19.4",
      "type_vocab_size": 2,
      "use_cache": true,
      "vocab_size": 30522
    }


    [0912 11:43.24 @maputils.py:205]Ground-Truth category distribution:
    |  category  | #box   |  category  | #box   |  category  | #box   |
    |:----------:|:-------|:----------:|:-------|:----------:|:-------|
    |  B-ANSWER  | 2802   |   B-HEAD   | 441    | B-QUESTION | 3266   |
    |  I-ANSWER  | 6924   |   I-HEAD   | 1044   | I-QUESTION | 4064   |
    |     O      | 3971   |            |        |            |        |
    |   total    | 22512  |            |        |            |        |
    [0912 11:43.24 @custom.py:133][0m Make sure to call .reset_state() for the dataflow otherwise an error will be raised
```

## Setting up evaluation


In order to pass the model to a pipeline component and hence to the
evaluator, we first have to provide a model wrapper

```python

    from copy import copy
    
    from typing import Sequence, Mapping,  Literal, List

    class HFLayoutLmWithImageFeaturesTokenClassifier(dd.HFLayoutLmTokenClassifier):
    
        def __init__(
            self,
            name: str,
            path_config_json: str,
            path_d2_yaml: str,
            path_weights: str,
            categories_semantics: Optional[Sequence[str]] = None,
            categories_bio: Optional[Sequence[str]] = None,
            categories: Optional[Mapping[str, str]] = None,
            device: Optional[Literal["cpu", "cuda"]] = None,
        ):
            if categories is None:
                assert categories_semantics is not None
                assert categories_bio is not None

            self.name = name
            self.path_config = path_config_json
            self.path_d2_yaml = path_d2_yaml
            self.path_weights = path_weights
            self.categories_semantics = categories_semantics
            self.categories_bio = categories_bio
            if categories:
                self.categories = copy(categories)
            else:
                self.categories = self._categories_orig_to_categories(categories_semantics, categories_bio)  # type: ignore
    
            config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json)
            d2_config = get_cfg()
            d2_config.merge_from_file(self.path_d2_yaml)
    
            self.model = LayoutLMWithImageFeaturesForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config, d2_config=d2_config)
    
            if device is not None:
                self.device = device
            else:
                self.device = dd.set_torch_auto_device()
            self.model.to(self.device)
    
        def predict(self, **encodings: Union[List[List[str]], "torch.Tensor"]) -> List[dd.TokenClassResult]:
            """
            Launch inference on LayoutLm for token classification. Pass the following arguments
    
            :param input_ids: Token converted to ids to be taken from LayoutLMTokenizer
            :param attention_mask: The associated attention masks from padded sequences taken from LayoutLMTokenizer
            :param token_type_ids: Torch tensor of token type ids taken from LayoutLMTokenizer
            :param boxes: Torch tensor of bounding boxes of type 'xyxy'
            :param tokens: List of original tokens taken from LayoutLMTokenizer
    
            :return: A list of TokenClassResults
            """
    
            ann_ids = encodings.get("ann_ids")
            input_ids = encodings.get("input_ids")
            attention_mask = encodings.get("attention_mask")
            token_type_ids = encodings.get("token_type_ids")
            boxes = encodings.get("bbox")
            tokens = encodings.get("tokens")
            images = encodings.get("images")
    
            assert isinstance(ann_ids, list)
            if len(ann_ids) > 1:
                raise ValueError("HFLayoutLmTokenClassifier accepts for inference only batch size of 1")
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(attention_mask, torch.Tensor)
            assert isinstance(token_type_ids, torch.Tensor)
            assert isinstance(boxes, torch.Tensor)
            assert isinstance(tokens, list)
            if images is not None:
                assert isinstance(images, list)
                images = [img.to(self.device) for img in images]
    
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            boxes = boxes.to(self.device)
    
            results = dd.predict_token_classes(
                ann_ids[0],
                input_ids,
                attention_mask,
                token_type_ids,
                boxes,
                tokens[0],
                self.model,
                images
            )
    
            return self._map_category_names(results)
    
        @classmethod
        def get_requirements(cls) -> List[dd.Requirement]:
            return [dd.get_pytorch_requirement(), dd.get_transformers_requirement(), dd.get_detectron2_requirement()]
    
        def clone(self) -> "HFLayoutLmWithImageFeaturesTokenClassifier":
            return self.__class__(
                self.name,
                self.path_config,
                self.path_d2_yaml,
                self.path_weights,
                self.categories_semantics,
                self.categories_bio,
                self.categories,
            )
```

```python

    def mean_f1_score(f1_per_label):
        total = 0.
        sum = 0.
        for res in f1_per_label:
            total+=res["val"]*res["num_samples"]
            sum+=res["num_samples"]
        
        return total/sum
```

```python

    config = "/path/to/dir/Vis_backbone/checkpoint-200/config.json"
    
    
    path_yaml = dd.ModelCatalog.get_full_path_configs("cell/d2_model-1800000-cell.pkl")
    tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    
    for checkp in range(1,30):
        weights = f"/path/to/dir/Vis_backbone/checkpoint-{200*checkp}/pytorch_model.bin"
        print(weights)
        dataset_val = dd.get_dataset("funsd")
    
        categories = dataset_val.dataflow.categories.get_sub_categories(
            categories="token_tag", sub_categories={"word": ["token_tag"]}, keys=False
        )["word"]["token_tag"]
    
        metric = dd.get_metric("f1")
        metric.set_categories(sub_category_names={"word": ["token_tag"]})
        #language_model = dd.HFLayoutLmTokenClassifier(config,weights,categories=categories)
        language_model = HFLayoutLmWithImageFeaturesTokenClassifier(config, path_yaml, weights,
                                                                    categories=categories)
        pipeline_component = dd.LMTokenClassifierService(tokenizer_fast, language_model, dd.image_to_layoutlm_features,
                                                         True)
        evaluator = dd.Evaluator(dataset_val, pipeline_component, metric, num_threads=2)
        f1_per_label = evaluator.run()
        print(f"mean f1 score: {mean_f1_score(f1_per_label)}")
```

```

   /path/to/dir/Vis_backbone/checkpoint-200/pytorch_model.bin

    [0912 12:59.44 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.830211 | 821           |
    | token_tag | 2             | 0.460674 | 122           |
    | token_tag | 3             | 0.871058 | 1077          |
    | token_tag | 4             | 0.815841 | 2544          |
    | token_tag | 5             | 0.52505  | 257           |
    | token_tag | 6             | 0.755478 | 1594          |
    | token_tag | 7             | 0.759437 | 2558          |


    mean f1 score: 0.7838230791255405


   /path/to/dir/Vis_backbone/checkpoint-400/pytorch_model.bin

    [0912 12:59.54 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.848272 | 821           |
    | token_tag | 2             | 0.592058 | 122           |
    | token_tag | 3             | 0.881132 | 1077          |
    | token_tag | 4             | 0.830082 | 2544          |
    | token_tag | 5             | 0.516229 | 257           |
    | token_tag | 6             | 0.76908  | 1594          |
    | token_tag | 7             | 0.755625 | 2558          |


    mean f1 score: 0.7935855779424233


    /path/to/dir/Vis_backbone/checkpoint-600/pytorch_model.bin

    [0912 13:00.05 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.845103 | 821           |
    | token_tag | 2             | 0.579592 | 122           |
    | token_tag | 3             | 0.87699  | 1077          |
    | token_tag | 4             | 0.812662 | 2544          |
    | token_tag | 5             | 0.560403 | 257           |
    | token_tag | 6             | 0.7667   | 1594          |
    | token_tag | 7             | 0.740614 | 2558          |

    mean f1 score: 0.78425321409946

.. parsed-literal::
   /path/to/dir/Vis_backbone/checkpoint-800/pytorch_model.bin

    [0912 13:00.17 @accmetric.py:346] F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.845203 | 821           |
    | token_tag | 2             | 0.625    | 122           |
    | token_tag | 3             | 0.863345 | 1077          |
    | token_tag | 4             | 0.81896  | 2544          |
    | token_tag | 5             | 0.575139 | 257           |
    | token_tag | 6             | 0.767656 | 1594          |
    | token_tag | 7             | 0.759337 | 2558          |

    mean f1 score: 0.7909570408298823


    /path/to/dir/Vis_backbone/checkpoint-1000/pytorch_model.bin

    [0912 13:00.28 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.856803 | 821           |
    | token_tag | 2             | 0.622407 | 122           |
    | token_tag | 3             | 0.878758 | 1077          |
    | token_tag | 4             | 0.824247 | 2544          |
    | token_tag | 5             | 0.548951 | 257           |
    | token_tag | 6             | 0.780972 | 1594          |
    | token_tag | 7             | 0.758919 | 2558          |

    mean f1 score: 0.7968284393763414


    /path/to/dir/Vis_backbone/checkpoint-1200/pytorch_model.bin

    [0912 13:00.39 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.856265 | 821           |
    | token_tag | 2             | 0.601562 | 122           |
    | token_tag | 3             | 0.875    | 1077          |
    | token_tag | 4             | 0.826446 | 2544          |
    | token_tag | 5             | 0.512077 | 257           |
    | token_tag | 6             | 0.773773 | 1594          |
    | token_tag | 7             | 0.764029 | 2558          |

    mean f1 score: 0.795789744453888


    /path/to/dir/Vis_backbone/checkpoint-1400/pytorch_model.bin

    [0912 13:00.51 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.862768 | 821           |
    | token_tag | 2             | 0.626506 | 122           |
    | token_tag | 3             | 0.871551 | 1077          |
    | token_tag | 4             | 0.8082   | 2544          |
    | token_tag | 5             | 0.521452 | 257           |
    | token_tag | 6             | 0.754504 | 1594          |
    | token_tag | 7             | 0.755716 | 2558          |

    mean f1 score: 0.7856123499985105


    /path/to/dir/Vis_backbone/checkpoint-1600/pytorch_model.bin

    [0912 13:01.01 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.862418 | 821           |
    | token_tag | 2             | 0.644351 | 122           |
    | token_tag | 3             | 0.879236 | 1077          |
    | token_tag | 4             | 0.829003 | 2544          |
    | token_tag | 5             | 0.57041  | 257           |
    | token_tag | 6             | 0.782462 | 1594          |
    | token_tag | 7             | 0.766997 | 2558          |

    mean f1 score: 0.8022285304592011


    /path/to/dir/Vis_backbone/checkpoint-1800/pytorch_model.bin

    [0912 13:01.11 @eval.py:157] Starting evaluation...
    [0912 13:01.13 @accmetric.py:346] F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.863362 | 821           |
    | token_tag | 2             | 0.609442 | 122           |
    | token_tag | 3             | 0.880717 | 1077          |
    | token_tag | 4             | 0.807097 | 2544          |
    | token_tag | 5             | 0.565836 | 257           |
    | token_tag | 6             | 0.778129 | 1594          |
    | token_tag | 7             | 0.756757 | 2558          |

    mean f1 score: 0.7919870785780183


    /path/to/dir/Vis_backbone/checkpoint-2000/pytorch_model.bin

    [0912 13:01.24 @accmetric.py:346] F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.859044 | 821           |
    | token_tag | 2             | 0.598291 | 122           |
    | token_tag | 3             | 0.880114 | 1077          |
    | token_tag | 4             | 0.818494 | 2544          |
    | token_tag | 5             | 0.561151 | 257           |
    | token_tag | 6             | 0.784038 | 1594          |
    | token_tag | 7             | 0.766191 | 2558          |

    mean f1 score: 0.798204199723473


    /path/to/dir/Vis_backbone/checkpoint-2200/pytorch_model.bin

    [0912 13:01.35 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.872072 | 821           |
    | token_tag | 2             | 0.606635 | 122           |
    | token_tag | 3             | 0.88227  | 1077          |
    | token_tag | 4             | 0.825581 | 2544          |
    | token_tag | 5             | 0.582031 | 257           |
    | token_tag | 6             | 0.790148 | 1594          |
    | token_tag | 7             | 0.775231 | 2558          |

    mean f1 score: 0.8060384049708412

.. parsed-literal::

    /path/to/dir/Vis_backbone/checkpoint-2400/pytorch_model.bin

    [0912 13:01.46 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.864702 | 821           |
    | token_tag | 2             | 0.622222 | 122           |
    | token_tag | 3             | 0.882974 | 1077          |
    | token_tag | 4             | 0.820471 | 2544          |
    | token_tag | 5             | 0.613936 | 257           |
    | token_tag | 6             | 0.774738 | 1594          |
    | token_tag | 7             | 0.771018 | 2558          |

    mean f1 score: 0.8011870771317523


    /path/to/dir/Vis_backbone/checkpoint-2600/pytorch_model.bin

    [0912 13:01.57 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.850519 | 821           |
    | token_tag | 2             | 0.639344 | 122           |
    | token_tag | 3             | 0.888574 | 1077          |
    | token_tag | 4             | 0.820766 | 2544          |
    | token_tag | 5             | 0.575188 | 257           |
    | token_tag | 6             | 0.777989 | 1594          |
    | token_tag | 7             | 0.761813 | 2558          |

    mean f1 score: 0.7977212918133954


    /path/to/dir/Vis_backbone/checkpoint-2800/pytorch_model.bin

    [0912 13:02.09 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.851942 | 821           |
    | token_tag | 2             | 0.594378 | 122           |
    | token_tag | 3             | 0.881822 | 1077          |
    | token_tag | 4             | 0.813532 | 2544          |
    | token_tag | 5             | 0.574627 | 257           |
    | token_tag | 6             | 0.781488 | 1594          |
    | token_tag | 7             | 0.765437 | 2558          |

    mean f1 score: 0.7960172594005404


    /path/to/dir/Vis_backbone/checkpoint-3000/pytorch_model.bin

    [0912 13:02.21 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.8568   | 821           |
    | token_tag | 2             | 0.597701 | 122           |
    | token_tag | 3             | 0.875969 | 1077          |
    | token_tag | 4             | 0.82385  | 2544          |
    | token_tag | 5             | 0.551724 | 257           |
    | token_tag | 6             | 0.777209 | 1594          |
    | token_tag | 7             | 0.765081 | 2558          |

    mean f1 score: 0.7972124336953419


    /path/to/dir/Vis_backbone/checkpoint-3200/pytorch_model.bin

    [0912 13:02.32 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.854277 | 821           |
    | token_tag | 2             | 0.577982 | 122           |
    | token_tag | 3             | 0.878002 | 1077          |
    | token_tag | 4             | 0.824099 | 2544          |
    | token_tag | 5             | 0.577154 | 257           |
    | token_tag | 6             | 0.782979 | 1594          |
    | token_tag | 7             | 0.756447 | 2558          |

    mean f1 score: 0.7963200667654791


    /path/to/dir/Vis_backbone/checkpoint-3400/pytorch_model.bin

    [0912 13:02.43 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.863095 | 821           |
    | token_tag | 2             | 0.581197 | 122           |
    | token_tag | 3             | 0.880189 | 1077          |
    | token_tag | 4             | 0.822634 | 2544          |
    | token_tag | 5             | 0.56391  | 257           |
    | token_tag | 6             | 0.777485 | 1594          |
    | token_tag | 7             | 0.757907 | 2558          |

    mean f1 score: 0.7960786308650052


    /path/to/dir/Vis_backbone/checkpoint-3600/pytorch_model.bin

    [0912 13:02.54 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.846291 | 821           |
    | token_tag | 2             | 0.590717 | 122           |
    | token_tag | 3             | 0.871154 | 1077          |
    | token_tag | 4             | 0.818693 | 2544          |
    | token_tag | 5             | 0.56102  | 257           |
    | token_tag | 6             | 0.774107 | 1594          |
    | token_tag | 7             | 0.75607  | 2558          |

    mean f1 score: 0.7912620719466341


    /path/to/dir/Vis_backbone/checkpoint-3800/pytorch_model.bin

    [0912 13:03.04 @eval.py:157]Starting evaluation...
    [0912 13:03.05 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.848665 | 821           |
    | token_tag | 2             | 0.575107 | 122           |
    | token_tag | 3             | 0.874046 | 1077          |
    | token_tag | 4             | 0.820722 | 2544          |
    | token_tag | 5             | 0.563071 | 257           |
    | token_tag | 6             | 0.776621 | 1594          |
    | token_tag | 7             | 0.759802 | 2558          |

    mean f1 score: 0.7937586962016544


    /path/to/dir/Vis_backbone/checkpoint-4000/pytorch_model.bin

    [0912 13:03.16 @accmetric.py:346]F1 results:
    |    key    | category_id   | val      | num_samples   |
    |:---------:|:--------------|:---------|:--------------|
    | token_tag | 1             | 0.856124 | 821           |
    | token_tag | 2             | 0.591093 | 122           |
    | token_tag | 3             | 0.879245 | 1077          |
    | token_tag | 4             | 0.822911 | 2544          |
    | token_tag | 5             | 0.558719 | 257           |
    | token_tag | 6             | 0.777032 | 1594          |
    | token_tag | 7             | 0.756508 | 2558          |

    mean f1 score: 0.7949125349027464
```

We get a top score after 2.4K iterations with a mean f1 score: 0.806
which is slightly better as the result mentioned in the paper (which was
0.7927). However the additional overhead seems to me quite immense for
such a little improvement.

