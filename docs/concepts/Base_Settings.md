# deepdoctection setup

**deep**doctection has various setup options. We are going to cover the most important here.


## Current setup in local environment

If you are looking for something more specific than `pip list`, use 

```python
import deepdoctection as dd

print(collect_env_info())
```

It will show you what main third party dependencies have been installed, and what PyTorch setup is available.


## Environment variables

**deep**doctection uses Pydantic Settings Management and its [Dotenv (.env) support](https://docs.pydantic.dev/latest/concepts/pydantic_settings/#dotenv-env-support). Save the `.env` file
with your variables in your workdir.

- `LOG_LEVEL` (str)
    Global logging level (default: `INFO`). Use `DEBUG` for a verbose logging.

- `LOG_PROPAGATE` (bool)
    Whether log records should propagate to parent loggers (default: `False`).

- `FILTER_THIRD_PARTY_LIB` (bool)
    Whether to filter third-party library output in logs (default: `False`).

- `STD_OUT_VERBOSE` (bool)
    Increase verbosity for stdout output (default: `False`). Especially useful to
    understand, if the `DatapointManager` has issues to add `Annotation`s to an `Image`.

- `HF_CREDENTIALS` (Secret / sensitive)
    Hugging Face credentials used by the `ModelDownloadManager`. Treated as a secret.

- `MODEL_CATALOG` (str / path)
    Optional path to a `.jsonl` model catalog (default: `None`). This will add your (custom)
    models to the `ModelCatalog`. Do not confuse with `MODEL_CATALOG_BASE`.

- `USE_DD_PILLOW` (bool)
    Prefer Pillow for viz handling (default: `True` unless OpenCV detected). Pillow will be installed by default

- `USE_DD_OPENCV` (bool)
    Prefer OpenCV for viz handling (default: `True` when OpenCV is available).
    Note, that OpenCV will have to be installed independently.

- `USE_DD_PDFIUM` (bool)
    Prefer PyPDFium2 for PDF rendering (default: auto-detected when available). This is also the default choice.

- `USE_DD_POPPLER` (bool)
    Prefer Poppler-based rendering (legacy option; used if pdfium not available). Note, that Poppler wheels cannot
    by installed by any Python package and have to be installed separately.

- `DPI` (int)
    Default DPI used for rendering (default: `300`). The default setting is very high and for historical
    reasons we will be keeping this value, even though we recommend 200

- `IMAGE_WIDTH` (int)
    Optional default image width (default: `0`). Only relevant when rendering with Poppler.

- `IMAGE_HEIGHT` (int)
    Optional default image height (default: `0`). Only relevant when rendering with Poppler.

- `DD_ONE_CONFIG` (path)
    Path to the `dd_analyzer` configuration in the configs directory (exported for compatibility).

- `DEEPDOCTECTION_CACHE` (path)
    Root cache directory for deepdoctection (default: `~/.cache/deepdoctection`).

- `MODEL_DIR` (path)
    Directory for downloaded model weights (default: `DEEPDOCTECTION_CACHE/weights`).

- `CONFIGS_DIR` (path)
    Directory for config files (default: `DEEPDOCTECTION_CACHE/configs`).

- `DATASET_DIR` (path)
    Directory for datasets (default: `DEEPDOCTECTION_CACHE/datasets`).


## Cache directory

A short reference for deepdoctection’s cache layout, how models are downloaded/saved, and to install datasets. All
paths below are relative to the cache root (default: ~/.cache/deepdoctection) unless overridden by
`DEEPDOCTECTION_CACHE`.

!!! info "Cache directory structure"

    - `weights/` (also exposed via `MODEL_DIR`) — stored model weights, per-model directories with weights only.

    - `configs/` (also exposed via `CONFIGS_DIR`) — configuration files.

    - `datasets/` (also exposed via `DATASET_DIR`) — downloaded or prepared dataset folders.


### How models are downloaded and saved

The model download manager writes into MODEL_DIR (by default DEEPDOCTECTION_CACHE/weights).
Authentication (when required) uses the `HF_CREDENTIALS` environment variable for Hugging Face downloads.

Where models are downloaded from and where they are cached is encoded in the `ModelCatalog`.

```python
profile = dd.ModelCatalog.get_profile("deepdoctection/tatr_tab_struct_v2/model.safetensors")
```

- `profile.name`: determines the target directory and artifact name relative to  `MODEL_DIR`.

- `profile.config`: determines the target directory of any model config relative to `CONFIGS_DIR`.

-  Same goes for `profile.preprocessor_config`.

If models are remote available on the Huggingface Hub:

- `profile.hf_repo_id`: determines the HF repo_id

- `profile.hf_model_name`: determines the remote model artifact name

- `profile.hf_config_file`: determines the remote model config filename. Use lists if the config consists of
                            different files: `['config.json', 'preprocessor_config.json']`.


Some models need to be downloaded from elsewhere. 

- `profile.urls`: determines the url of the remote model. Use lists if models consists of different files.


```python
profile.hf_repo_id, profile.hf_model_name, profile.hf_config_file
```


```python
import deepdoctection as dd

dd.ModelCatalog.get_profile("deepdoctection/tatr_tab_struct_v2/model.safetensors")
```


??? info "Output"

    ```python
    {'name': 'deepdoctection/tatr_tab_struct_v2/model.safetensors',
     'description': 'Table Transformer (DETR) model trained on PubTables1M. It was introduced in the paper Aligning
                     benchmark datasets for table structure recognition by Smock et al. This model is devoted to table
                     structure recognition and assumes to receive a slightly croppedtable as input. It will predict
                     rows, column and spanning cells. Use a padding of around 5 pixels. This artefact has been converted
                     from deepdoctection/tatr_tab_struct_v2/pytorch_model.bin and should be used to reduce security
                     issues',
     'size': [115511753],
     'config': 'deepdoctection/tatr_tab_struct_v2/config.json',
     'preprocessor_config': 'deepdoctection/tatr_tab_struct_v2/preprocessor_config.json',
     'hf_repo_id': 'deepdoctection/tatr_tab_struct_v2',
     'hf_model_name': 'model.safetensors',
     'hf_config_file': ['config.json', 'preprocessor_config.json'],
     'urls': None,
     'categories': {1: <LayoutType.TABLE>,
      2: <LayoutType.COLUMN>,
      3: <LayoutType.ROW>,
      4: <CellType.COLUMN_HEADER>,
      5: <CellType.PROJECTED_ROW_HEADER>,
      6: <CellType.SPANNING>},
     'categories_orig': None,
     'dl_library': 'PT',
     'model_wrapper': 'HFDetrDerivedDetector',
     'architecture': None,
     'padding': None}
    ```



### How datasets need to be saved

There is no download mechanism for datasets. Each dataset is stored under datasets/<dataset_name>/. For each built-in datasets (`DocLayNet`, `Funsd`, `RVLCDIP`) you can find the directory structure in the docs.

```python
├── datasets/
│   └── DocLayNet_core/
│       ├── COCO/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── test.json
│       └── PNG/
│           ├── 0a0d43e3...a2.png
│           └── ...
```


