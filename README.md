<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
</p>

![GitHub Repo stars](https://img.shields.io/github/stars/deepdoctection/deepdoctection)
![PyPI - Version](https://img.shields.io/pypi/v/deepdoctection)
![PyPI - License](https://img.shields.io/pypi/l/deepdoctection)


------------------------------------------------------------------------------------------------------------------------
# NEW 

Version `v.0.43` includes a significant redesign of the Analyzer's default configuration.  Key changes include:

* More powerful models for Document Layout Analysis and OCR.
* Expanded functionality.
* Less dependencies.

------------------------------------------------------------------------------------------------------------------------

<p align="center">
  <h1 align="center">
  A Package for Document Understanding
  </h1>
</p>


**deep**doctection is a Python library that orchestrates Scan and PDF document layout analysis and extraction for RAG.
It also provides a framework for training, evaluating and inferencing Document AI models.

# Overview

- Document layout analysis and table recognition in PyTorch with 
[**Detectron2**](https://github.com/facebookresearch/detectron2/tree/main/detectron2) and 
[**Transformers**](https://github.com/huggingface/transformers)
  or Tensorflow and [**Tensorpack**](https://github.com/tensorpack),
- OCR with support of [**Tesseract**](https://github.com/tesseract-ocr/tesseract), [**DocTr**](https://github.com/mindee/doctr) and 
  [**AWS Textract**](https://aws.amazon.com/textract/),
- Document and token classification with the [**LayoutLM**](https://github.com/microsoft/unilm) family,
  [**LiLT**](https://github.com/jpWang/LiLT) and selected
  [**Bert**](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)-style including features like sliding windows.
- Text mining for native PDFs with [**pdfplumber**](https://github.com/jsvine/pdfplumber),
- Language detection with `papluca/xlm-roberta-base-language-detection`. [**fastText**](https://github.com/facebookresearch/fastText) is still available but
  but will be removed in a future version.
- Deskewing and rotating images with [**jdeskew**](https://github.com/phamquiluan/jdeskew).
- Fine-tuning and evaluation tools.
- Lot's of [tutorials](https://github.com/deepdoctection/notebooks)

Have a look at the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Analyzer_Get_Started.ipynb)
for an easy start.

Check the [**release notes**](https://github.com/deepdoctection/deepdoctection/releases) for recent updates.


----------------------------------------------------------------------------------------

# Hugging Face Space Demo

Check the demo of a document layout analysis pipeline with OCR on 🤗
[**Hugging Face spaces**](https://huggingface.co/spaces/deepdoctection/deepdoctection) or use the gradio client. 

```
pip install gradio_client   # requires Python >= 3.10 
```

To process a single image:

```python
from gradio_client import Client, handle_file

if __name__ == "__main__":

    client = Client("deepdoctection/deepdoctection")
    result = client.predict(
        img=handle_file('/local_path/to/dir/file_name.jpeg'),  # accepts image files, e.g. JPEG, PNG
        pdf=None,   
        max_datapoints = 2,
        api_name = "/analyze_image"
    )
    print(result)
```

To process a PDF document:

```python
from gradio_client import Client, handle_file

if __name__ == "__main__":

    client = Client("deepdoctection/deepdoctection")
    result = client.predict(
        img=None,
        pdf=handle_file("/local_path/to/dir/your_doc.pdf"),
        max_datapoints = 2, # increase to process up to 9 pages
        api_name = "/analyze_image"
    )
    print(result)
```

--------------------------------------------------------------------------------------------------------

# Example

```python
import deepdoctection as dd
from IPython.core.display import HTML
from matplotlib import pyplot as plt

analyzer = dd.get_dd_analyzer()  # instantiate the built-in analyzer similar to the Hugging Face space demo

df = analyzer.analyze(path = "/path/to/your/doc.pdf")  # setting up pipeline
df.reset_state()                 # Trigger some initialization

doc = iter(df)
page = next(doc) 

image = page.viz(show_figures=True, show_residual_layouts=True)
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
```

<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_sample.png" 
alt="sample" width="40%">
</p>

```
HTML(page.tables[0].html)
```

<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_table.png" 
alt="table" width="40%">
</p>

```
print(page.text)
```

<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_text.png" 
alt="text" width="40%">
</p>


-----------------------------------------------------------------------------------------

# Requirements

![requirements](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/install_01.png)

- Linux or macOS. Windows is not supported but there is a [Dockerfile](./docker/pytorch-cpu-jupyter/Dockerfile) available.
- Python >= 3.9
- 2.6 \<= PyTorch **or** 2.11 \<= Tensorflow < 2.16. (For lower Tensorflow versions the code will only run on a GPU).
  Tensorflow support will be stopped from Python 3.11 onwards.
- To fine-tune models, a GPU is recommended.

| Task | PyTorch | Torchscript | Tensorflow |
|---------------------------------------------|:-------:|----------------|:------------:|
| Layout detection via Detectron2/Tensorpack | ✅ | ✅ (CPU only) | ✅ (GPU only) |
| Table recognition via Detectron2/Tensorpack | ✅ | ✅ (CPU only) | ✅ (GPU only) |
| Table transformer via Transformers | ✅ | ❌ | ❌ |
| Deformable-Detr | ✅ | ❌ | ❌ |
| DocTr | ✅ | ❌ | ✅ |
| LayoutLM (v1, v2, v3, XLM) via Transformers | ✅ | ❌ | ❌ |

------------------------------------------------------------------------------------------

# Installation

We recommend using a virtual environment.

## Get started installation

For a simple setup which is enough to parse documents with the default setting, install the following:

**PyTorch**

```
pip install transformers
pip install python-doctr==0.10.0 # If you use Python 3.10 or higher you can use the latest version.
pip install deepdoctection
```

**TensorFlow**

```
pip install tensorpack
pip install deepdoctection
pip install "numpy>=1.21,<2.0" --upgrade --force-reinstall  # because TF 2.11 does not support numpy 2.0 
pip install "python-doctr==0.9.0"
```

Both setups are sufficient to run the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb).

### Full installation

The following installation will give you ALL models available within the Deep Learning framework as well as all models
that are independent of Tensorflow/PyTorch.

**PyTorch**

First install **Detectron2** separately as it is not distributed via PyPi. Check the instruction
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) or try:

```
pip install detectron2@git+https://github.com/deepdoctection/detectron2.git
```

Then install **deep**doctection with all its dependencies:

```
pip install deepdoctection[pt]
```

**Tensorflow**

```
pip install deepdoctection[tf]
```


For further information, please consult the [**full installation instructions**](https://deepdoctection.readthedocs.io/en/latest/install/).


## Installation from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

**PyTorch**

```
cd deepdoctection
pip install ".[pt]" # or "pip install -e .[pt]"
```

**Tensorflow**

```
cd deepdoctection
pip install ".[tf]" # or "pip install -e .[tf]"
```


## Running a Docker container from Docker hub

Pre-existing Docker images can be downloaded from the [Docker hub](https://hub.docker.com/r/deepdoctection/deepdoctection).

```
docker pull deepdoctection/deepdoctection:<release_tag> 
```

Use the Docker compose file `./docker/pytorch-gpu/docker-compose.yaml`.
In the `.env` file provided, specify the host directory where **deep**doctection's cache should be stored.
Additionally, specify a working directory to mount files to be processed into the container.

```
docker compose up -d
```

will start the container. There is no endpoint exposed, though.

-----------------------------------------------------------------------------------------------

# Credits

We thank all libraries that provide high quality code and pre-trained models. Without, it would have been impossible
to develop this framework.


# If you like **deep**doctection ...

...you can easily support the project by making it more visible. Leaving a star or a recommendation will help.

# License

Distributed under the Apache 2.0 License. Check [LICENSE](https://github.com/deepdoctection/deepdoctection/blob/master/LICENSE) for additional information.
