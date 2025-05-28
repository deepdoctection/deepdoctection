<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  A Document AI Package
  </h3>
</p>

**deep**doctection is a Python library that orchestrates document extraction and document layout analysis tasks using
deep learning models. It does not implement models but enables you to build pipelines using highly acknowledged libraries
for object detection, OCR and selected NLP tasks and provides an integrated framework for fine-tuning, evaluating and running models. For more
specific text processing tasks use one of the many other great NLP libraries.

With the rise of LLMs and the growing adoption of Retrieval-Augmented Generation (RAG), the challenges of parsing PDFs
have gained wider attention among developers. Over the past few years, the number of available parsers has grown
significantly. Development on deepdoctection began in 2021, making it one of the first libraries to enable document
parsing using deep learning methods.

**deep**doctection focuses on applications and is made for those who want to solve real world problems related to
document extraction from PDFs or scans in various image formats.

Check the demo of a document layout analysis pipeline with OCR on 🤗
[**Hugging Face spaces**](https://huggingface.co/spaces/deepdoctection/deepdoctection).

# Overview

**deep**doctection provides model wrappers of supported libraries for various tasks to be integrated into
pipelines. Its core function does not depend on any specific deep learning library. Selected models for the following
tasks are currently supported:

- Document layout analysis including table recognition in PyTorch with [**Detectron2**](https://github.com/facebookresearch/detectron2/tree/main/detectron2) and [**Transformers**](https://github.com/huggingface/transformers)
  or Tensorflow with [**Tensorpack**](https://github.com/tensorpack),
- OCR with support of [**Tesseract**](https://github.com/tesseract-ocr/tesseract), [**DocTr**](https://github.com/mindee/doctr) and the commerical solution
  [**AWS Textract**](https://aws.amazon.com/textract/),
- Document and token classification with the [**LayoutLM**](https://github.com/microsoft/unilm) family,
  [**LiLT**](https://github.com/jpWang/LiLT) and selected
  [**Bert**](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)-style models provided by
  [**Transformers**](https://github.com/huggingface/transformers), including features like sliding windows.
- Table detection and table structure recognition with [**table-transformer**](https://github.com/microsoft/table-transformer).
- Text mining for native PDFs with [**pdfplumber**](https://github.com/jsvine/pdfplumber),
- Language detection with [**fastText**](https://github.com/facebookresearch/fastText),
- Deskewing and rotating images with [**jdeskew**](https://github.com/phamquiluan/jdeskew).
- Lot's of [tutorials](https://github.com/deepdoctection/notebooks)

**deep**doctection provides on top of that methods for pre-processing inputs to models like cropping or resizing and to
post-process results, like validating duplicate outputs, relating words to detected layout segments or ordering words
into contiguous text. You will get an output in JSON format that you can customize even further by yourself.

Have a look at the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb) in the [notebook repo](https://github.com/deepdoctection/notebooks) for an easy start.

Check the [**release notes**](https://github.com/deepdoctection/deepdoctection/releases) for recent updates.

## Models

deepdoctection maintains a model registry that catalogs deployable models. Most models are available via the
[**Hugging Face Model Hub**](https://huggingface.co/deepdoctection).

## Datasets and training scripts

Training is a key component when building pipelines tailored to specific domains—whether for document layout analysis,
classification, or named entity recognition (NER). While some commercial providers or Open Source solutions may already
offer strong performance on certain document types, there is (still) no silver bullet. **deep**doctection provides
training scripts for supported models, using the training interfaces of the underlying model libraries. It also
includes code for working with well-established datasets such as **DocLayNet**, enabling quick experimentation. Data
format interoperability is a central feature: the library includes format mappings (e.g., from COCO) and offers a
lightweight dataset framework—similar in spirit to datasets—to streamline training on custom datasets.

See [**this notebook**](https://github.com/deepdoctection/notebooks/blob/main/Datasets_and_Eval.ipynb) for a step-by-step example of how to prepare and evaluate datasets within the
**deep**doctection ecosystem.

## Evaluation

**deep**doctection comes equipped with a framework that allows you to evaluate predictions of a single or multiple
models in a pipeline against some ground truth. Check again [**here**](https://github.com/deepdoctection/notebooks/blob/main/Datasets_and_Eval.ipynb) how it is
done.

## Parsing

Having set up a pipeline it takes you a few lines of code to instantiate the pipeline and after a for loop all pages will
be processed through the pipeline.

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

![text](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_rm_sample.png)

```
HTML(page.tables[0].html)
```

<table><tr><td>(absolute Rendite-Pertomancebelag</td><td>zum Teilfonds)</td></tr><tr><td>BE Semi 2024</td><td>-26,56% /-0,57%</td></tr><tr><td>Bayer (in Covestro)</td><td>-18,03% /-0,49%</td></tr><tr><td>Fresenius Medical Care 2020</td><td>-17,87% /-0,25%</td></tr><tr><td>Implenia 2022</td><td>-17,27% /-0,16%</td></tr><tr><td>STMicroelectronics 2024</td><td>-14,64% /-0,20%</td></tr></table>

```
print(page.text)
```

```
Gewinner/Verlierer und deren Performancebeitrage
Die Gewinnerliste des letzten Jahres fuhrt die LVMH-Wandelanlelne an, welche uber das Gesamtjahr lediglich um +9,90% zulegte jedoch mit +0,84% den grossten Performancebeitrag lieferte. Nach den ersten Anzeichen einer wirtschaflichen Verlangsamung halbierten wir die LVMH-Position und brachten einen Teil des Gewinns ins Trockene. Das gleiche Vorgehen wendeten wir auch bei der MTU Aero- Wandelanleihe (+0,4296/+8,279) an und konnten SO Gewinne unter dem Jahr sichern. In beiden Fâllen konnten wir durch aktives Verkaufen von Gewinnpositionen einen Mehrwert fur die Investoren generieren.
Die Verliererliste fuhren Zyklische Sektoren wie die Halbleiter- und Chemie-Sektoren an. Nach sehr dynamischer Kursentwicklung 2017 kam es im Halbleiter-Sektor letztes Jahr zu einer grossen Konsolidierung, wobei die Wandelanleihe der belgischen BE Semi mehr als -25% ihres Wertes einbusste und eine Negativperformance von -0,57% beisteuerte. Die Aktie von BE Semi verlor im gleichen Zeitraum seit dem Hôchststand uber 50%. Unter den deutlich nach unten angepassten Wachstumsprognosen litt vor allem der Chemie-Sektor. Im Portfolio befanden sich die Bayer 2020 (wandelbar in Covestro) und BASF 2023. Die Covestro-Aktie verlor seit dem Hôchststand im Januar 2018 beinahe die Hâlfte ihres Wertes; die Wandelanleihe gab im gleichen Zeitraum -18,03% nach und steuerte -0,49% zur Fonds-Performance bei.


Funf beste Wandelanleihen im Portfolio'
(absolute Rendite-Petomancebelrag Zum Teilfonds)
Performance bzw. Attribution: 01.01.2018 bis 31.12.2018, Quelle: Bloomberg PORT
```

## Documentation

There is an extensive [**documentation**](https://deepdoctection.readthedocs.io/en/latest/index.html#) available containing [**tutorials**](https://deepdoctection.readthedocs.io/en/latest/tutorials/get_started_notebook/), design concepts and the API. We want to
present things as comprehensively and understandably as possible. However, we are aware that there are still many areas
where significant improvements can be made in terms of clarity, grammar and correctness. We look forward to every hint
and comment that increases the quality of the documentation.

## Requirements

![requirements](https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/requirements_deepdoctection_220525.png)

Everything in the overview listed below the **deep**doctection layer are necessary requirements and have to be installed
separately.

- Linux or macOS. Windows is not supported but there is a [Dockerfile](./docker/pytorch-cpu-jupyter/Dockerfile) available.
- Python >= 3.9
- 1.13 \<= PyTorch **or** 2.11 \<= Tensorflow < 2.16. (For lower Tensorflow versions the code will only run on a GPU).
  Tensorflow support will be stopped from Python 3.11 onwards.
- To fine-tune models, a GPU is recommended.

The following overview shows the availability of the models in conjunction with the DL framework.

| Task | PyTorch | Torchscript | Tensorflow |
|---------------------------------------------|:-------:|----------------|:------------:|
| Layout detection via Detectron2/Tensorpack | ✅ | ✅ (CPU only) | ✅ (GPU only) |
| Table recognition via Detectron2/Tensorpack | ✅ | ✅ (CPU only) | ✅ (GPU only) |
| Table transformer via Transformers | ✅ | ❌ | ❌ |
| Deformable-Detr | ✅ | ❌ | ❌ |
| DocTr | ✅ | ❌ | ✅ |
| LayoutLM (v1, v2, v3, XLM) via Transformers | ✅ | ❌ | ❌ |

## Installation

We recommend using a virtual environment.

#### Get started installation

**Deep**doctection, as an integration project, requires different libraries depending on the setup.

For a simple setup which is enough to parse documents with the default setting, install the following:

```
pip install transformers
pip install python-doctr
pip install deepdoctection
```

For a TensorFlow-based setup, install:

```
pip install tensorpack
pip install python-doctr
pip install deepdoctection
```

Both setups are sufficient to run the [**introduction notebook**](https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb).

#### Full installation

The following installation will give you ALL models available within the Deep Learning framework as well as all models
that are independent of Tensorflow/PyTorch.

For **Tensorflow**, run

```
pip install deepdoctection[tf]
```

For **PyTorch**,

first install **Detectron2** separately as it is not distributed via PyPi. Check the instruction
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) or try:

```
pip install detectron2@git+https://github.com/deepdoctection/detectron2.git
```

Then install **deep**doctection with all its dependencies:

```
pip install deepdoctection[pt]
```

For further information, please consult the [**full installation instructions**](https://deepdoctection.readthedocs.io/en/latest/install/).

### Installation from source

Download the repository or clone via

```
git clone https://github.com/deepdoctection/deepdoctection.git
```

To get started with **Tensorflow**, run:

```
cd deepdoctection
pip install ".[tf]" # or "pip install -e .[tf]"
```

Installing the full **PyTorch** setup from source will also install **Detectron2** for you:

```
cd deepdoctection
pip install ".[source-pt]" # or "pip install -e .[source-pt]"
```

### Running a Docker container from Docker hub

Starting from release `v.0.27.0`, pre-existing Docker images can be downloaded from the
[Docker hub](https://hub.docker.com/r/deepdoctection/deepdoctection).

```
docker pull deepdoctection/deepdoctection:<release_tag> 
```

To start the container, you can use the Docker compose file `./docker/pytorch-gpu/docker-compose.yaml`.
In the `.env` file provided, specify the host directory where **deep**doctection's cache should be stored.
This directory will be mounted. Additionally, specify a working directory to mount files to be processed into the
container.

```
docker compose up -d
```

will start the container. There is no endpoint exposed, though.

## Credits

We thank all libraries that provide high quality code and pre-trained models. Without, it would have been impossible
to develop this framework.

## Problems

We try hard to eliminate bugs. We also know that the code is not free of issues. We welcome all issues relevant to this
repo and try to address them as quickly as possible.

## If you like **deep**doctection ...

...you can easily support the project by making it more visible. Leaving a star or a recommendation will help.

## License

Distributed under the Apache 2.0 License. Check [LICENSE](https://github.com/deepdoctection/deepdoctection/blob/master/LICENSE) for additional information.
