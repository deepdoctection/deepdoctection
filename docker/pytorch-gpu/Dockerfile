FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
ENV DEEPDOCTECTION_CACHE="/root/.cache/deepdoctection"

WORKDIR /home/repo
COPY . .

RUN apt-get update && \
    apt-get install -y git \
                   poppler-utils \
                   python3-pip \
                   python3-opencv \
                   tesseract-ocr

RUN mkdir -p $DEEPDOCTECTION_CACHE

ARG PYTORCH='1.13.1'
ARG TORCH_VISION='0.14.1'
ARG CUDA='cu117'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='$TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install -U pip setuptools
RUN python3 -m pip install --no-cache-dir ".[source-pt]"
