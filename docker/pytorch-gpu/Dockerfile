FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /home/repo
COPY . .

RUN apt-get update && \
    apt-get install -y git \
                   poppler-utils \
                   python3-pip \
                   python3-opencv \
                   tesseract-ocr

ARG PYTORCH='1.9.0'
ARG TORCH_VISION=''
ARG CUDA='cu111'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir ".[source-pt]"

# start container with
# docker run --rm -gpus all -d -t <image:tag>
# Will require NVIDIA CONTAINER Toolkit, see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide
