FROM python:3.9-slim
LABEL repository="deepdoctection"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /home/repo
COPY . .

RUN apt-get update && \
    apt install -y git \
                   poppler-utils \
                   python3-pip \
                   python3-opencv \
                   tesseract-ocr

RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

RUN python3 -m pip install --no-cache-dir ".[source-pt]"
