FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev python3-venv python3-pil git wget sudo poppler-utils curl ninja-build

RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --user && rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN make -C deepdoctection clean
RUN make -C deepdoctection install-docker-env

# nodejs is required for installing jupyter lab extensions
RUN curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
RUN sudo apt-get install -y nodejs
RUN make -C deepdoctection install-kernel-deepdoc

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]