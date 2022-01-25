FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev python3-venv python3-pil git wget sudo poppler-utils curl

RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --user && rm get-pip.py

# tf version depends on np<1.20. However, we need np>1.20.5 because of our typing. However, this discrepancy has not
# resulted in crashs
RUN pip install --user tensorflow==2.5.
RUN git clone https://github.com/deepdoctection/deepdoctection.git

RUN make -C deepdoctection clean
RUN make -C deepdoctection install-docker-env

# nodejs is required for installing jupyter lab extensions
RUN curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
RUN sudo apt-get install -y nodejs
RUN make -C deepdoctection install-kernel-deepdoc

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
