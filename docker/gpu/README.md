# DeepDoctection Docker Image

This repository provides a Dockerfile to build a containerized environment for **DeepDoctection**. The image is based 
on **NVIDIA CUDA 12.8** and comes pre-configured with PyTorch, Detectron2, and other necessary dependencies.

## **Table of Contents**
- [Dockerfile Breakdown](#dockerfile-breakdown)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Docker Container](#running-the-docker-container)
- [Getting a Docker image from the Docker hub for the last published release](#getting-a-docker-image-from-the-docker-hub-for-the-last-published-release)
- [Pulling images from the Docker hub:](#pulling-images-from-the-docker-hub)
- [Starting a container with docker compose](#starting-a-container-with-docker-compose)

---

## **Dockerfile Breakdown**
The Dockerfile consists of several key steps:

### **1️⃣ Base Image**
```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```
The image is built on **NVIDIA CUDA 12.8**, ensuring compatibility with GPU acceleration and deep learning frameworks.

### **2️⃣ Define Non-Root User**
```dockerfile
ARG USERNAME=developer
ARG USER_UID=1001
ARG USER_GID=1001
```
We define a non-root user (`developer`) with a configurable UID and GID.

### **3️⃣ Define Version Variables**
```dockerfile
ARG DEEPDOCTECTION_VERSION=1.0.0
ENV DEEPDOCTECTION_VERSION=${DEEPDOCTECTION_VERSION}
```
The DeepDoctection version is set as an environment variable, making it easier to update the version.

### **4️⃣ Install System Dependencies**
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv \
    git curl sudo \
    libsm6 libxext6 libxrender-dev \
    ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```
These system dependencies are required for Python, PyTorch, and Detectron2.

### **5️⃣ Create Non-Root User and Set Permissions**
```dockerfile
RUN if ! getent group $USER_GID; then groupadd --gid $USER_GID $USERNAME; fi && \
    if ! id -u $USER_UID > /dev/null 2>&1; then useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME; fi && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME
```
This ensures the `developer` user is created and has the necessary permissions.

### **6️⃣ Switch to Non-Root User and Set Up Environment**
```dockerfile
USER $USERNAME
WORKDIR /home/$USERNAME
ENV HOME="/home/$USERNAME"
```
The working directory is set, and the home directory is properly configured.

### **7️⃣ Create and Configure a Python Virtual Environment**
```dockerfile
RUN python3 -m venv $HOME/venv && \
    echo "source $HOME/venv/bin/activate" >> $HOME/.bashrc
```
This ensures that all Python dependencies are installed within a virtual environment.

### **8️⃣ Install Python Dependencies Using `uv`**
```dockerfile
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    pip install --no-cache-dir uv"
```
`uv` is a modern package manager used for fast and efficient dependency installation.

### **9️⃣ Install PyTorch and Related Dependencies**
```dockerfile
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    uv pip install --no-cache-dir uv wheel ninja && \
    uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
```
PyTorch and its dependencies are installed with CUDA 12.1 support.

### **🔟 Install Detectron2**
```dockerfile
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    uv pip install --no-cache-dir 'detectron2 @ git+https://github.com/deepdoctection/detectron2.git' --no-build-isolation"
```
Detectron2 is installed from a specific GitHub repository.

### **🔟+1 Install DeepDoctection and OpenCV**
```dockerfile
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    uv pip install --no-cache-dir deepdoctection[full]==$DEEPDOCTECTION_VERSION && \
    uv pip install --no-cache-dir opencv-python"
```
DeepDoctection and OpenCV are installed for document processing tasks.

### **🔟+2 Default Shell and CMD**
```dockerfile
SHELL ["/bin/bash", "-c"]
CMD ["bash"]
```
The default shell is set to bash, and the container will start with an interactive shell.

---

## **Building the Docker Image**
To build the Docker image, use the following command:

```bash
docker build -t deepdoctection/dd:1.0.0 -f Dockerfile .
```
This will create an image tagged as `deepdoctection/base:1.0.0`.

---

## **Running the Docker Container**
To start a container from the built image, run:

```bash
docker run --gpus all -it --rm \
    -v deepdoctection_cache:/home/developer/.cache/deepdoctection \
    deepdoctection/base:1.0.0
```
This ensures GPU support and mounts a cache directory.

## Getting a Docker image from the Docker hub for the last published release

With the release of version v.0.27.0, we are starting to provide Docker images for the full installation. 
This is due to the fact that the requirements and dependencies are complex and even the construction of Docker images 
can lead to difficulties.

## Pulling images from the Docker hub:

```
docker pull deepdoctection/deepdoctection:<release_tag>
```

The container can be started with the above `docker run` command.

## Starting a container with docker compose

We provide a `docker-compose.yaml` file to start the generated image pulled from the hub. In order to use it, replace 
first the image argument with the tag, you want to use. Second, in the `.env` file, set the two environment variables:

`CACHE_HOST`: Model weights/configuration files, as well as potentially datasets, are not baked into the image during 
the build time, but mounted into the container with volumes. For a local installation, this is 
usually `~/.cache/deepdoctection`.

`WORK_DIR`: A temporary directory where documents can be loaded and also mounted into the container.

The container can be started as usual, for example, with:

```
docker compose up -d
```

Using the interpreter of the container, you can then run something like this:

```python
import deepdoctection as dd

if __name__=="__main__":

    analyzer = dd.get_dd_analyzer()

    df = analyzer.analyze(path = "/home/files/your_doc.pdf")
    df.reset_state()

    for dp in df:
        print(dp.text)
```



