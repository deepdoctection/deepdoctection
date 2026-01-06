FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set non-root user variables
ARG USERNAME=developer
ARG USER_UID=1001
ARG USER_GID=1001

# Define version variables
ARG DEEPDOCTECTION_VERSION=1.0.1
ENV DEEPDOCTECTION_VERSION=${DEEPDOCTECTION_VERSION}

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv \
    git curl sudo \
    libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 \
    ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions, handling existing UID/GID
RUN if ! getent group $USER_GID; then groupadd --gid $USER_GID $USERNAME; fi && \
    if ! id -u $USER_UID > /dev/null 2>&1; then useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME; fi && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Ensure the home directory is set correctly
ENV HOME="/home/$USERNAME"

# Create doctr models cache directory and set permissions
RUN mkdir -p $HOME/.cache/doctr/models && chmod -R a+rwX $HOME/.cache/doctr

# Set up a Python virtual environment in the correct directory
RUN python3 -m venv $HOME/venv && \
    echo "source $HOME/venv/bin/activate" >> $HOME/.bashrc

# Activate virtual environment and install necessary packages
ENV PATH="$HOME/venv/bin:$PATH"

# Ensure that uv is set installed
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    pip install --no-cache-dir uv"

# Install base dependencies
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    uv pip install --no-cache-dir uv wheel ninja && \
    uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"

# Ensure torch is installed before detectron2, and install detectron2 with --no-build-isolation
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    uv pip install --no-cache-dir 'detectron2 @ git+https://github.com/deepdoctection/detectron2.git' --no-build-isolation"

# Install remaining dependencies
RUN /bin/bash -c "source $HOME/venv/bin/activate && \
    uv pip install --no-cache-dir deepdoctection[full]==$DEEPDOCTECTION_VERSION && \
    uv pip install --no-cache-dir opencv-python"



# Set default shell
SHELL ["/bin/bash", "-c"]

# Command to start the container
CMD ["bash"]


