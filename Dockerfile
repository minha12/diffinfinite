# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stage for Miniforge installation - Copy all from base
FROM base AS conda-install
# We inherit the base image including system dependencies
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh

# Set up Conda environment
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash && \
    conda install -y mamba -n base -c conda-forge

# Stage for conda environment setup
FROM conda-install AS conda-env
# We inherit from conda-install which has both system deps and conda
COPY environment.yaml .
RUN mamba env create -f environment.yaml

# Stage for pip requirements
FROM conda-env AS final
# We inherit everything from conda-env
COPY requirements.txt .
# Add conda environment to PATH to use the correct pip
RUN /opt/conda/envs/diffinf/bin/pip install -r requirements.txt

# Set default environment variables
ENV PATH /opt/conda/envs/diffinf/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES=0

FROM final AS ultimate
# We inherit everything from final
RUN /opt/conda/envs/diffinf/bin/pip install scikit-learn

# Copy scripts directory
COPY scripts/download_sd_model.py /app/scripts/download_sd_model.py

# Verify CUDA and PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda}')"

# Download SD model during build
RUN python /app/scripts/download_sd_model.py

WORKDIR /app
CMD ["python"]