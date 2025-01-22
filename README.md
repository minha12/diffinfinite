# DiffInfinite 

## Installation Options

### 1. Using Docker (Recommended)

The Docker build process is organized in multiple stages to optimize build time and caching:

1. `base`: NVIDIA CUDA base image with system dependencies
2. `conda-install`: Installs Miniforge (minimal Conda)
3. `conda-env`: Creates Conda environment from environment.yaml
4. `final`: Installs additional pip requirements
5. `ultimate`: Installs extra dependencies and downloads SD model

Build the image using:

```bash
cd ~/diffinfinite

# Build intermediate stage (conda environment)
docker build --target conda-env . -t diffinf:conda-env

# Build final image using cache from previous stage
docker build --cache-from diffinf:conda-env --target ultimate . -t diffinf:latest
```

The image includes:
- CUDA 11.8 with cuDNN 8
- Python environment from environment.yaml
- Pre-downloaded Stable Diffusion model
- All required ML libraries and dependencies

To push to Docker Hub:
```bash
docker login

# Tag with version
docker tag diffinf:latest hale0007/diffinf:2.0.1

# Push to registry
docker push hale0007/diffinf:2.0.1
```

To pull and use the pre-built image:
```bash
docker pull hale0007/diffinf:2.0.1

# Run with GPU support
docker run --gpus all -it hale0007/diffinf:2.0.1

# For development (mount current directory)
docker run --gpus all -v $(pwd):/app -it hale0007/diffinf:2.0.1
```

#### Running on Verdi System

1. Launch container in detached mode:
```bash
sudo docker run -d --gpus all --shm-size=240g -p 6006:6006 --name diffinf_container \
-v "$(pwd)/diffinfinite:/app/diffinfinite" \
-v /usr/local/share/ca-certificates/verdi.crt:/usr/local/share/ca-certificates/verdi.crt \
-e HTTP_PROXY=https://10.253.254.250:3130/ \
-e HTTPS_PROXY=https://10.253.254.250:3130/ \
-e http_proxy=https://10.253.254.250:3130/ \
-e https_proxy=https://10.253.254.250:3130/ \
-e REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/verdi.crt \
hale0007/diffinf:2.0.1 tail -f /dev/null
```

2. Access the container:
```bash
docker exec -it diffinf_container bash
```

3. Update certificates (required before any network operations):
```bash
chmod 644 /usr/local/share/ca-certificates/verdi.crt
update-ca-certificates
```

4. Start training:
```bash
cd /app/diffinfinite
nohup accelerate launch --config_file config/accelerate_config.yaml train.py --config_file config/image_gen_train.yaml > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

5. Monitor training with Tensorboard:
```bash
# Open a new terminal and access the container again
docker exec -it diffinf_container bash

# Launch Tensorboard (replace with your specific log directory)
tensorboard --logdir=/app/diffinfinite/logs/drsk_512x256_5class_20240122_15:09/tensorboard --host 0.0.0.0 --port 6006
```
Then access Tensorboard in your browser at `http://localhost:6006`

#### Container Management

Monitor and control your container:
```bash
# Check container status
docker ps  # List running containers
docker logs diffinf_container  # View container logs

# Monitor training progress
tail -f /app/diffinfinite/logs/training_*.log  # Inside container

# Container lifecycle
docker stop diffinf_container   # Stop container
docker start diffinf_container  # Restart container
```

### 2. Local Installation

Create a conda environment using the requirements file.

```
conda env create -n env_name -f environment.yaml
conda activate env_name
```

Download and unzip the models (```n_classes``` can be 5 or 10):

```
python download.py --n_classes=5
```

Usage example in [Jupyter Notebook](main.ipynb). 


## Synthetic data visualisation

In ```./results```, we share some synthetic data generated with the model. 

In ```./results/large``` we show 2048x2048 images for different ω.

In ```./results/patches``` we show 512x512 images for different ω.