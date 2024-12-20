#!/bin/bash

#SBATCH -A Berzelius-2024-460
#SBATCH -C fat
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00  # 7 days
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err
#SBATCH -J diffusion_train

# Load modules
module load Miniforge3

# Activate conda environment
mamba activate diffinf  # Replace with your env name

# Set working directory (optional)
cd /proj/berzelius-2023-296/users/x_lemin/diffinfinite

# Run training script
python train.py --config_file config/image_gen_train.yaml
