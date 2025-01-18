#!/bin/bash

#SBATCH -A Berzelius-2024-460
#SBATCH -C thin
#SBATCH --gpus=8
#SBATCH -t 36:00:00  
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH -J diffusion_train

# Load modules
module load Miniforge3

# Activate conda environment
mamba activate diffinf  # Replace with your env name

# Print GPU information before starting
echo "=== GPU Information at Start ==="
nvidia-smi

# Set working directory
cd /proj/berzelius-2023-296/users/x_lemin/diffinfinite

# Run training with GPU monitoring in background
(while true; do 
    echo "=== GPU Usage $(date) ===" >> gpu_usage.log
    nvidia-smi >> gpu_usage.log
    sleep 60  # Check every minute
done) &

# Store the monitoring process ID
MONITOR_PID=$!

# Set working directory (optional)
cd /proj/berzelius-2023-296/users/x_lemin/diffinfinite

# Run training script
accelerate launch --config_file config/accelerate_config-x8.yaml train.py --config_file config/image_gen_train-x8.yaml

# Kill the monitoring process
kill $MONITOR_PID