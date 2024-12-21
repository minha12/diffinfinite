#!/bin/bash

#SBATCH -A Berzelius-2024-460
#SBATCH -C fat
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00  # 3 days
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err
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
python train.py

# Kill the monitoring process
kill $MONITOR_PID