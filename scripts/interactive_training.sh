#!/bin/bash

# Request interactive session
interactive --gpus=8 -C thin -t 36:00:00 -A Berzelius-2024-460 << EOF

# Load module and activate environment
module load Miniforge3
mamba activate diffinf

# Navigate to project directory (adjust path as needed)
cd /proj/berzelius-2023-296/users/x_lemin/diffinfinite

# Run training
accelerate launch --config_file config/accelerate_config-x8.yaml train.py --config_file config/image_gen_train-x8.yaml

EOF