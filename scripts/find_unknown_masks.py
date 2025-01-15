import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Path to the training data directory
data_dir = Path("../pathology-datasets/DRSK/full_dataset/dm-training-data")

# List to store matching filenames
unknown_only_files = []

# Get all mask files
mask_files = list(data_dir.glob("*_mask.png"))

# Process each mask file
for mask_path in tqdm(mask_files, desc="Processing mask files", unit="file"):
    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"\nError: Could not load mask at {mask_path}")
        continue

    # Check if mask contains only label 0
    unique_labels = np.unique(mask)
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        # Get original filename without _mask.png
        base_name = mask_path.stem[:-5]  # removes "_mask" from the stem
        unknown_only_files.append(base_name)

# Write results to file
output_file = 'unknown_tissue_only_files.txt'
with open(output_file, 'w') as f:
    for filename in unknown_only_files:
        f.write(f"{filename}\n")

print(f"Found {len(unknown_only_files)} files containing only unknown tissue (label 0)")
print(f"File names have been written to {output_file}")
