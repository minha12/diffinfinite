import os
import csv
import shutil
from pathlib import Path
from tqdm import tqdm

# Define paths
TRAIN_DATA_PATH = "../pathology-datasets/DRSK/full_dataset/dm-training-data"
UNCOND_DATA_PATH = "../pathology-datasets/DRSK/full_dataset/unconditional-data"
LABELED_DATA_PATH = "../pathology-datasets/DRSK/full_dataset/labeled-data"
CSV_PATH = os.path.join(TRAIN_DATA_PATH, "dataset.csv")

def process_unconditional_data(used_files):
    # Create unconditional data directory if it doesn't exist
    Path(UNCOND_DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Find all jpg files in training directory
    train_path = Path(TRAIN_DATA_PATH)
    jpg_files = list(train_path.glob("*.jpg"))

    # Process each jpg file
    for jpg_file in tqdm(jpg_files, desc="Processing unconditional data"):
        file_stem = jpg_file.stem
        if file_stem not in used_files:
            # Copy jpg to unconditional directory
            shutil.copy2(jpg_file, UNCOND_DATA_PATH)
            
def process_labeled_data(labeled_files):
    # Create labeled data directory if it doesn't exist
    Path(LABELED_DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Process each labeled file
    for file_stem in tqdm(labeled_files, desc="Processing labeled data"):
        jpg_file = Path(TRAIN_DATA_PATH) / f"{file_stem}.jpg"
        mask_file = Path(TRAIN_DATA_PATH) / f"{file_stem}_mask.png"
        
        # Copy files if they exist
        if jpg_file.exists():
            shutil.copy2(jpg_file, LABELED_DATA_PATH)
        if mask_file.exists():
            shutil.copy2(mask_file, LABELED_DATA_PATH)

def main():
    # Read dataset.csv to get list of labeled files
    labeled_files = set()
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # make sure row is not empty
                labeled_files.add(row[0])

    # Process both labeled and unconditional data
    process_labeled_data(labeled_files)
    process_unconditional_data(labeled_files)

if __name__ == "__main__":
    main()