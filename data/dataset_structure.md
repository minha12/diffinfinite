# Dataset Structure and Processing Flow

## Directory Structure
```bash
/path/to/dataset/
├── dataset.csv # Generated if missing (image file names and classes)
├── label_map_5.yml # Default label map file
├── label_map_10.yml # (Optional) Another label map if your config needs 10 classes
├── image1.jpg
├── image1_mask.png
├── image2.jpg
├── image2_mask.png
├── ... (additional image/mask pairs)
└── ...
```

## Data Processing Flow

### a) CSV Generation (create_dataset_csv)
- Scans data_path directory (default ".images/path") for image files (.jpg)
- For each image:
  - Checks corresponding mask file (.png) for non-zero area above threshold
  - Loads mask as numpy array and processes it:
    - Calculates mean of non-zero pixels (np.mean(mask > 0))
    - Only includes images where mean exceeds threshold
    - Extracts unique integer labels from mask using np.unique()
  - Formats data for CSV:
    - Image filename (without extension)
    - Space-separated string of unique class labels
- Writes image filenames and labels to dataset.csv

### b) Class Name Loading and Label Mapping (get_class_names)
- Reads YAML file (e.g., label_map_5.yml) containing class-to-integer mappings
- Implements fallback logic for missing files in data_path
- Handles special cases:
  - Maps label 9 to index 2
  - Adjusts indexes for consistency

### c) Dataset Splitting (split_dataset)
- Processes dataset.csv into class-to-images dictionary via dataset_to_dict
- Uses train_test_split for per-class train/test subdivision
- Returns separate dictionaries for train and test sets

### d) Dataset Class (DatasetLung)
- Loads images and masks using provided filename dictionaries
- Supports optional features:
  - Data transforms (ComposeState, identity, RandomRotate90)
  - Unbalanced data sampling with configurable class probabilities
- Returns image-mask pairs

### e) DataLoader Creation (import_dataset)
- Generates dataset.csv if missing or when force=True
- Creates train/test splits via split_dataset
- Initializes DatasetLung instances:
  - Training: configurable cond_drop_prob
  - Testing: cond_drop_prob=1
- Returns PyTorch DataLoader objects for batch processing