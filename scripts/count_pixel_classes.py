import numpy as np
import cv2
from pathlib import Path
import csv
from tqdm import tqdm

# Define the labels dictionary
labels = {
    "tissue_unknown": 0,
    "background": 1,
    "Artifact_Artifact": 2,
    "Dermis_, Squamous cell carcinoma, Keratoacanthoma": 3,
    "Dermis_Abnormal, Basal cell carcinoma": 4,
    "Dermis_Abnormal, Benign fibrous histiocytoma": 5,
    "Dermis_Abnormal, Compound nevus": 6,
    "Dermis_Abnormal, Dermatofibroma": 7,
    "Dermis_Abnormal, Dysplastic nevus": 8,
    "Dermis_Abnormal, Granuloma": 9,
    "Dermis_Abnormal, Inflammation": 10,
    "Dermis_Abnormal, Inflammation, Basal cell carcinoma": 11,
    "Dermis_Abnormal, Inflammation, Fibrosis": 12,
    "Dermis_Abnormal, Inflammation, Squamous cell carcinoma, Keratoacanthoma": 13,
    "Dermis_Abnormal, Inflammation, fibrosis": 14,
    "Dermis_Abnormal, Inflammatory edema": 15,
    "Dermis_Abnormal, Malignant melanoma": 16,
    "Dermis_Abnormal, Neurofibroma": 17,
    "Dermis_Abnormal, Neurofibroma, Surgical margin": 18,
    "Dermis_Abnormal, Reactive cellular changes": 19,
    "Dermis_Abnormal, Reactive cellular changes, Surgical margin": 20,
    "Dermis_Abnormal, Scar": 21,
    "Dermis_Abnormal, Scar, Surgical margin": 22,
    "Dermis_Abnormal, Seborrheic keratosis": 23,
    "Dermis_Abnormal, Squamous cell carcinoma": 24,
    "Dermis_Abnormal, Squamous cell carcinoma in situ": 25,
    "Dermis_Abnormal, Squamous cell carcinoma, Inflammation": 26,
    "Dermis_MARGIN": 27,
    "Dermis_Normal skin": 28,
    "Dermis_Normal skin, Surgical margin": 29,
    "Epidermis_, Squamous cell carcinoma, Keratoacanthoma": 30,
    "Epidermis_Abnormal, Actinic keratosis": 31,
    "Epidermis_Abnormal, Basal cell carcinoma": 32,
    "Epidermis_Abnormal, Dysplastic nevus": 33,
    "Epidermis_Abnormal, Inflammatory edema": 34,
    "Epidermis_Abnormal, Lentigo maligna melanoma": 35,
    "Epidermis_Abnormal, Malignant melanoma": 36,
    "Epidermis_Abnormal, Melanoma in situ": 37,
    "Epidermis_Abnormal, Reactive cellular changes": 38,
    "Epidermis_Abnormal, Seborrheic keratosis": 39,
    "Epidermis_Abnormal, Squamous cell carcinoma in situ": 40,
    "Epidermis_Normal skin": 41,
    "Epidermis_Normal skin, Surgical margin": 42,
    "Perichondrium_Normal skin": 43,
    "Perichondrium_Normal skin, Surgical margin": 44,
    "Pilosebaceous apparatus structure_Normal skin": 45,
    "Pilosebaceous apparatus structure_Normal skin, Surgical margin": 46,
    "Skin appendage structure_Normal skin": 47,
    "Skin appendage structure_Normal skin, Surgical margin": 48,
    "Structure of cartilage of auditory canal_Normal skin": 49,
    "Structure of cartilage of auditory canal_Normal skin, Surgical margin": 50,
    "Subcutaneous fatty tissue_Normal skin": 51,
    "Subcutaneous tissue_Abnormal, Reactive cellular changes": 52,
    "Subcutaneous tissue_Normal skin": 53,
    "Subcutaneous tissue_Normal skin, Surgical margin": 54
}

# Reverse the dictionary for label to name lookup
label_to_name = {v: k for k, v in labels.items()}

# Initialize a dictionary to hold pixel counts per class
pixel_counts = {label_id: 0 for label_id in labels.values()}

# Total number of pixels
total_pixels = 0

# Path to the masks directory
mask_dir = Path("../pathology-datasets/DRSK/full_dataset/masks")

# Get list of all mask files and count them
mask_files = list(mask_dir.glob("*.jpg"))
total_files = len(mask_files)

# Loop through all mask files with proper progress bar
for mask_path in tqdm(mask_files, 
                     total=total_files,
                     desc="Processing mask files",
                     unit="file"):
    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"\nError: Could not load mask at {mask_path}")
        continue

    # Update total pixels
    total_pixels += mask.size

    # Get unique labels and their counts in the mask
    unique_labels, counts = np.unique(mask, return_counts=True)

    # Accumulate counts per label
    for label, count in zip(unique_labels, counts):
        if label in pixel_counts:
            pixel_counts[label] += count
        else:
            pixel_counts[label] = count

# Calculate percentages
percentages = {label_id: (count / total_pixels) * 100 for label_id, count in pixel_counts.items()}

# Write results to a CSV file
output_file = 'pixel_class_percentages_full.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Label ID', 'Class Name', 'Pixel Count', 'Percentage'])
    for label_id, count in pixel_counts.items():
        class_name = label_to_name.get(label_id, 'Unknown')
        percentage = percentages[label_id]
        writer.writerow([label_id, class_name, count, f"{percentage:.4f}"])

print(f"Results written to {output_file}")