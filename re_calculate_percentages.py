import pandas as pd
import numpy as np
from fire import Fire

# Mapping from original label to new 5-class system
label_map_5 = {
    0: 0,  # unknown
    1: 1,  # background
    2: 1,  # artifact
    3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3,  # carcinoma related
    10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,  # inflammatory
    16: 3, 17: 3, 18: 3,  # more carcinoma
    19: 2, 20: 2,  # reactive
    21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 2, 27: 3,  # mixed
    28: 4, 29: 4,  # normal tissue
    30: 3, 31: 3, 32: 3, 33: 3,  # epidermis abnormal
    34: 2,  # inflammatory
    35: 3, 36: 3, 37: 3, 38: 2, 39: 3, 40: 3,  # mixed
    41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4,  # normal
    47: 4, 48: 4, 49: 4, 50: 4, 51: 4,  # normal
    52: 2,  # reactive
    53: 4, 54: 4  # normal
}

# Mapping from original label to new 10-class system
label_map_10 = {
    0: 0,  # tissue_unknown -> unknown
    1: 1,  # background -> background
    2: 2,  # Artifact_Artifact -> artifacts
    3: 3,  # Dermis_SCC_Keratoacanthoma -> Carcinoma_High_Grade
    4: 4,  # Dermis_Abnormal_BCC -> Carcinoma_Low_Grade
    5: 9,  # Dermis_Abnormal_Benign_fibrous_histiocytoma -> Reactive_Changes
    6: 9,  # Dermis_Abnormal_Compound_nevus -> Reactive_Changes
    7: 9,  # Dermis_Abnormal_Dermatofibroma -> Reactive_Changes
    8: 4,  # Dermis_Abnormal_Dysplastic_nevus -> Carcinoma_Low_Grade
    9: 8,  # Dermis_Abnormal_Granuloma -> Inflammatory_Chronic
    10: 7,  # Dermis_Abnormal_Inflammation -> Inflammatory_Acute
    11: 4,  # Dermis_Abnormal_Inflammation_BCC -> Carcinoma_Low_Grade
    12: 8,  # Dermis_Abnormal_Inflammation_Fibrosis -> Inflammatory_Chronic
    13: 7,  # Dermis_Abnormal_Inflammation_SCC -> Inflammatory_Acute
    14: 8,  # Dermis_Abnormal_Inflammation_fibrosis -> Inflammatory_Chronic
    15: 7,  # Dermis_Abnormal_Inflammatory_edema -> Inflammatory_Acute
    16: 3,  # Dermis_Abnormal_Malignant_melanoma -> Carcinoma_High_Grade
    17: 9,  # Dermis_Abnormal_Neurofibroma -> Reactive_Changes
    18: 9,  # Dermis_Abnormal_Neurofibroma_Surgical -> Reactive_Changes
    19: 9,  # Dermis_Abnormal_Reactive_cellular -> Reactive_Changes
    20: 9,  # Dermis_Abnormal_Reactive_cellular_Surgical -> Reactive_Changes
    21: 9,  # Dermis_Abnormal_Scar -> Reactive_Changes
    22: 9,  # Dermis_Abnormal_Scar_Surgical -> Reactive_Changes
    23: 4,  # Dermis_Abnormal_Seborrheic_keratosis -> Carcinoma_Low_Grade
    24: 3,  # Dermis_Abnormal_SCC -> Carcinoma_High_Grade
    25: 4,  # Dermis_Abnormal_SCC_in_situ -> Carcinoma_Low_Grade
    26: 7,  # Dermis_Abnormal_SCC_Inflammation -> Inflammatory_Acute
    27: 0,  # Dermis_MARGIN -> unknown
    28: 5,  # Dermis_Normal -> Normal_Epithelial
    29: 5,  # Dermis_Normal_Surgical -> Normal_Epithelial
    30: 3,  # Epidermis_SCC_Keratoacanthoma -> Carcinoma_High_Grade
    31: 4,  # Epidermis_Abnormal_Actinic_keratosis -> Carcinoma_Low_Grade
    32: 4,  # Epidermis_Abnormal_BCC -> Carcinoma_Low_Grade
    33: 4,  # Epidermis_Abnormal_Dysplastic_nevus -> Carcinoma_Low_Grade
    34: 7,  # Epidermis_Abnormal_Inflammatory_edema -> Inflammatory_Acute
    35: 3,  # Epidermis_Abnormal_Lentigo_maligna -> Carcinoma_High_Grade
    36: 3,  # Epidermis_Abnormal_Malignant_melanoma -> Carcinoma_High_Grade
    37: 4,  # Epidermis_Abnormal_Melanoma_in_situ -> Carcinoma_Low_Grade
    38: 9,  # Epidermis_Abnormal_Reactive -> Reactive_Changes
    39: 4,  # Epidermis_Abnormal_Seborrheic_keratosis -> Carcinoma_Low_Grade
    40: 4,  # Epidermis_Abnormal_SCC_in_situ -> Carcinoma_Low_Grade
    41: 5,  # Epidermis_Normal -> Normal_Epithelial
    42: 5,  # Epidermis_Normal_Surgical -> Normal_Epithelial
    43: 6,  # Perichondrium_Normal -> Normal_Supporting
    44: 6,  # Perichondrium_Normal_Surgical -> Normal_Supporting
    45: 6,  # Pilosebaceous_Normal -> Normal_Supporting
    46: 6,  # Pilosebaceous_Normal_Surgical -> Normal_Supporting
    47: 6,  # Skin_appendage_Normal -> Normal_Supporting
    48: 6,  # Skin_appendage_Normal_Surgical -> Normal_Supporting
    49: 6,  # Structure_cartilage_Normal -> Normal_Supporting
    50: 6,  # Structure_cartilage_Normal_Surgical -> Normal_Supporting
    51: 6,  # Subcutaneous_fatty_Normal -> Normal_Supporting
    52: 9,  # Subcutaneous_Abnormal_Reactive -> Reactive_Changes
    53: 6,  # Subcutaneous_Normal -> Normal_Supporting
    54: 6   # Subcutaneous_Normal_Surgical -> Normal_Supporting
}

def calculate_class_distribution(csv_path='pixel_class_percentages.csv', num_classes=5):
    """Calculate pixel distribution for different classification schemes.
    
    Args:
        csv_path: Path to the CSV file with original pixel counts
        num_classes: Number of target classes (5 or 10)
    """
    df = pd.read_csv(csv_path)
    
    # Select appropriate label map and class names
    if num_classes == 5:
        label_map = label_map_5
        class_names = {
            0: "Unknown",
            1: "Background/Artifact",
            2: "Inflammatory/Reactive",
            3: "Carcinoma",
            4: "Normal Tissue"
        }
    elif num_classes == 10:
        label_map = label_map_10
        class_names = {
            0: "Unknown",
            1: "Background",
            2: "Artifacts",
            3: "Carcinoma_High_Grade",
            4: "Carcinoma_Low_Grade",
            5: "Normal_Epithelial",
            6: "Normal_Supporting",
            7: "Inflammatory_Acute",
            8: "Inflammatory_Chronic",
            9: "Reactive_Changes"
        }
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")

    # Initialize counters
    class_pixels = {i: 0 for i in range(num_classes)}
    total_pixels = 0

    # Sum pixels for each new class
    for idx, row in df.iterrows():
        label_id = row['Label ID']
        pixels = row['Pixel Count']
        new_class = label_map[label_id]
        class_pixels[new_class] += pixels
        total_pixels += pixels

    # Print results
    print(f"\n{num_classes}-Class Distribution:")
    print("-" * 30)
    for class_id in range(num_classes):
        pixels = class_pixels[class_id]
        percentage = (pixels / total_pixels) * 100
        print(f"Class {class_id} ({class_names[class_id]}): {percentage:.2f}%")

if __name__ == "__main__":
    Fire(calculate_class_distribution)