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
    
    # Carcinoma group
    3: 3,  # Dermis_SCC -> carcinoma
    4: 3,  # Dermis_BCC -> carcinoma
    16: 3, # Malignant_melanoma -> carcinoma
    24: 3, # SCC -> carcinoma
    25: 3, # SCC_in_situ -> carcinoma
    30: 3, # Epidermis_SCC -> carcinoma
    31: 3, # Actinic_keratosis -> carcinoma
    32: 3, # Epidermis_BCC -> carcinoma
    35: 3, # Lentigo_maligna_melanoma -> carcinoma
    36: 3, # Epidermis_Malignant_melanoma -> carcinoma
    37: 3, # Melanoma_in_situ -> carcinoma
    40: 3, # Epidermis_SCC_in_situ -> carcinoma
    
    # Normal dermis
    28: 4, # Dermis_Normal -> normal_dermis
    29: 4, # Dermis_Normal_Surgical -> normal_dermis
    
    # Normal epidermis
    41: 5, # Epidermis_Normal -> normal_epidermis
    42: 5, # Epidermis_Normal_Surgical -> normal_epidermis
    
    # Normal appendages
    43: 6, # Perichondrium_Normal -> normal_appendages
    44: 6, # Perichondrium_Normal_Surgical -> normal_appendages
    45: 6, # Pilosebaceous_Normal -> normal_appendages
    46: 6, # Pilosebaceous_Normal_Surgical -> normal_appendages
    47: 6, # Skin_appendage_Normal -> normal_appendages
    48: 6, # Skin_appendage_Normal_Surgical -> normal_appendages
    49: 6, # Structure_cartilage -> normal_appendages
    50: 6, # Structure_cartilage_Surgical -> normal_appendages
    51: 6, # Subcutaneous_fatty -> normal_appendages
    53: 6, # Subcutaneous_Normal -> normal_appendages
    54: 6, # Subcutaneous_Normal_Surgical -> normal_appendages
    
    # Inflammatory conditions
    10: 7, # Inflammation -> inflammatory
    11: 7, # Inflammation_BCC -> inflammatory
    12: 7, # Inflammation_Fibrosis -> inflammatory
    13: 7, # Inflammation_SCC -> inflammatory
    14: 7, # Inflammation_fibrosis -> inflammatory
    15: 7, # Inflammatory_edema -> inflammatory
    26: 7, # SCC_Inflammation -> inflammatory
    27: 0, # Dermis_MARGIN -> unknown
    34: 7, # Epidermis_Inflammatory -> inflammatory
    
    # Reactive changes
    19: 8, # Reactive_cellular -> reactive
    20: 8, # Reactive_cellular_Surgical -> reactive
    38: 8, # Epidermis_Reactive -> reactive
    52: 8, # Subcutaneous_Reactive -> reactive
    
    # Structural changes
    5: 9,  # Benign_fibrous_histiocytoma -> structural
    6: 9,  # Compound_nevus -> structural
    7: 9,  # Dermatofibroma -> structural
    8: 9,  # Dysplastic_nevus -> structural
    9: 9,  # Granuloma -> structural
    17: 9, # Neurofibroma -> structural
    18: 9, # Neurofibroma_Surgical -> structural
    21: 9, # Scar -> structural
    22: 9, # Scar_Surgical -> structural
    23: 9, # Seborrheic_keratosis -> structural
    33: 9, # Epidermis_Dysplastic -> structural
    39: 9  # Seborrheic_keratosis -> structural
}

def calculate_class_distribution(csv_path='pixel_class_percentages_full.csv', num_classes=5):
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
            3: "Carcinoma",
            4: "Normal Dermis",
            5: "Normal Epidermis",
            6: "Normal Appendages",
            7: "Inflammatory",
            8: "Reactive",
            9: "Structural"
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