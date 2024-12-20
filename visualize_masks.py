import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random

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

# Define a color scheme for all label IDs (RGB values in range 0-255)
colors = {
    0: (128, 128, 128),   # tissue_unknown
    1: (255, 255, 255),   # background
    2: (255, 0, 0),       # Artifact_Artifact
    3: (0, 255, 0),       # Dermis_, Squamous cell carcinoma, Keratoacanthoma
    4: (0, 0, 255),       # Dermis_Abnormal, Basal cell carcinoma
    5: (255, 255, 0),     # Dermis_Abnormal, Benign fibrous histiocytoma
    6: (255, 165, 0),     # Dermis_Abnormal, Compound nevus
    7: (128, 0, 128),     # Dermis_Abnormal, Dermatofibroma
    8: (0, 255, 255),     # Dermis_Abnormal, Dysplastic nevus
    9: (255, 192, 203),   # Dermis_Abnormal, Granuloma
    10: (0, 128, 128),    # Dermis_Abnormal, Inflammation
    11: (255, 105, 180),  # Dermis_Abnormal, Inflammation, Basal cell carcinoma
    12: (75, 0, 130),     # Dermis_Abnormal, Inflammation, Fibrosis
    13: (0, 100, 0),      # Dermis_Abnormal, Inflammation, Squamous cell carcinoma, Keratoacanthoma
    14: (85, 107, 47),    # Dermis_Abnormal, Inflammation, fibrosis
    15: (46, 139, 87),    # Dermis_Abnormal, Inflammatory edema
    16: (139, 0, 0),      # Dermis_Abnormal, Malignant melanoma
    17: (70, 130, 180),   # Dermis_Abnormal, Neurofibroma
    18: (0, 191, 255),    # Dermis_Abnormal, Neurofibroma, Surgical margin
    19: (244, 164, 96),   # Dermis_Abnormal, Reactive cellular changes
    20: (112, 128, 144),  # Dermis_Abnormal, Reactive cellular changes, Surgical margin
    21: (210, 105, 30),   # Dermis_Abnormal, Scar
    22: (188, 143, 143),  # Dermis_Abnormal, Scar, Surgical margin
    23: (255, 20, 147),   # Dermis_Abnormal, Seborrheic keratosis
    24: (65, 105, 225),   # Dermis_Abnormal, Squamous cell carcinoma
    25: (0, 250, 154),    # Dermis_Abnormal, Squamous cell carcinoma in situ
    26: (127, 255, 0),    # Dermis_Abnormal, Squamous cell carcinoma, Inflammation
    27: (220, 20, 60),    # Dermis_MARGIN
    28: (173, 216, 230),  # Dermis_Normal skin
    29: (255, 182, 193),  # Dermis_Normal skin, Surgical margin
    30: (0, 255, 127),    # Epidermis_, Squamous cell carcinoma, Keratoacanthoma
    31: (216, 191, 216),  # Epidermis_Abnormal, Actinic keratosis
    32: (218, 165, 32),   # Epidermis_Abnormal, Basal cell carcinoma
    33: (124, 252, 0),    # Epidermis_Abnormal, Dysplastic nevus
    34: (102, 205, 170),  # Epidermis_Abnormal, Inflammatory edema
    35: (153, 50, 204),   # Epidermis_Abnormal, Lentigo maligna melanoma
    36: (160, 82, 45),    # Epidermis_Abnormal, Malignant melanoma
    37: (135, 206, 250),  # Epidermis_Abnormal, Melanoma in situ
    38: (255, 99, 71),    # Epidermis_Abnormal, Reactive cellular changes
    39: (255, 140, 0),    # Epidermis_Abnormal, Seborrheic keratosis
    40: (255, 215, 0),    # Epidermis_Abnormal, Squamous cell carcinoma in situ
    41: (100, 149, 237),  # Epidermis_Normal skin
    42: (72, 61, 139),    # Epidermis_Normal skin, Surgical margin
    43: (205, 92, 92),    # Perichondrium_Normal skin
    44: (255, 228, 196),  # Perichondrium_Normal skin, Surgical margin
    45: (0, 0, 139),      # Pilosebaceous apparatus structure_Normal skin
    46: (0, 128, 0),      # Pilosebaceous apparatus structure_Normal skin, Surgical margin
    47: (85, 107, 47),    # Skin appendage structure_Normal skin
    48: (128, 0, 0),      # Skin appendage structure_Normal skin, Surgical margin
    49: (219, 112, 147),  # Structure of cartilage of auditory canal_Normal skin
    50: (244, 164, 96),   # Structure of cartilage of auditory canal_Normal skin, Surgical margin
    51: (128, 128, 0),    # Subcutaneous fatty tissue_Normal skin
    52: (255, 228, 225),  # Subcutaneous tissue_Abnormal, Reactive cellular changes
    53: (95, 158, 160),   # Subcutaneous tissue_Normal skin
    54: (0, 0, 0),        # Subcutaneous tissue_Normal skin, Surgical margin
}

# Ensure colors are in the range [0,1] for matplotlib
colors = {label_id: tuple(c / 255.0 for c in color) for label_id, color in colors.items()}

def get_random_mask():
    mask_dir = Path("../pathology-datasets/DRSK/image_patches_512_20x/masks")
    mask_files = list(mask_dir.glob("*.png"))
    if not mask_files:
        raise ValueError("No mask files found in the directory")
    return random.choice(mask_files).name

def visualize_mask(image_name):
    # Path to the mask
    mask_path = Path("../pathology-datasets/DRSK/image_patches_512_20x/masks") / image_name

    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error: Could not load mask at {mask_path}")
        return

    # Create RGB image for visualization
    height, width = mask.shape
    visualization = np.zeros((height, width, 3))
    
    # Apply colors to each label
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label in label_to_name and label in colors:
            mask_region = (mask == label)
            visualization[mask_region] = colors[label]
        else:
            print(f"No color defined for label ID {label}")
            visualization[mask == label] = (0.5, 0.5, 0.5)  # Default gray color

    # Plot the visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(visualization)
    plt.title(f"Mask Visualization for {image_name}")
    
    # Create legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[label]) 
                      for label in unique_labels if label in label_to_name and label in colors]
    legend_labels = [label_to_name[label][:30] + "..." if len(label_to_name[label]) > 30 
                    else label_to_name[label] for label in unique_labels if label in label_to_name and label in colors]
    
    plt.legend(legend_elements, legend_labels, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'visualization_{image_name}.png', bbox_inches='tight', dpi=300)
    plt.show()

# Example usage
image_name = get_random_mask()
print(f"Randomly selected mask: {image_name}")
visualize_mask(image_name)