import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from fire import Fire
import json

def validate_label_maps(label_maps, label_enum_path='label_enum.json'):
    """Validate that label maps cover all classes in label_enum.json"""
    # Load original label enumeration
    with open(label_enum_path, 'r') as f:
        label_enum = json.load(f)
    
    # Verify each map
    for map_name, label_map in label_maps.items():
        # Check that all labels from enum are in map
        missing_labels = set(label_enum.values()) - set(label_map.keys())
        if missing_labels:
            raise ValueError(f"Missing labels in {map_name}: {missing_labels}")
        
        # Check that no extra labels exist in map
        extra_labels = set(label_map.keys()) - set(label_enum.values())
        if extra_labels:
            raise ValueError(f"Extra labels in {map_name}: {extra_labels}")
            
        print(f"✓ {map_name} validated successfully")

# First option (6 classes) with swapped normal and unknown indices
label_map_6 = {
    0: 0,  # tissue_unknown -> unknown (was 5)
    1: 1,  # background -> background/artifact (unchanged)
    2: 1,  # Artifact_Artifact -> background/artifact (unchanged)
    3: 4,  # Dermis_Squamous cell carcinoma, Keratoacanthoma -> carcinoma (unchanged)
    4: 4,  # Dermis_Abnormal, Basal cell carcinoma -> carcinoma
    5: 4,  # Dermis_Abnormal, Benign fibrous histiocytoma -> carcinoma
    6: 4,  # Dermis_Abnormal, Compound nevus -> carcinoma
    7: 4,  # Dermis_Abnormal, Dermatofibroma -> carcinoma
    8: 4,  # Dermis_Abnormal, Dysplastic nevus -> carcinoma
    9: 4,  # Dermis_Abnormal, Granuloma -> carcinoma
    10: 3,  # Dermis_Abnormal, Inflammation -> inflammatory
    11: 3,  # Dermis_Abnormal, Inflammation, Basal cell carcinoma -> inflammatory
    12: 3,  # Dermis_Abnormal, Inflammation, Fibrosis -> inflammatory
    13: 3,  # Dermis_Abnormal, Inflammation, SCC Keratoacanthoma -> inflammatory
    14: 3,  # Dermis_Abnormal, Inflammation, fibrosis -> inflammatory
    15: 3,  # Dermis_Abnormal, Inflammatory edema -> inflammatory
    16: 4,  # Dermis_Abnormal, Malignant melanoma -> carcinoma
    17: 4,  # Dermis_Abnormal, Neurofibroma -> carcinoma
    18: 4,  # Dermis_Abnormal, Neurofibroma, Surgical margin -> carcinoma
    19: 2,  # Dermis_Abnormal, Reactive cellular changes -> reactive
    20: 2,  # Dermis_Abnormal, Reactive cellular changes, Surgical -> reactive
    21: 4,  # Dermis_Abnormal, Scar -> carcinoma
    22: 4,  # Dermis_Abnormal, Scar, Surgical margin -> carcinoma
    23: 4,  # Dermis_Abnormal, Seborrheic keratosis -> carcinoma
    24: 4,  # Dermis_Abnormal, Squamous cell carcinoma -> carcinoma
    25: 4,  # Dermis_Abnormal, Squamous cell carcinoma in situ -> carcinoma
    26: 3,  # Dermis_Abnormal, SCC, Inflammation -> inflammatory
    27: 4,  # Dermis_MARGIN -> carcinoma
    28: 5,  # Dermis_Normal skin -> normal (was 0)
    29: 5,  # Dermis_Normal skin, Surgical margin -> normal (was 0)
    30: 4,  # Epidermis_SCC, Keratoacanthoma -> carcinoma
    31: 4,  # Epidermis_Abnormal, Actinic keratosis -> carcinoma
    32: 4,  # Epidermis_Abnormal, Basal cell carcinoma -> carcinoma
    33: 4,  # Epidermis_Abnormal, Dysplastic nevus -> carcinoma
    34: 3,  # Epidermis_Abnormal, Inflammatory edema -> inflammatory
    35: 4,  # Epidermis_Abnormal, Lentigo maligna melanoma -> carcinoma
    36: 4,  # Epidermis_Abnormal, Malignant melanoma -> carcinoma
    37: 4,  # Epidermis_Abnormal, Melanoma in situ -> carcinoma
    38: 2,  # Epidermis_Abnormal, Reactive cellular changes -> reactive
    39: 4,  # Epidermis_Abnormal, Seborrheic keratosis -> carcinoma
    40: 4,  # Epidermis_Abnormal, SCC in situ -> carcinoma
    41: 5,  # Epidermis_Normal skin -> normal (was 0)
    42: 5,  # Epidermis_Normal skin, Surgical margin -> normal (was 0)
    43: 5,  # Perichondrium_Normal skin -> normal (was 0)
    44: 5,  # Perichondrium_Normal skin, Surgical margin -> normal (was 0)
    45: 5,  # Pilosebaceous apparatus Normal skin -> normal (was 0)
    46: 5,  # Pilosebaceous apparatus Normal skin, Surgical -> normal (was 0)
    47: 5,  # Skin appendage structure_Normal skin -> normal (was 0)
    48: 5,  # Skin appendage structure_Normal skin, Surgical -> normal (was 0)
    49: 5,  # Structure of cartilage Normal skin -> normal (was 0)
    50: 5,  # Structure of cartilage Normal skin, Surgical -> normal (was 0)
    51: 5,  # Subcutaneous fatty tissue_Normal skin -> normal (was 0)
    52: 2,  # Subcutaneous tissue_Abnormal, Reactive -> reactive (unchanged)
    53: 5,  # Subcutaneous tissue_Normal skin -> normal (was 0)
    54: 5   # Subcutaneous tissue_Normal skin, Surgical -> normal (was 0)
}

# Second option (9 classes) with swapped background and unknown indices
label_map_9 = {
    0: 0,  # tissue_unknown -> unknown (was 8)
    1: 8,  # background -> background (was 0)
    2: 7,  # Artifact_Artifact -> artifacts (unchanged)
    3: 0,  # Dermis_SCC_Keratoacanthoma -> unknown (was 8)
    4: 5,  # Dermis_Abnormal_BCC -> basal_cell_carcinoma (unchanged)
    5: 0,  # Dermis_Abnormal_Benign_fibrous_histiocytoma -> unknown (was 8)
    6: 0,  # Dermis_Abnormal_Compound_nevus -> unknown (was 8)
    7: 0,  # Dermis_Abnormal_Dermatofibroma -> unknown (was 8)
    8: 0,  # Dermis_Abnormal_Dysplastic_nevus -> unknown (was 8)
    9: 0,  # Dermis_Abnormal_Granuloma -> unknown (was 8)
    10: 6,  # Dermis_Abnormal_Inflammation -> reactive_inflammatory
    11: 5,  # Dermis_Abnormal_Inflammation_BCC -> basal_cell_carcinoma
    12: 6,  # Dermis_Abnormal_Inflammation_Fibrosis -> reactive_inflammatory
    13: 6,  # Dermis_Abnormal_Inflammation_SCC -> reactive_inflammatory
    14: 6,  # Dermis_Abnormal_Inflammation_fibrosis -> reactive_inflammatory
    15: 6,  # Dermis_Abnormal_Inflammatory_edema -> reactive_inflammatory
    16: 0,  # Dermis_Abnormal_Malignant_melanoma -> unknown (was 8)
    17: 0,  # Dermis_Abnormal_Neurofibroma -> unknown (was 8)
    18: 0,  # Dermis_Abnormal_Neurofibroma_Surgical -> unknown (was 8)
    19: 6,  # Dermis_Abnormal_Reactive_cellular -> reactive_inflammatory
    20: 6,  # Dermis_Abnormal_Reactive_cellular_Surgical -> reactive_inflammatory
    21: 0,  # Dermis_Abnormal_Scar -> unknown (was 8)
    22: 0,  # Dermis_Abnormal_Scar_Surgical -> unknown (was 8)
    23: 0,  # Dermis_Abnormal_Seborrheic_keratosis -> unknown (was 8)
    24: 0,  # Dermis_Abnormal_SCC -> unknown (was 8)
    25: 0,  # Dermis_Abnormal_SCC_in_situ -> unknown (was 8)
    26: 0,  # Dermis_Abnormal_SCC_Inflammation -> unknown (was 8)
    27: 0,  # Dermis_MARGIN -> unknown (was 8)
    28: 1,  # Dermis_Normal -> normal_dermis
    29: 1,  # Dermis_Normal_Surgical -> normal_dermis
    30: 0,  # Epidermis_SCC_Keratoacanthoma -> unknown (was 8)
    31: 0,  # Epidermis_Abnormal_Actinic_keratosis -> unknown (was 8)
    32: 5,  # Epidermis_Abnormal_BCC -> basal_cell_carcinoma
    33: 0,  # Epidermis_Abnormal_Dysplastic_nevus -> unknown (was 8)
    34: 6,  # Epidermis_Abnormal_Inflammatory_edema -> reactive_inflammatory
    35: 0,  # Epidermis_Abnormal_Lentigo_maligna -> unknown (was 8)
    36: 0,  # Epidermis_Abnormal_Malignant_melanoma -> unknown (was 8)
    37: 0,  # Epidermis_Abnormal_Melanoma_in_situ -> unknown (was 8)
    38: 6,  # Epidermis_Abnormal_Reactive_cellular -> reactive_inflammatory
    39: 0,  # Epidermis_Abnormal_Seborrheic_keratosis -> unknown (was 8)
    40: 0,  # Epidermis_Abnormal_SCC_in_situ -> unknown (was 8)
    41: 2,  # Epidermis_Normal -> normal_epidermis
    42: 2,  # Epidermis_Normal_Surgical -> normal_epidermis
    43: 0,  # Perichondrium_Normal -> unknown (was 8)
    44: 0,  # Perichondrium_Normal_Surgical -> unknown (was 8)
    45: 3,  # Pilosebaceous_Normal -> normal_appendages
    46: 3,  # Pilosebaceous_Normal_Surgical -> normal_appendages
    47: 3,  # Skin_appendage_Normal -> normal_appendages
    48: 3,  # Skin_appendage_Normal_Surgical -> normal_appendages
    49: 0,  # Cartilage_Normal -> unknown (was 8)
    50: 0,  # Cartilage_Normal_Surgical -> unknown (was 8)
    51: 4,  # Subcutaneous_fatty_Normal -> normal_subcutaneous
    52: 6,  # Subcutaneous_Abnormal_Reactive -> reactive_inflammatory
    53: 4,  # Subcutaneous_Normal -> normal_subcutaneous
    54: 4   # Subcutaneous_Normal_Surgical -> normal_subcutaneous
}

def reduce_mask_classes(mask_path, output_path, label_map):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # Vectorized conversion or loop
    for old_label, new_label in label_map.items():
        mask_array[mask_array == old_label] = new_label

    new_mask = Image.fromarray(mask_array.astype(np.uint8))
    new_mask.save(output_path)

def process_directory(input_dir: str = "../pathology-datasets/DRSK/init_dataset/masks",
                     output_dir: str = "../pathology-datasets/DRSK/init_dataset/dm-training-data",
                     num_classes: int = 6,
                     validate: bool = False):
    """
    Process mask images to reduce number of classes.
    Args:
        input_dir: Directory containing mask images
        output_dir: Directory to save processed masks
        num_classes: Number of classes to reduce to (6 or 9)
        validate: Whether to validate label maps against label_enum.json
    """
    if validate:
        validate_label_maps({
            'label_map_6': label_map_6,
            'label_map_9': label_map_9
        })
    
    os.makedirs(output_dir, exist_ok=True)
    mask_files = list(Path(input_dir).glob('*.png'))
    
    label_map = label_map_6 if num_classes == 6 else label_map_9
    
    for mask_file in tqdm(mask_files, desc=f"Processing masks ({num_classes} classes)"):
        # Construct output filename with _mask.png suffix
        output_name = mask_file.stem
        if not output_name.endswith('_mask'):
            output_name = f"{output_name}_mask"
        output_file = Path(output_dir) / f"{output_name}.png"
        
        reduce_mask_classes(mask_file, output_file, label_map)

if __name__ == "__main__":
    Fire(process_directory)