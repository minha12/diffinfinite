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

# Third option (5 classes)
label_map_5 = {
    0: 0,  # tissue_unknown -> unknown
    1: 1,  # background -> background/artifact
    2: 1,  # Artifact_Artifact -> background/artifact
    3: 3,  # Dermis_Squamous cell carcinoma, Keratoacanthoma -> carcinoma
    4: 3,  # Dermis_Abnormal, Basal cell carcinoma -> carcinoma
    5: 3,  # Dermis_Abnormal, Benign fibrous histiocytoma -> carcinoma
    6: 3,  # Dermis_Abnormal, Compound nevus -> carcinoma
    7: 3,  # Dermis_Abnormal, Dermatofibroma -> carcinoma
    8: 3,  # Dermis_Abnormal, Dysplastic nevus -> carcinoma
    9: 3,  # Dermis_Abnormal, Granuloma -> carcinoma
    10: 2,  # Dermis_Abnormal, Inflammation -> inflammatory/reactive
    11: 2,  # Dermis_Abnormal, Inflammation, Basal cell carcinoma -> inflammatory/reactive
    12: 2,  # Dermis_Abnormal, Inflammation, Fibrosis -> inflammatory/reactive
    13: 2,  # Dermis_Abnormal, Inflammation, SCC Keratoacanthoma -> inflammatory/reactive
    14: 2,  # Dermis_Abnormal, Inflammation, fibrosis -> inflammatory/reactive
    15: 2,  # Dermis_Abnormal, Inflammation, Inflammatory edema -> inflammatory/reactive
    16: 3,  # Dermis_Abnormal, Malignant melanoma -> carcinoma
    17: 3,  # Dermis_Abnormal, Neurofibroma -> carcinoma
    18: 3,  # Dermis_Abnormal, Neurofibroma, Surgical margin -> carcinoma
    19: 2,  # Dermis_Abnormal, Reactive cellular changes -> inflammatory/reactive
    20: 2,  # Dermis_Abnormal, Reactive cellular changes, Surgical -> inflammatory/reactive
    21: 3,  # Dermis_Abnormal, Scar -> carcinoma
    22: 3,  # Dermis_Abnormal, Scar, Surgical margin -> carcinoma
    23: 3,  # Dermis_Abnormal, Seborrheic keratosis -> carcinoma
    24: 3,  # Dermis_Abnormal, Squamous cell carcinoma -> carcinoma
    25: 3,  # Dermis_Abnormal, Squamous cell carcinoma in situ -> carcinoma
    26: 2,  # Dermis_Abnormal, SCC, Inflammation -> inflammatory/reactive
    27: 3,  # Dermis_MARGIN -> carcinoma
    28: 4,  # Dermis_Normal skin -> normal
    29: 4,  # Dermis_Normal skin, Surgical margin -> normal
    30: 3,  # Epidermis_SCC, Keratoacanthoma -> carcinoma
    31: 3,  # Epidermis_Abnormal, Actinic keratosis -> carcinoma
    32: 3,  # Epidermis_Abnormal, Basal cell carcinoma -> carcinoma
    33: 3,  # Epidermis_Abnormal, Dysplastic nevus -> carcinoma
    34: 2,  # Epidermis_Abnormal, Inflammatory edema -> inflammatory/reactive
    35: 3,  # Epidermis_Abnormal, Lentigo maligna melanoma -> carcinoma
    36: 3,  # Epidermis_Abnormal, Malignant melanoma -> carcinoma
    37: 3,  # Epidermis_Abnormal, Melanoma in situ -> carcinoma
    38: 2,  # Epidermis_Abnormal, Reactive cellular changes -> inflammatory/reactive
    39: 3,  # Epidermis_Abnormal, Seborrheic keratosis -> carcinoma
    40: 3,  # Epidermis_Abnormal, SCC in situ -> carcinoma
    41: 4,  # Epidermis_Normal skin -> normal
    42: 4,  # Epidermis_Normal skin, Surgical margin -> normal
    43: 4,  # Perichondrium_Normal skin -> normal
    44: 4,  # Perichondrium_Normal skin, Surgical margin -> normal
    45: 4,  # Pilosebaceous apparatus Normal skin -> normal
    46: 4,  # Pilosebaceous apparatus Normal skin, Surgical -> normal
    47: 4,  # Skin appendage structure_Normal skin -> normal
    48: 4,  # Skin appendage structure_Normal skin, Surgical -> normal
    49: 4,  # Structure of cartilage Normal skin -> normal
    50: 4,  # Structure of cartilage Normal skin, Surgical -> normal
    51: 4,  # Subcutaneous fatty tissue_Normal skin -> normal
    52: 2,  # Subcutaneous tissue_Abnormal, Reactive -> inflammatory/reactive
    53: 4,  # Subcutaneous tissue_Normal skin -> normal
    54: 4   # Subcutaneous tissue_Normal skin, Surgical -> normal
}

# Fourth option (10 classes)
label_map_10 = {
    0: 0,  # tissue_unknown -> unknown
    1: 1,  # background -> background
    2: 2,  # Artifact_Artifact -> artifacts
    
    # Carcinoma group (combines all types)
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
    
    # Inflammatory conditions
    10: 7, # Inflammation -> inflammatory
    11: 7, # Inflammation_BCC -> inflammatory
    12: 7, # Inflammation_Fibrosis -> inflammatory
    13: 7, # Inflammation_SCC -> inflammatory
    14: 7, # Inflammation_fibrosis -> inflammatory
    15: 7, # Inflammatory_edema -> inflammatory
    26: 7, # SCC_Inflammation -> inflammatory
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
    39: 9, # Seborrheic_keratosis -> structural
    
    # Normal subcutaneous (goes to normal appendages)
    49: 6, # Structure_cartilage -> normal_appendages
    50: 6, # Structure_cartilage_Surgical -> normal_appendages
    51: 6, # Subcutaneous_fatty -> normal_appendages
    53: 6, # Subcutaneous_Normal -> normal_appendages
    54: 6  # Subcutaneous_Normal_Surgical -> normal_appendages
}

def reduce_mask_classes(mask_path, output_path, label_map):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    
    # Create output array filled with unknown class (0)
    output_array = np.zeros_like(mask_array)
    
    # Apply label mapping for known classes
    for old_label, new_label in label_map.items():
        output_array[mask_array == old_label] = new_label
        
    # Any unmapped values will remain 0 (unknown class)
    new_mask = Image.fromarray(output_array.astype(np.uint8))
    new_mask.save(output_path)

def process_directory(input_dir: str = "../pathology-datasets/DRSK/full_dataset/masks",
                     output_dir: str = "../pathology-datasets/DRSK/full_dataset/dm-training-data",
                     num_classes: int = 5,
                     validate: bool = False,
                     ext: str = 'jpg'):
    """
    Process mask images to reduce number of classes.
    Args:
        input_dir: Directory containing mask images
        output_dir: Directory to save processed masks
        num_classes: Number of classes to reduce to (5, 6, 9, or 10)
        validate: Whether to validate label maps against label_enum.json
        ext: File extension of mask images ('png' or 'jpg')
    """
    if validate:
        validate_label_maps({
            'label_map_5': label_map_5,
            'label_map_6': label_map_6,
            'label_map_9': label_map_9,
            'label_map_10': label_map_10
        })
    
    os.makedirs(output_dir, exist_ok=True)
    mask_files = list(Path(input_dir).glob(f'*.{ext}'))
    
    label_map = {
        5: label_map_5,
        6: label_map_6,
        9: label_map_9,
        10: label_map_10
    }.get(num_classes)
    
    if label_map is None:
        raise ValueError(f"Invalid number of classes: {num_classes}. Must be 5, 6, 9, or 10.")
    
    for mask_file in tqdm(mask_files, desc=f"Processing masks ({num_classes} classes)"):
        # Construct output filename with _mask.png suffix
        output_name = mask_file.stem
        if not output_name.endswith('_mask'):
            output_name = f"{output_name}_mask"
        output_file = Path(output_dir) / f"{output_name}.png"
        
        reduce_mask_classes(mask_file, output_file, label_map)

if __name__ == "__main__":
    Fire(process_directory)