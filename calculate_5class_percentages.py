import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('pixel_class_percentages.csv')

# Initialize counters for each new class
class_pixels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
total_pixels = 0

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

# Sum pixels for each new class
for idx, row in df.iterrows():
    label_id = row['Label ID']
    pixels = row['Pixel Count']
    new_class = label_map_5[label_id]
    class_pixels[new_class] += pixels
    total_pixels += pixels

# Calculate percentages
class_names = {
    0: "Unknown",
    1: "Background/Artifact",
    2: "Inflammatory/Reactive",
    3: "Carcinoma",
    4: "Normal Tissue"
}

print("\nNew 5-Class Distribution:")
print("------------------------")
for class_id in range(5):
    pixels = class_pixels[class_id]
    percentage = (pixels / total_pixels) * 100
    print(f"Class {class_id} ({class_names[class_id]}): {percentage:.2f}%")