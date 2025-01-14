# %%
import os
from PIL import Image
import numpy as np

# Specify the directory containing the mask images
mask_dir = 'images/inpainting/'

# Get a list of all files in the directory
file_list = os.listdir(mask_dir)

# Iterate through the files
for filename in file_list:
    # Check if the file is a mask image (you might need to adjust the suffix)
    if filename.endswith('_mask.png'):
        # Construct the full path to the mask image
        mask_path = os.path.join(mask_dir, filename)

        try:
            # Load the mask image using PIL
            mask_image = Image.open(mask_path)

            # Convert the PIL image to a NumPy array
            mask_array = np.array(mask_image)

            # Find the unique class values in the mask array
            unique_classes = np.unique(mask_array)

            # Print the filename and the unique classes
            print(f"Unique classes in {filename}: {unique_classes}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
# %%
import os
import random
from PIL import Image
import numpy as np

# Specify the directory containing the mask images
mask_dir = '../pathology-datasets/DRSK/image_patches_512_20x/masks'

# Get a list of all files in the directory
file_list = os.listdir(mask_dir)

# Filter for files ending with '.png'
mask_files = [f for f in file_list if f.endswith('.png')]

# Check if there are any mask files
if not mask_files:
    print(f"No mask files found in: {mask_dir}")
else:
    # Select a random mask file
    random_mask_filename = random.choice(mask_files)

    # Construct the full path to the random mask image
    random_mask_path = os.path.join(mask_dir, random_mask_filename)

    try:
        # Load the mask image using PIL
        mask_image = Image.open(random_mask_path)

        # Convert the PIL image to a NumPy array
        mask_array = np.array(mask_image)

        # Find the unique class values in the mask array
        unique_classes = np.unique(mask_array)

        # Print the filename and the unique classes
        print(f"Unique classes in random mask '{random_mask_filename}': {unique_classes}")

    except Exception as e:
        print(f"Error processing {random_mask_filename}: {e}")
# %%
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Load the first mask ---
mask_dir1 = 'images/inpainting/'
file_list1 = os.listdir(mask_dir1)
mask_files1 = [f for f in file_list1 if f.endswith(('_mask.png', '.png'))] # Adjust suffix if needed
if mask_files1:
    mask_filename1 = random.choice(mask_files1)
    mask_path1 = os.path.join(mask_dir1, mask_filename1)
    try:
        mask1_pil = Image.open(mask_path1)
        mask1 = np.array(mask1_pil)
        unique_classes1 = np.unique(mask1)
        print(f"Unique classes in '{mask_filename1}': {unique_classes1}")
    except Exception as e:
        print(f"Error loading or processing '{mask_filename1}': {e}")
        mask1 = None
else:
    print(f"No suitable mask files found in: {mask_dir1}")
    mask1 = None

# --- Load the second mask ---
mask_dir2 = '../pathology-datasets/DRSK/image_patches_512_20x/masks'
file_list2 = os.listdir(mask_dir2)
mask_files2 = [f for f in file_list2 if f.endswith('.png')]
if mask_files2:
    mask_filename2 = random.choice(mask_files2)
    mask_path2 = os.path.join(mask_dir2, mask_filename2)
    try:
        mask2_pil = Image.open(mask_path2)
        mask2 = np.array(mask2_pil)
        unique_classes2 = np.unique(mask2)
        print(f"Unique classes in '{mask_filename2}': {unique_classes2}")
    except Exception as e:
        print(f"Error loading or processing '{mask_filename2}': {e}")
        mask2 = None
else:
    print(f"No mask files found in: {mask_dir2}")
    mask2 = None

# --- Analyze the differences ---
print("\n--- Analyzing Differences Between the Masks ---")

if mask1 is not None and mask2 is not None:
    # 1. Shape
    print(f"Shape of mask from '{mask_dir1}': {mask1.shape}")
    print(f"Shape of mask from '{mask_dir2}': {mask2.shape}")
    if mask1.shape != mask2.shape:
        print("-> The masks have different shapes (dimensions).")
    else:
        print("-> The masks have the same shape.")

    # 2. Data Type
    print(f"Data type of mask from '{mask_dir1}': {mask1.dtype}")
    print(f"Data type of mask from '{mask_dir2}': {mask2.dtype}")
    if mask1.dtype != mask2.dtype:
        print("-> The masks have different data types (e.g., uint8, int32).")
    else:
        print("-> The masks have the same data type.")

    # 3. Pixel Value Range and Distribution (beyond unique classes)
    print(f"Min/Max pixel value in mask from '{mask_dir1}': {mask1.min()}/{mask1.max()}")
    print(f"Min/Max pixel value in mask from '{mask_dir2}': {mask2.min()}/{mask2.max()}")

    # Visualize pixel value distributions (histograms)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(mask1.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Pixel Value Distribution - {os.path.basename(mask_path1)}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(mask2.flatten(), bins=50, color='lightcoral', edgecolor='black')
    plt.title(f'Pixel Value Distribution - {os.path.basename(mask_path2)}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # 4. Number of Channels (if applicable)
    if mask1.ndim == 3 and mask2.ndim == 3:
        print(f"Number of channels in mask from '{mask_dir1}': {mask1.shape[2]}")
        print(f"Number of channels in mask from '{mask_dir2}': {mask2.shape[2]}")
        if mask1.shape[2] != mask2.shape[2]:
            print("-> The masks have a different number of channels.")
        else:
            print("-> The masks have the same number of channels.")
    elif mask1.ndim == 3 or mask2.ndim == 3:
        print("-> The masks have a different number of dimensions, one might be grayscale and the other color.")
    else:
        print("-> Both masks appear to be grayscale (2D).")

    # 5. Visual Inspection
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask1)
    plt.title(os.path.basename(mask_path1))
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(mask2)
    plt.title(os.path.basename(mask_path2))
    plt.colorbar()
    plt.tight_layout()
    plt.show()

else:
    print("Could not load both masks for comparison.")
# %%
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Load the first image ---
image_dir1 = 'images/inpainting/'
file_list1 = os.listdir(image_dir1)
image_files1 = [f for f in file_list1 if f.endswith('.jpg')]
if image_files1:
    image_filename1 = random.choice(image_files1)
    image_path1 = os.path.join(image_dir1, image_filename1)
    try:
        image1_pil = Image.open(image_path1)
        image1 = np.array(image1_pil)
        print(f"Loaded image from '{image_filename1}'")
    except Exception as e:
        print(f"Error loading or processing '{image_filename1}': {e}")
        image1 = None
else:
    print(f"No suitable image files found in: {image_dir1}")
    image1 = None

# --- Load the second image ---
image_dir2 = '../pathology-datasets/DRSK/image_patches_512_20x/images'
file_list2 = os.listdir(image_dir2)
image_files2 = [f for f in file_list2 if f.endswith('.png')]
if image_files2:
    image_filename2 = random.choice(image_files2)
    image_path2 = os.path.join(image_dir2, image_filename2)
    try:
        image2_pil = Image.open(image_path2)
        image2 = np.array(image2_pil)
        print(f"Loaded image from '{image_filename2}'")
    except Exception as e:
        print(f"Error loading or processing '{image_filename2}': {e}")
        image2 = None
else:
    print(f"No suitable image files found in: {image_dir2}")
    image2 = None

# --- Analyze the differences ---
print("\n--- Analyzing Differences Between the Images ---")

if image1 is not None and image2 is not None:
    # 1. Shape
    print(f"Shape of image from '{image_dir1}': {image1.shape}")
    print(f"Shape of image from '{image_dir2}': {image2.shape}")
    if image1.shape != image2.shape:
        print("-> The images have different shapes (dimensions).")
    else:
        print("-> The images have the same shape.")

    # 2. Data Type
    print(f"Data type of image from '{image_dir1}': {image1.dtype}")
    print(f"Data type of image from '{image_dir2}': {image2.dtype}")
    if image1.dtype != image2.dtype:
        print("-> The images have different data types (e.g., uint8, float32).")
    else:
        print("-> The images have the same data type.")

    # 3. Pixel Value Range and Distribution
    print(f"Min/Max pixel value in image from '{image_dir1}': {image1.min()}/{image1.max()}")
    print(f"Min/Max pixel value in image from '{image_dir2}': {image2.min()}/{image2.max()}")

    # Visualize pixel value distributions (histograms) for each channel
    num_channels1 = image1.shape[2] if image1.ndim == 3 else 1
    num_channels2 = image2.shape[2] if image2.ndim == 3 else 1

    plt.figure(figsize=(14, 5))

    for i in range(num_channels1):
        plt.subplot(2, num_channels1, i + 1)
        if image1.ndim == 3:
            plt.hist(image1[:, :, i].flatten(), bins=50, color=f'C{i}', edgecolor='black', alpha=0.7)
        else:
            plt.hist(image1.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'{os.path.basename(image_path1)} - Channel {i+1 if image1.ndim == 3 else ""}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    for i in range(num_channels2):
        plt.subplot(2, num_channels2, num_channels1 + i + 1)
        if image2.ndim == 3:
            plt.hist(image2[:, :, i].flatten(), bins=50, color=f'C{i}', edgecolor='black', alpha=0.7)
        else:
            plt.hist(image2.flatten(), bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title(f'{os.path.basename(image_path2)} - Channel {i+1 if image2.ndim == 3 else ""}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # 4. Number of Channels
    print(f"Number of dimensions for image from '{image_dir1}': {image1.ndim}")
    print(f"Number of dimensions for image from '{image_dir2}': {image2.ndim}")
    if image1.ndim == 3 and image2.ndim == 3:
        print(f"Number of channels in image from '{image_dir1}': {image1.shape[2]}")
        print(f"Number of channels in image from '{image_dir2}': {image2.shape[2]}")
        if image1.shape[2] != image2.shape[2]:
            print("-> The images have a different number of channels.")
        else:
            print("-> The images have the same number of channels.")
    elif image1.ndim != image2.ndim:
        print("-> The images have a different number of dimensions, one might be grayscale and the other color.")
    else:
        print("-> Both images appear to be grayscale (2D).")

    # 5. Visual Inspection
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(os.path.basename(image_path1))

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title(os.path.basename(image_path2))
    plt.tight_layout()
    plt.show()

else:
    print("Could not load both images for comparison.")
# %%
from types import SimpleNamespace
import json
import yaml

def dict_to_namespace(d):
    json_str = json.dumps(d)
    return json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))

# Test YAML with different nesting levels
test_yaml = """
level1:
  level2:
    level3:
      level4:
        level5:
          value: "deep"
          array: [1,2,3]
          nested:
            even_deeper: true
simple: "top"
"""

def test_nesting():
    config = yaml.safe_load(test_yaml)
    ns = dict_to_namespace(config)
    
    # Test access at different levels
    print(ns.simple)  # top level
    print(ns.level1.level2.level3.level4.level5.value)  # deep nesting
    print(ns.level1.level2.level3.level4.level5.nested.even_deeper)  # deepest
    print(ns.level1.level2.level3.level4.level5.array)  # array access

if __name__ == "__main__":
    test_nesting()
# %%
import pandas as pd
import matplotlib.pyplot as plt
from shapely.wkt import loads
from shapely.geometry import MultiPolygon, Polygon

def visualize_tissue_patch(polygon_str):
    # Convert WKT string to MultiPolygon
    multi_polygon = loads(polygon_str)
    
    # Create new figure
    plt.figure(figsize=(10, 10))
    
    # Handle both MultiPolygon and single Polygon cases
    if isinstance(multi_polygon, MultiPolygon):
        polygons = list(multi_polygon.geoms)
    else:
        polygons = [multi_polygon]
    
    # Plot each polygon
    for polygon in polygons:
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'b-')
        plt.fill(x, y, alpha=0.3)
        
        # Plot any interior rings (holes)
        for interior in polygon.interiors:
            xi, yi = interior.xy
            plt.plot(xi, yi, 'r-')
    
    plt.axis('equal')
    plt.title('Tissue Region Boundary')
    plt.grid(True)
    plt.show()

# Read CSV and visualize first patch
df = pd.read_csv('../aida_drsk_512_patches_otzu.csv')
visualize_tissue_patch(df.iloc[0]['polygon_str'])
# %%
import cv2
import numpy as np

# Load mask image
mask_path = './logs/model_init_dataset/masks-1.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Print characteristics
print(f"Shape: {mask.shape}")
print(f"Data type: {mask.dtype}")
print(f"Min value: {mask.min()}")
print(f"Max value: {mask.max()}")
print(f"Unique values: {np.unique(mask)}")
print(f"Mean value: {mask.mean():.2f}")
print(f"Standard deviation: {mask.std():.2f}")
print(f"Number of non-zero pixels: {np.count_nonzero(mask)}")
print(f"Percentage of non-zero pixels: {(np.count_nonzero(mask) / mask.size * 100):.2f}%")

# Optional: Display histogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.hist(mask.ravel(), bins=256)
plt.title('Mask Histogram')
plt.show()
# %%
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_mask():
    # Find mask files
    mask_dir = "../pathology-datasets/DRSK/init_dataset/dm-training-data"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
    
    if not mask_files:
        print("No mask files found!")
        return
    
    # Select random mask
    random_mask = random.choice(mask_files)
    mask_path = os.path.join(mask_dir, random_mask)
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Print characteristics
    print(f"Selected mask: {random_mask}")
    print(f"Shape: {mask.shape}")
    print(f"Data type: {mask.dtype}")
    print(f"Min/Max values: {mask.min()}/{mask.max()}")
    print(f"Unique values: {np.unique(mask)}")
    print(f"Mean ± std: {mask.mean():.2f} ± {mask.std():.2f}")
    print(f"Non-zero pixels: {np.count_nonzero(mask)}")
    print(f"Non-zero percentage: {(np.count_nonzero(mask)/mask.size*100):.2f}%")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask Image')
    
    plt.subplot(122)
    plt.hist(mask.ravel(), bins=256)
    plt.title('Value Distribution')
    
    plt.tight_layout()
    plt.show()


analyze_mask()
# %%
