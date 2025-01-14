import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert_images(
    input_dir: str = '../pathology-datasets/DRSK/init_dataset/images',
    output_dir: str = '../pathology-datasets/DRSK/init_dataset/dm-training-data',
    target_size: tuple = (512, 512)
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files in the input directory
    input_files = list(Path(input_dir).glob('*.png'))
    
    print(f"Found {len(input_files)} PNG files to convert")
    
    for input_file in tqdm(input_files, desc="Converting images"):
        # Load the image
        img = Image.open(input_file)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Check if image has 4 channels
        if img_array.shape[-1] == 4:
            # Take only the RGB channels (drop alpha channel)
            img_array = img_array[..., :3]
        
        # Convert back to PIL Image
        img = Image.fromarray(img_array)
        
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if needed
        if img.size != target_size:
            print(f"Resizing {input_file.name} from {img.size} to {target_size}")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create output filename (change extension to .jpg)
        output_file = Path(output_dir) / (input_file.stem + '.jpg')
        
        # Save as JPG with maximum effective quality
        img.save(output_file, 'JPEG', quality=100)
        
        # Verify the saved image (optional verification step)
        saved_img = Image.open(output_file)
        orig_array = np.array(img)
        saved_array = np.array(saved_img)
        max_diff = np.abs(orig_array - saved_array).max()
        mean_diff = np.abs(orig_array - saved_array).mean()
        
        # if max_diff > 5:  # threshold for acceptable difference
        #     print(f"Warning: High difference detected for {input_file.name}")
        #     print(f"Max difference: {max_diff}, Mean difference: {mean_diff}")

if __name__ == '__main__':
    convert_images()
