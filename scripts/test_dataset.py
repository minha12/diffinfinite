# test_dataset.py

import os
import shutil
import pytest
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Make sure to import all required symbols from your dataset.py
# Adjust the import paths based on your project structure:
# e.g. from src.dataset import import_dataset, ComposeState
from dataset import import_dataset, ComposeState


@pytest.fixture
def dummy_data(tmp_path):
    """
    Create a temporary directory with minimal dummy data (images + masks)
    to test the dataset logic. Each image is saved as a .jpg and the
    corresponding mask as a _mask.png with the same stem.
    """
    # Make a few dummy images/masks
    for i in range(3):
        # Create a random RGB image
        dummy_image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image_array, mode='RGB')
        pil_image.save(tmp_path / f"sample_{i}.jpg")

        # Create a random mask
        dummy_mask_array = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        pil_mask = Image.fromarray(dummy_mask_array, mode='L')
        pil_mask.save(tmp_path / f"sample_{i}_mask.png")

    return tmp_path


@pytest.mark.parametrize("resize_size", [(512, 512)])
def test_dataset_lung_resize(dummy_data, resize_size):
    """
    Test that the imported dataset can be constructed with a transform that
    resizes images and masks to the desired size (e.g. 512x512). Also print
    out data types, shapes, etc.
    """
    # 1. Define a transform pipeline that includes resizing to 512x512 and converting to tensors.
    #    ComposeState is the class from your dataset.py that combines transforms.
    transform = ComposeState([
        T.Resize(resize_size),
        T.ToTensor()  # Convert PIL images to torch tensors
    ])

    # 2. Call import_dataset, forcing it to use our dummy data path.
    #    - This will internally create a dataset.csv if not present.
    #    - We set force=True here just to ensure it doesn't rely on any existing CSV.
    train_loader, test_loader = import_dataset(
        data_path=str(dummy_data),
        batch_size=2,
        num_workers=0,
        subclasses=None,
        cond_drop_prob=0.5,
        threshold=0.0,
        force=True,           # Force creation of dataset.csv using the dummy files
        transform=transform,
        config_file=None
    )

    # 3. Fetch a single batch from the train_loader and print shapes/dtypes for debugging.
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        print("\n--- Train Loader Batch Debug Info ---")
        print(f"Image batch shape: {imgs.shape}, dtype: {imgs.dtype}")
        print(f"Mask batch shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Unique mask values: {torch.unique(masks)}")
        # Confirm the resize was applied correctly
        assert imgs.shape[-2:] == resize_size, f"Images not resized to {resize_size}."
        assert masks.shape[-2:] == resize_size, f"Masks not resized to {resize_size}."
        break  # Only test the first batch

    # 4. Do the same for the test_loader.
    for batch_idx, (imgs, masks) in enumerate(test_loader):
        print("\n--- Test Loader Batch Debug Info ---")
        print(f"Image batch shape: {imgs.shape}, dtype: {imgs.dtype}")
        print(f"Mask batch shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Unique mask values: {torch.unique(masks)}")
        # Confirm the resize was applied correctly
        assert imgs.shape[-2:] == resize_size, f"Images not resized to {resize_size}."
        assert masks.shape[-2:] == resize_size, f"Masks not resized to {resize_size}."
        break  # Only test the first batch

    print("\nTest completed successfully for dataset resizing and basic shape checks.")