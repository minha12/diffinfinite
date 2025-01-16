import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import csv    
import yaml

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import random

from PIL import Image

join=os.path.join

root_path='.'
local_path='./images/path'

data_path=join(root_path,local_path)

def set_global_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

set_global_seed(10)


class ComposeState(T.Compose):
    def __init__(self, transforms):
        self.transforms = []
        self.mask_transforms = []

        for t in transforms:
            self.transforms.append(t)

        self.seed = None
        self.retain_state = False

    def __call__(self, x):
        if self.seed is not None:   # retain previous state
            set_global_seed(self.seed)
        if self.retain_state:    # save state for next call
            self.seed = self.seed or torch.seed()
            set_global_seed(self.seed)
        else:
            self.seed = None    # reset / ignore state

        if isinstance(x, (list, tuple)):
            return self.apply_sequence(x)
        else:
            return self.apply_img(x)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def apply_sequence(self, seq):
        self.retain_state=True
        seq = list(map(self, seq))
        self.retain_state=False
        return seq


def identity(x):
    return x

def get_augmentation(name='identity'):
    if name == 'identity':
        augmentation = identity

    return augmentation

class RandomRotate90():  # Note: not the same as T.RandomRotation(90)
    def __call__(self, x):
        x = x.rot90(random.randint(0, 3), dims=(-1, -2))
        return x

    def __repr__(self):
        return self.__class__.__name__


def create_dataset_csv(data_path: str, threshold=0.3, debug: bool = False):
    if debug:
        print(f"\n=== CSV Generation [create_dataset_csv()] ===")
        print(f"Scanning directory: {data_path}")
    
    files = [f for f in Path(data_path).iterdir() if f.name.endswith('jpg')]
    imgs, labels = [], []
    
    for f in files:
        f_mask = str(f).replace('.jpg', '_mask.png')
        mask = np.array(Image.open(f_mask))
        if np.mean(mask > 0) > threshold:
            imgs.append(f.stem)
            unique_labels = list(map(str, np.int32(np.unique(mask))))
            labels.append(' '.join(unique_labels))
            if debug:
                print(f"Image: {f.stem}, Labels: {unique_labels}")

    with open(Path(data_path, "dataset.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(imgs, labels))
    
    if debug:
        print(f"CSV file created at: {Path(data_path, 'dataset.csv')}")
        print(f"Total images: {len(imgs)}")


def remove_common_elements(train_set, test_set):
    # Convert train_set and test_set to sets to remove duplicates and enable set operations
    unique_train_set = set(train_set)
    unique_test_set = set(test_set)

    # Compute the unique elements in train_set that are not in test_set
    unique_elements = unique_train_set.difference(unique_test_set)

    # Convert the unique sets back to lists for compatibility with the original function signature
    unique_train_list = list(unique_train_set)
    unique_test_list = list(unique_test_set)

    return unique_train_list, unique_test_list


def get_class_counts(data_path: str = data_path):
    # Get a list of all files in the data directory
    files = os.listdir(data_path)

    # Extract all unique pixel values from the masks
    values = []
    for file in files:
        if file.endswith('.png'):
            mask = np.array(Image.open(os.path.join(data_path, file)))
            values += np.unique(mask).tolist()

    # Map class label 9 to 17 and shift all labels >= 9 down by one
    values = [v if v < 9 else v - 1 if v > 9 else 17 for v in values]

    # Count the number of instances of each class
    class_names = get_class_names(data_path) 
    counts = np.zeros(len(class_names))
    for v in values:
        counts[v] += 1

    return counts


def get_label_map_filename(config_file: str) -> str:
    """Get the appropriate label map filename based on config."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            num_classes = config['unet']['num_classes']
    except (FileNotFoundError, KeyError):
        num_classes = 5
    
    return f'label_map_{num_classes}.yml'

def get_class_names(data_path: str = data_path, config_file: str = None, debug: bool = False):
    if config_file:
        label_map_file = get_label_map_filename(config_file)
    else:
        label_map_file = 'label_map_5.yml'  # fallback default
    
    if debug:
        print(f"\n=== Class Name Loading [get_class_names()] ===")
        print(f"Loading label map file: {label_map_file}")

    try:
        with open(os.path.join(data_path, label_map_file), 'r') as file:
            contents = yaml.safe_load(file)
    except FileNotFoundError:
        with open(os.path.join(data_path, 'label_map_5.yml'), 'r') as file:
            contents = yaml.safe_load(file)

    class_dict = contents['classes']['class_to_int']
    class_names = [None] * len(class_dict)
    for name, idx in class_dict.items():
        if idx != 9:
            class_names[idx - (idx > 9)] = name

    if debug:
        print(f"Class names: {class_names}")

    return class_names


def dataset_to_dict(data_path: str = data_path):
    # Get the list of class names present in the dataset
    class_names = get_class_names(data_path)
    # Get the number of classes
    num_classes = len(class_names)+1
    # Create an empty dictionary for each class, to store the images belonging to that class
    subsets = {i: [] for i in range(num_classes-1)}

    # Read the dataset CSV file
    with open(join(data_path, 'dataset.csv'), 'r') as file:
        # Use a CSV reader to iterate over the rows of the file
        reader = csv.reader(file)
        for row in reader:
            # Extract the image name and label string from the row
            img, labels_str = row[0], row[1]
            # Convert the label string to a list of integers
            labels = [int(l) for l in labels_str.split()]
            # Iterate over the labels of the image
            for label in labels:
                # Fix necrosis = 9 -> 2, clamp >4 to 4
                if label == 9:
                    label = 2
                elif label > 4:
                    label = 4
                # Add the image to the subset dictionary for the corresponding class
                subsets[label].append(img)
    return subsets


def split_dataset(data_path: str = data_path, train_size: float = 0.9, config_file: str = None, debug: bool = False):
    if debug:
        print(f"\n=== Dataset Splitting [split_dataset()] ===")
        print(f"Data path: {data_path}")
        print(f"Train size: {train_size}")

    subset_dict = dataset_to_dict(data_path)
    classes = get_class_names(data_path, config_file=config_file)
    num_classes = len(classes)
    subclasses = list(range(num_classes))

    train_set, test_set = {}, {}
    for i in subclasses:
        train_set[i], test_set[i] = [], []

    for i in range(num_classes):
        class_set = subset_dict[i]
        class_counts = len(class_set)

        if class_counts > 1:
            train_index, test_index = train_test_split(
                torch.linspace(0, class_counts - 1, class_counts), 
                train_size=train_size)
            train_set[i] += [class_set[j] for j in train_index.int()]
            test_set[i] += [class_set[j] for j in test_index.int()]

    if debug:
        print(f"Train set summary: {[len(train_set[i]) for i in range(num_classes)]}")
        print(f"Test set summary: {[len(test_set[i]) for i in range(num_classes)]}")

    return train_set, test_set


def add_unconditional(data_path: str, data_dict: str, no_check=False):
    files = [f for f in Path(data_path).iterdir()]

    for f in files:
        if os.path.isdir(f):
            data_dict = add_unconditional(f,data_dict)

    if not no_check:
        active_images=[]
        for key in data_dict:
            active_images+=data_dict[key]
        for f in files:
            if f.name.endswith('jpg') and f not in active_images:
                data_dict[0].append(f.stem)
    else:
        for f in files:
            if f.name.endswith('jpg'):
                data_dict[0].append(f.stem)

    return data_dict

def import_dataset(
        data_path: str = data_path,
        batch_size: int = 32,
        num_workers: int = 0,
        subclasses: list = None,
        cond_drop_prob: float = 0.5,
        threshold: float = 0.,
        force: bool = False,
        transform=None,
        config_file: str = None,
        extra_data_path: str = None,
        debug: bool = False  # Add debug parameter
):
    # Generate the dataset CSV file if it does not exist
    if not os.path.exists(join(data_path, "dataset.csv")) or force:
        create_dataset_csv(data_path=data_path, threshold=threshold, debug=debug)

    if debug:
        print(f"\n=== Dataset Import [import_dataset()] ===")
        print(f"Data path: {data_path}")
        print(f"Batch size: {batch_size}")
        print(f"Extra data path: {extra_data_path}")

    train_dict, test_dict = split_dataset(data_path, train_size=0.9, config_file=config_file, debug=debug)

    # Create the train and test datasets
    train_set = DatasetLung(data_path=data_path, data_dict=train_dict, 
                            subclasses=subclasses, cond_drop_prob=cond_drop_prob,
                            transform=transform,
                            extra_unknown_data_path=[extra_data_path],
                            debug=debug)  # Pass debug flag
    test_set = DatasetLung(data_path=data_path, data_dict=test_dict, 
                           subclasses=subclasses, cond_drop_prob=1.,
                           transform=transform,
                           extra_unknown_data_path=[extra_data_path],
                           debug=debug)  # Pass debug flag

    # Create the train and test data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

class DatasetLung(Dataset):
    def __init__(self,
            data_path: str,
            data_dict: dict,
            subclasses: list = None,
            cond_drop_prob: float = 0.5,
            extra_unknown_data_path: list = [],
            transform = None,
            debug: bool = False):  # Add debug parameter

        self.debug = debug
        if self.debug:
            print(f"\n=== Dataset Init [DatasetLung.__init__()] ===")
            for cls, samples in data_dict.items():
                print(f"Class {cls}: {len(samples)} samples")

        for extra in extra_unknown_data_path:
            data_dict = add_unconditional(data_path=extra, 
                                          data_dict=data_dict, no_check=True)

        N_classes = len(data_dict)

        self.data_path = data_path
        self.extra = extra_unknown_data_path
        self.data_dict = data_dict
        self.subclasses = subclasses
        self.cutoffs = self._cutoffs(subclasses, cond_drop_prob)
        self.N_classes = N_classes
        self.transform = transform

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDataset[{self.__len__()}]"
        for n in range(self.N_classes):
            rep += f'\nClass {n} has N samples: {len(self.data_dict[n])}\t'
        return rep

    def __len__(self):
        counts = 0
        for i in range(len(self.data_dict)):
            counts += len(self.data_dict[i])
        return counts

    def _cutoffs(self, subclasses, cond_drop_prob=0.5):
        # Handle None or empty subclasses
        if not subclasses:
            return torch.tensor([1.0])  # Single cutoff
        probs = [cond_drop_prob / (len(subclasses) + 1) for n in range(len(subclasses) + 1)]
        probs.insert(0, 1. - cond_drop_prob)
        return torch.Tensor(probs).cumsum(dim=0)

    def unbalanced_data(self):
        # generate a random number in [0,1)
        rand_num = torch.rand(1)
        # find the index of the interval that the random number falls into
        index = torch.sum(rand_num >= self.cutoffs)
        self.tmp_index = index
        # map the index to the appropriate tensor value using PyTorch indexing
        oneclass_data = self.data_dict[index.item()]
        # generate a random number in [0,1)
        rand_num = (torch.rand(1) * len(oneclass_data)).int()
        # extract random img from the selected class
        core_path = oneclass_data[rand_num]
        # return img and mask path
        img_path = join(self.data_path, core_path + '.jpg')
        mask_path = join(self.data_path, core_path + '_mask.png')

        if not os.path.exists(img_path):
            for extra in self.extra:
                extra_path = join(extra, core_path + '.jpg')
                if os.path.exists(extra_path):
                    img_path = extra_path

        # load img and mask
        img = Image.open(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            h, w, c = np.array(img).shape
            mask = np.zeros((h, w, 1))

        if self.debug:
            print(f"\n=== Data Loading [DatasetLung.unbalanced_data()] ===")
            print(f"Current cutoff probabilities: {self.cutoffs}")
            print(f"Selected class index: {index.item()}")
            print(f"Loading image: {img_path}")
            print(f"Original image size: {img.size}")
            if os.path.exists(mask_path):
                print(f"Mask size: {mask.size}")
            else:
                print("No mask found - using zero mask")

        return img, mask

    def __getitem__(self, idx):
        img, mask = self.unbalanced_data()

        if self.transform is not None:
            img, mask = self.transform((img, mask))
            mask = (mask * 255).int()
            if self.debug and idx == 0:
                print("\n=== Transformation Results [DatasetLung.__getitem__()] ===")
                print(f"Image tensor shape: {img.shape}")
                print(f"Image value range: [{img.min():.2f}, {img.max():.2f}]")
                print(f"Mask tensor shape: {mask.shape}")
                print(f"Unique mask values: {torch.unique(mask)}")

        return img, mask
