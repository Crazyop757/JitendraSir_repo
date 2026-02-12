"""
preprocessing.py

- Loads images from the dataset directory
- Applies explicit resizing, cropping, normalization, and augmentation:
    - Resize(256), CenterCrop(224) for validation/test
    - Resize(256), RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(15°), ColorJitter for training
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)
- Supports reproducible splits for training, validation, and testing
- Handles stratified k-fold cross-validation
- All random seeds are set for reproducibility
"""

import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torchvision import transforms, datasets

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation for training
def get_train_transforms():
    """
    Training transforms:
    - Resize to 256
    - RandomResizedCrop to 224
    - RandomHorizontalFlip
    - RandomRotation (±15°)
    - ColorJitter (brightness, contrast, saturation, hue)
    - ToTensor
    - Normalize (ImageNet mean/std)
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# Preprocessing for validation/test
def get_val_transforms():
    """
    Validation/Test transforms:
    - Resize to 256
    - CenterCrop to 224
    - ToTensor
    - Normalize (ImageNet mean/std)
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# Dataset loader using ImageFolder
def get_datasets(data_dir, val_split=0.2, seed=42):
    set_seed(seed)
    full_dataset = datasets.ImageFolder(data_dir, transform=get_train_transforms())
    targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        stratify=targets,
        random_state=seed
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=get_val_transforms()), val_idx)
    return train_dataset, val_dataset, full_dataset.classes

# Stratified K-Fold split
def get_stratified_kfold(data_dir, n_splits=5, seed=42):
    set_seed(seed)
    full_dataset = datasets.ImageFolder(data_dir, transform=get_train_transforms())
    targets = [s[1] for s in full_dataset.samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(
            datasets.ImageFolder(data_dir, transform=get_val_transforms()), val_idx)
        yield fold, train_dataset, val_dataset, full_dataset.classes
