from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def training_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def evaluation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def imagefolder(root: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    if not root.is_dir(): raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    dataset = datasets.ImageFolder(root, transform=transform)
    if len(dataset.classes) < 2: raise ValueError(f"Expected at least two class folders in {root}")
    return dataset


def make_train_loader(dataset: datasets.ImageFolder, batch_size: int, workers: int) -> DataLoader:
    targets = torch.tensor(dataset.targets); counts = torch.bincount(targets)
    sampler = WeightedRandomSampler((1.0 / counts.float())[targets], len(targets), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def make_eval_loader(dataset: datasets.ImageFolder, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
