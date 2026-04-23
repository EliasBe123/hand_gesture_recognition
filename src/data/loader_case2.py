import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils.config_case2 import HGR_TRAIN_DIR, HGR_TEST_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloaders():
    full_train = datasets.ImageFolder(HGR_TRAIN_DIR, transform=get_transforms(train=True))
    test_dataset = datasets.ImageFolder(HGR_TEST_DIR, transform=get_transforms(train=False))

    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes: {full_train.classes}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
