import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.utils.config_case2v2 import HGR_TRAIN_DIR, HGR_TEST_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT


class AdaptiveThreshold:
    """
    Replaces the old fixed Threshold(x > 0.5).
    Converts a (3, H, W) float tensor back to a numpy array,
    applies adaptive Gaussian thresholding, and returns a (3, H, W) float tensor
    so the rest of the pipeline (Normalize etc.) is unchanged.
    """
    def __call__(self, x):
        # x is a (3, H, W) float tensor in [0, 1] at this point (after ToTensor)
        # Convert to uint8 grayscale for OpenCV
        np_img = (x[0].numpy() * 255).astype(np.uint8)   # take first channel (all 3 are identical after Grayscale)

        # Optional blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(np_img, (5, 5), 0)

        # ── OPTION A: Adaptive Gaussian (recommended) ──────────────────────
        binary = cv2.adaptiveThreshold(
            blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,   # must be odd — try 7, 11, 15
            C=2             # try 1, 2, 3
        )

        # ── OPTION B: Otsu — comment out A and uncomment this to compare ───
        # _, binary = cv2.threshold(
        #     blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # )
        # ───────────────────────────────────────────────────────────────────

        # Optional: morphological closing to reconnect broken finger contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert back to float tensor in [0, 1] and repeat across 3 channels
        binary_float = torch.from_numpy(binary / 255.0).float()
        return binary_float.unsqueeze(0).repeat(3, 1, 1)   # → (3, H, W)


class ApplyTransform(torch.utils.data.Dataset):
    """A small wrapper to apply different transforms to different splits."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            AdaptiveThreshold(),                              # ← replaces Threshold()
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        AdaptiveThreshold(),                                  # ← replaces Threshold()
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloaders():
    # 1. Load the dataset WITHOUT transforms first
    base_dataset = datasets.ImageFolder(HGR_TRAIN_DIR)

    # 2. Split the raw indices
    val_size = int(len(base_dataset) * VAL_SPLIT)
    train_size = len(base_dataset) - val_size
    train_indices, val_indices = random_split(
        base_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 3. Wrap the indices with their specific transforms
    train_dataset = ApplyTransform(train_indices, transform=get_transforms(train=True))
    val_dataset   = ApplyTransform(val_indices,   transform=get_transforms(train=False))

    # 4. Test set always uses clean (False) transforms
    test_dataset = datasets.ImageFolder(HGR_TEST_DIR, transform=get_transforms(train=False))

    # 5. Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes: {base_dataset.classes}")
    print(f"Split Summary -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader