import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from src.utils.config_case2v2 import HGR_TRAIN_DIR, HGR_TEST_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT


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


class PairedGestureDataset(Dataset):
    """Loads image pairs grouped by shared image ID (e.g. 1007_image503 + 1008_image503)."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.pairs = []  # list of ((path1, path2), class_idx)
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            # Group files by image ID (part after underscore)
            groups = defaultdict(list)
            for fname in os.listdir(cls_dir):
                if not fname.endswith(".png") and not fname.endswith(".jpg"):
                    continue
                img_id = fname.split("_", 1)[1]  # e.g. "image503.png"
                groups[img_id].append(os.path.join(cls_dir, fname))

            for img_id, paths in groups.items():
                if len(paths) == 2:
                    paths.sort()  # deterministic ordering by filename
                    self.pairs.append((paths, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (path1, path2), label = self.pairs[idx]
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # Stack into (2, C, H, W) sequence
        pair_tensor = torch.stack([img1, img2], dim=0)
        return pair_tensor, label


def get_dataloaders():
    full_dataset = PairedGestureDataset(HGR_TRAIN_DIR, transform=get_transforms(train=True))
    
    # Use same class mapping as train set so indices are consistent
    test_dataset = SingleGestureDataset(
        HGR_TEST_DIR,
        transform=get_transforms(train=False),
        class_to_idx=full_dataset.class_to_idx
    )

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes: {full_dataset.classes}")
    print(f"Pairs — Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

class SingleGestureDataset(Dataset):
    """Loads single images for test set (no pairing)."""
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = class_to_idx if class_to_idx else {c: i for i, c in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label