import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils.config_case2v2 import HGR_TRAIN_DIR, HGR_TEST_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT


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
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
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
    val_dataset = ApplyTransform(val_indices, transform=get_transforms(train=False))
     
    # 4. Test set always uses clean (False) transforms
    test_dataset = datasets.ImageFolder(HGR_TEST_DIR, transform=get_transforms(train=False))

    # 5. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Classes: {base_dataset.classes}")
    print(f"Split Summary -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader