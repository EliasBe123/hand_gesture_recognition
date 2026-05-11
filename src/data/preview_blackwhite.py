import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# Config
DATA_DIR   = "data/hgr_cropped/train"
OUTPUT_DIR = "data/bw_samples"
IMG_SIZE   = 100
N_SAMPLES  = 10

class Threshold:
    def __call__(self, x):
        return (x > 0.5).float()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    Threshold(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def tensor_to_pil(tensor):
    """Convert normalised tensor back to PIL Image for saving."""
    # Undo normalisation: x = (tensor * std) + mean
    img = tensor * 0.5 + 0.5  # back to [0, 1]
    img = img.clamp(0, 1)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
    return Image.fromarray(img)

classes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(cls_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:N_SAMPLES]  # take first 10

    for fname in files:
        img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
        tensor = transform(img)
        bw_img = tensor_to_pil(tensor)
        bw_img.save(os.path.join(out_dir, fname))
        print(f"Saved {cls}/{fname}")

print(f"\nDone. Check {OUTPUT_DIR}/")