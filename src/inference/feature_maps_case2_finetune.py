"""
Visualize intermediate feature maps (activations) from each conv layer
of the fine-tuned CNN for a given input image.

Uses MediaPipe to detect & crop the hand (same pipeline as predict),
then runs the crop through the model and shows what each conv layer "sees".

Usage:
    python -m src.inference.feature_maps_case2_finetune path/to/image.jpg
"""

import os
import sys
import argparse
import math

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.cnn_case2v2 import HandGestureCNN_Case2
from src.utils.config_case2v2 import FINETUNED_MODEL_PATH_CASE2, IMG_SIZE, DEVICE

CLASS_NAMES = ["A", "F", "L", "Y"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Max channels to display per layer (feature maps can have 256+ channels)
MAX_CHANNELS_DISPLAY = 16


def detect_hand_crop(image_pil):
    """Detect the first hand with MediaPipe and return a square cropped PIL image."""
    img_np = np.array(image_pil)
    h, w, _ = img_np.shape

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(img_np)
        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]

        pad = 0.05
        x_min = max(0, int((min(x_coords) - pad) * w))
        y_min = max(0, int((min(y_coords) - pad) * h))
        x_max = min(w, int((max(x_coords) + pad) * w))
        y_max = min(h, int((max(y_coords) + pad) * h))

        # Square box
        width = x_max - x_min
        height = y_max - y_min
        max_side = max(width, height)
        cx = x_min + width // 2
        cy = y_min + height // 2
        half = max_side // 2

        x_min_sq = max(0, cx - half)
        y_min_sq = max(0, cy - half)
        x_max_sq = min(w, cx + half)
        y_max_sq = min(h, cy + half)

        return image_pil.crop((x_min_sq, y_min_sq, x_max_sq, y_max_sq))


def register_hooks(model):
    """Register forward hooks on each conv layer; return dict {name: activation tensor}."""
    activations = {}

    def make_hook(name):
        def hook(_module, _input, output):
            activations[name] = output.detach().cpu()
        return hook

    # Hook every named Conv2d layer
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    return activations, handles


def plot_feature_maps(activations, predicted_label, confidence):
    """Plot a grid of feature maps for each conv layer."""
    layer_names = list(activations.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 3 * n_layers))
    if n_layers == 1:
        axes = [axes]

    for ax_row, name in zip(axes, layer_names):
        fmap = activations[name][0]  # (C, H, W) — drop batch dim
        n_channels = min(fmap.shape[0], MAX_CHANNELS_DISPLAY)
        cols = 8
        rows = math.ceil(n_channels / cols)

        # Build a single mosaic image for this layer
        ch_h, ch_w = fmap.shape[1], fmap.shape[2]
        mosaic = np.ones((rows * ch_h, cols * ch_w)) * fmap.min().item()
        for i in range(n_channels):
            r, c = divmod(i, cols)
            mosaic[r * ch_h:(r + 1) * ch_h, c * ch_w:(c + 1) * ch_w] = fmap[i].numpy()

        ax_row.imshow(mosaic, cmap="viridis")
        ax_row.set_title(f"{name}  —  shape {tuple(fmap.shape)}  (showing first {n_channels} channels)")
        ax_row.axis("off")

    fig.suptitle(f"Feature Maps  —  Predicted: {predicted_label}  ({confidence:.1%})", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize(image_path):
    device = torch.device(DEVICE)
    model = HandGestureCNN_Case2().to(device)
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH_CASE2, map_location=device, weights_only=True))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    crop = detect_hand_crop(img)
    if crop is None:
        print("No hand detected — running on the full image.")
        crop = img
    crop = crop.convert("L").convert("RGB")  # match training pipeline

    tensor = transform(crop).unsqueeze(0).to(device)

    activations, handles = register_hooks(model)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    # Clean up hooks
    for h in handles:
        h.remove()

    label = CLASS_NAMES[pred_idx]
    conf = probs[pred_idx].item()
    print(f"\nPredicted: {label}  ({conf:.1%})")
    print(f"Captured activations from {len(activations)} conv layers:")
    for name, act in activations.items():
        print(f"  {name}: {tuple(act.shape)}")

    # Show the input crop separately
    plt.figure(figsize=(3, 3))
    plt.imshow(crop, cmap="gray")
    plt.title(f"Input (cropped)\nPredicted: {label} ({conf:.1%})")
    plt.axis("off")
    plt.show()

    plot_feature_maps(activations, label, conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CNN feature maps for an input image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    visualize(args.image_path)
