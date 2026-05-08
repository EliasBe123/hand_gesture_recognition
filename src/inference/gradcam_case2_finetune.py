"""
Grad-CAM heatmap visualization for the fine-tuned CNN.

Shows which pixels of the input image most influenced the model's prediction
by overlaying a colored heatmap on top of the original (cropped) image.

Uses MediaPipe to detect & crop the hand (same pipeline as predict),
then computes Grad-CAM on the last conv layer (conv4).

Usage:
    python -m src.inference.gradcam_case2_finetune path/to/image.jpg
    python -m src.inference.gradcam_case2_finetune path/to/image.jpg --class A
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
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


class GradCAM:
    """Grad-CAM implementation using forward + backward hooks on a target conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output  # (B, C, H, W)

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]  # (B, C, H, W)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        outputs = self.model(input_tensor)             # (1, num_classes)
        probs = torch.softmax(outputs, dim=1)[0]

        if class_idx is None:
            class_idx = probs.argmax().item()

        # Backprop the score for the chosen class
        score = outputs[0, class_idx]
        score.backward()

        # Global-average-pool the gradients across H,W -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)    # (1, 1, H, W)
        cam = F.relu(cam)                                              # only positive contributions

        # Upsample to input image size (IMG_SIZE x IMG_SIZE)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize 0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, probs.detach().cpu()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def visualize(image_path, target_class=None):
    device = torch.device(DEVICE)
    model = HandGestureCNN_Case2().to(device)
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH_CASE2, map_location=device, weights_only=True))
    model.eval()

    # Load + crop
    img = Image.open(image_path).convert("RGB")
    crop = detect_hand_crop(img)
    if crop is None:
        print("No hand detected — running on the full image.")
        crop = img
    crop = crop.convert("L").convert("RGB")
    crop_resized = crop.resize((IMG_SIZE, IMG_SIZE))

    tensor = transform(crop).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    # Resolve target class
    class_idx = None
    if target_class is not None:
        if target_class not in CLASS_NAMES:
            raise ValueError(f"Unknown class '{target_class}'. Choose from {CLASS_NAMES}")
        class_idx = CLASS_NAMES.index(target_class)

    # Run Grad-CAM on the last conv layer
    cam_helper = GradCAM(model, target_layer=model.conv4)
    cam, used_idx, probs = cam_helper(tensor, class_idx=class_idx)
    cam_helper.close()

    label = CLASS_NAMES[used_idx]
    conf = probs[used_idx].item()
    pred_idx = probs.argmax().item()
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = probs[pred_idx].item()

    print(f"\nModel prediction:  {pred_label}  ({pred_conf:.1%})")
    print(f"Heatmap shows:     {label}  ({conf:.1%})")
    print("\nAll class scores:")
    for i, name in enumerate(CLASS_NAMES):
        marker = "  ← shown" if i == used_idx else ""
        print(f"  {name}: {probs[i].item():.1%}{marker}")

    # --- Plot: original | heatmap | overlay ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(crop_resized, cmap="gray")
    axes[0].set_title("Input (cropped)")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(crop_resized, cmap="gray")
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay ({label})")
    axes[2].axis("off")

    fig.suptitle(
        f"Grad-CAM  —  Predicted: {pred_label} ({pred_conf:.1%})  |  Heatmap for: {label}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for a hand-gesture image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--class", dest="target_class", type=str, default=None,
        help=f"Class to compute heatmap for (default: predicted class). Choices: {CLASS_NAMES}",
    )
    args = parser.parse_args()
    visualize(args.image_path, args.target_class)
