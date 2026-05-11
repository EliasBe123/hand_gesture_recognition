import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mediapipe as mp
import numpy as np
import cv2
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn_case2v2 import HandGestureCNN_Case2
from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE4, IMG_SIZE, NUM_CLASSES, DEVICE

CLASS_NAMES = ["A", "F", "L", "Y"]


class AdaptiveThreshold:
    """
    Identical to the one in loader_case4_blackwhite.py — must stay in sync.
    Expects a (3, H, W) float tensor, returns a (3, H, W) float tensor.
    """
    def __call__(self, x):
        np_img = (x[0].numpy() * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(np_img, (5, 5), 0)

        # ── OPTION A: Adaptive Gaussian ─────────────────────────────────────
        binary = cv2.adaptiveThreshold(
            blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # ── OPTION B: Otsu — comment out A and uncomment to compare ─────────
        # _, binary = cv2.threshold(
        #     blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # )
        # ────────────────────────────────────────────────────────────────────

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        binary_float = torch.from_numpy(binary / 255.0).float()
        return binary_float.unsqueeze(0).repeat(3, 1, 1)


# Transform — must exactly match get_transforms(train=False) in the loader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    AdaptiveThreshold(),                                  # ← replaces the fixed threshold lambda
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def detect_hands(image_rgb):
    """Use MediaPipe to detect hands and return bounding boxes in pixel coords."""
    mp_hands = mp.solutions.hands
    h, w, _ = image_rgb.shape
    hands_result = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                pad = 0.05
                x_min = max(0, int((min(x_coords) - pad) * w))
                y_min = max(0, int((min(y_coords) - pad) * h))
                x_max = min(w, int((max(x_coords) + pad) * w))
                y_max = min(h, int((max(y_coords) + pad) * h))

                width  = x_max - x_min
                height = y_max - y_min
                max_side = max(width, height)

                center_x = x_min + width  // 2
                center_y = y_min + height // 2

                half_side = int(max_side / 2)
                x_min_sq = max(0, center_x - half_side)
                y_min_sq = max(0, center_y - half_side)
                x_max_sq = min(w, center_x + half_side)
                y_max_sq = min(h, center_y + half_side)

                hands_result.append((x_min_sq, y_min_sq, x_max_sq, y_max_sq))

    return hands_result


def apply_adaptive_threshold_for_display(pil_crop):
    """
    Runs the same adaptive threshold pipeline as AdaptiveThreshold()
    but returns a numpy array for display purposes instead of a tensor.
    This ensures the B&W preview in the plot matches exactly what the model sees.
    """
    gray = np.array(pil_crop.convert("L"))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Keep in sync with AdaptiveThreshold above
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def classify_crop(model, crop_img, device):
    """Classify a cropped hand image using the CNN."""
    tensor = transform(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        idx = probs.argmax().item()
    return CLASS_NAMES[idx], probs[idx].item(), probs


def predict(image_path):
    device = torch.device(DEVICE)
    model = HandGestureCNN_Case2()
    model.load_state_dict(torch.load(BEST_MODEL_PATH_CASE4, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    img    = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    boxes = detect_hands(img_np)
    print(f"Detected {len(boxes)} hand(s)")
    if not boxes:
        print(f"\nNo hands detected in {image_path}")
        plt.imshow(img)
        plt.title("No hands detected")
        plt.axis("off")
        plt.show()
        return

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax.imshow(img_np)

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        crop = img.crop((x_min, y_min, x_max, y_max))
        crop = crop.convert("L").convert("RGB")
        label, conf, probs = classify_crop(model, crop, device)

        print(f"\nHand {i + 1}:")
        print(f"  Position: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        print(f"  Gesture:  {label}  ({conf:.1%})")
        top = probs.topk(min(3, NUM_CLASSES))
        for prob, idx in zip(top.values, top.indices):
            print(f"    {CLASS_NAMES[idx]:<10} {prob.item():.1%}")

        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 8,
            f"{label} ({conf:.0%})",
            color='lime', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.6, pad=2)
        )

        # B&W preview now shows exactly what the model received
        binary_display = apply_adaptive_threshold_for_display(crop)
        ax2.imshow(binary_display, cmap='gray')
        ax2.set_title(f"Adaptive threshold crop — {label} ({conf:.0%})")
        ax2.axis("off")

    ax.set_title(f"Hand Gesture Detection — {image_path}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and classify hand gestures in an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    predict(args.image_path)