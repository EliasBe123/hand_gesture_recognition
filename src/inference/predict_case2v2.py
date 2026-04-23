import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mediapipe as mp
import numpy as np
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn_case2v2 import HandGestureCNN_Case2
from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE2, IMG_SIZE, NUM_CLASSES, DEVICE

# Class names — matches ImageFolder alphabetical ordering of A, F, L, Y
CLASS_NAMES = ["A", "F", "L", "Y"]

# Transform (matches training — RGB, no grayscale)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def detect_hands(image_rgb):
    """Use MediaPipe to detect hands and return bounding boxes (x, y, w, h) in pixel coords."""
    mp_hands = mp.solutions.hands
    h, w, _ = image_rgb.shape
    hands_result = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                # Bounding box with padding
                pad = 0.05
                x_min = max(0, int((min(x_coords) - pad) * w))
                y_min = max(0, int((min(y_coords) - pad) * h))
                x_max = min(w, int((max(x_coords) + pad) * w))
                y_max = min(h, int((max(y_coords) + pad) * h))

                hands_result.append((x_min, y_min, x_max, y_max))

    return hands_result


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
    model.load_state_dict(torch.load(BEST_MODEL_PATH_CASE2, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Detect hands
    boxes = detect_hands(img_np)
    if not boxes:
        print(f"\nNo hands detected in {image_path}")
        plt.imshow(img)
        plt.title("No hands detected")
        plt.axis("off")
        plt.show()
        return

    # Plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_np)

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        # Crop and classify
        crop = img.crop((x_min, y_min, x_max, y_max))
        label, conf, probs = classify_crop(model, crop, device)

        # Print results
        print(f"\nHand {i + 1}:")
        print(f"  Position: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        print(f"  Gesture:  {label}  ({conf:.1%})")
        top = probs.topk(min(3, NUM_CLASSES))
        for prob, idx in zip(top.values, top.indices):
            print(f"    {CLASS_NAMES[idx]:<10} {prob.item():.1%}")

        # Draw bounding box
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

    ax.set_title(f"Hand Gesture Detection — {image_path}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and classify hand gestures in an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    predict(args.image_path)
