import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn_case2v2 import HandGestureCNN_Case2
from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE2, IMG_SIZE, NUM_CLASSES, DEVICE

CLASS_NAMES = ["A", "F", "L", "Y"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ── Detection ────────────────────────────────────

def detect_hands(image_rgb):
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
                max_side  = max(width, height)
                center_x  = x_min + width  // 2
                center_y  = y_min + height // 2
                half_side = int(max_side / 2)

                x_min_sq = max(0, center_x - half_side)
                y_min_sq = max(0, center_y - half_side)
                x_max_sq = min(w, center_x + half_side)
                y_max_sq = min(h, center_y + half_side)

                hands_result.append((x_min_sq, y_min_sq, x_max_sq, y_max_sq))

    return hands_result


def classify_crop(model, crop_img, device):
    tensor = transform(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        idx     = probs.argmax().item()
    return CLASS_NAMES[idx], probs[idx].item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(test_dir):
    device = torch.device(DEVICE)
    model  = HandGestureCNN_Case2()
    model.load_state_dict(torch.load(BEST_MODEL_PATH_CASE2, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    total           = 0
    correct         = 0
    no_detection    = 0   # MediaPipe found no hand at all
    wrong           = []

    # Per-class tracking for the confusion matrix
    # confusion[true][predicted] = count
    confusion = defaultdict(lambda: defaultdict(int))

    # Per-class totals (for per-class accuracy)
    class_total   = defaultdict(int)
    class_correct = defaultdict(int)

    test_path = Path(test_dir)
    image_extensions = ["*.jpg", "*.png", "*.jpeg"]

    # Count total images upfront for ETA
    total_images = 0
    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        if class_folder.name not in CLASS_NAMES:
            continue
        for ext in image_extensions:
            total_images += len(list(class_folder.glob(ext)))

    start_time = time.time()
    
    
    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        true_label = class_folder.name
        if true_label not in CLASS_NAMES:
            print(f"  Warning: folder '{true_label}' not in CLASS_NAMES, skipping.")
            continue

        images = []
        for ext in image_extensions:
            images.extend(class_folder.glob(ext))
        images = sorted(images)

        for img_path in images:
            img    = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            boxes  = detect_hands(img_np)

            total          += 1
            class_total[true_label] += 1
            
            
            ## Progress indicator (every 20 images) with ETA
            # if total % 20 == 0:
            #     print(f"  Processed {total} images...", end="\r")
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            remaining = (total_images - total) / rate if rate > 0 else 0
            mins, secs = divmod(int(remaining), 60)
            print(f"  {total}/{total_images} images  |  {rate:.1f} img/s  |  ETA: {mins}m {secs}s    ", end="\r")

            if not boxes:
                no_detection += 1
                # Count as wrong — model never got a chance to classify
                confusion[true_label]["NO_DETECTION"] += 1
                wrong.append({
                    "file":      str(img_path),
                    "true":      true_label,
                    "predicted": "NO_DETECTION",
                    "confidence": 0.0,
                    "reason":    "mediapipe_no_detection"
                })
                continue

            # Use the first detected hand (same behaviour as predict script)
            x_min, y_min, x_max, y_max = boxes[0]
            crop      = img.crop((x_min, y_min, x_max, y_max))
            predicted, confidence = classify_crop(model, crop, device)

            confusion[true_label][predicted] += 1

            if predicted == true_label:
                correct               += 1
                class_correct[true_label] += 1
            else:
                wrong.append({
                    "file":       str(img_path),
                    "true":       true_label,
                    "predicted":  predicted,
                    "confidence": confidence,
                    "reason":     "misclassification"
                })

    # ── Print results ──────────────────────────────────────────────────────────

    detectable = total - no_detection
    accuracy_overall    = correct / total       if total       > 0 else 0
    accuracy_detectable = correct / detectable  if detectable  > 0 else 0

    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS")
    print(f"{'='*50}")
    print(f"Total images:          {total}")
    print(f"No hand detected:      {no_detection}  ({no_detection/total:.1%})")
    print(f"Detectable images:     {detectable}")
    print(f"Correct predictions:   {correct}")
    print(f"")
    print(f"Accuracy (all images): {accuracy_overall:.1%}   (counts no-detection as wrong)")
    print(f"Accuracy (detected):   {accuracy_detectable:.1%}  (only images where hand was found)")

    print(f"\n{'='*50}")
    print(f"PER-CLASS ACCURACY")
    print(f"{'='*50}")
    for cls in CLASS_NAMES:
        n   = class_total[cls]
        c   = class_correct[cls]
        acc = c / n if n > 0 else 0
        print(f"  {cls}:  {c:>4} / {n:<4}  ({acc:.1%})")
    print(f"\n{'='*50}")
    print(f"PER-CLASS ACCURACY (detected only)")
    print(f"{'='*50}")
    for cls in CLASS_NAMES:
        detected = class_total[cls] - sum(
            1 for w in wrong
            if w["true"] == cls and w["reason"] == "mediapipe_no_detection"
        )
        c   = class_correct[cls]
        acc = c / detected if detected > 0 else 0
        print(f"  {cls}:  {c:>4} / {detected:<4}  ({acc:.1%})")
    print(f"\n{'='*50}")
    print(f"CONFUSION MATRIX  (rows = true, cols = predicted)")
    print(f"{'='*50}")
    col_headers = CLASS_NAMES + ["NO_DETECTION"]
    header = f"{'':>6}" + "".join(f"{h:>14}" for h in col_headers)
    print(header)
    for true_cls in CLASS_NAMES:
        row = f"{true_cls:>6}"
        for pred_cls in col_headers:
            count = confusion[true_cls][pred_cls]
            row  += f"{count:>14}"
        print(row)

    print(f"\n{'='*50}")
    print(f"MISTAKES  (first 15)")
    print(f"{'='*50}")
    misclassified = [w for w in wrong if w["reason"] == "misclassification"]
    undetected    = [w for w in wrong if w["reason"] == "mediapipe_no_detection"]

    if misclassified:
        print(f"\nMisclassifications ({len(misclassified)} total):")
        for w in misclassified[:15]:
            print(f"  true={w['true']:<4} predicted={w['predicted']:<4} conf={w['confidence']:.1%}  {w['file']}")

    if undetected:
        print(f"\nNo detection ({len(undetected)} total, first 10):")
        for w in undetected[:10]:
            print(f"  true={w['true']:<4}  {w['file']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate case 2 model on a test directory")
    parser.add_argument("test_dir", type=str, help="Path to test directory (subfolders = class names)")
    args = parser.parse_args()
    evaluate(args.test_dir)