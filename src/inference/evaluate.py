import os
import sys
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


# ── Case configurations ───────────────────────────────────────────────────────
# Add new cases here as the project grows

def load_case(case):
    if case == "case1":
        from src.models.cnn import HandGestureCNN
        from src.utils.config import BEST_MODEL_PATH, IMG_SIZE, NUM_CLASSES, DEVICE
        model      = HandGestureCNN()
        model_path = BEST_MODEL_PATH
        device     = torch.device(DEVICE)
        class_names = sorted([str(i) for i in range(NUM_CLASSES)])
        transform  = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        use_mediapipe = False

    elif case == "case2":
        from src.models.cnn_case2v2 import HandGestureCNN_Case2
        from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE2, IMG_SIZE, NUM_CLASSES, DEVICE
        model      = HandGestureCNN_Case2()
        model_path = BEST_MODEL_PATH_CASE2
        device     = torch.device(DEVICE)
        class_names = ["A", "F", "L", "Y"]
        transform  = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        use_mediapipe = True

    # Future cases here
    elif case == "case3":
        from src.models.cnn_case2v2 import HandGestureCNN_Case2
        from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE3, IMG_SIZE, NUM_CLASSES, DEVICE
        model      = HandGestureCNN_Case2()
        model_path = BEST_MODEL_PATH_CASE3
        device     = torch.device(DEVICE)
        class_names = ["A", "F", "L", "Y"]
        transform  = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        use_mediapipe = True
   

    else:
        print(f"Unknown case '{case}'. Available: case1, case2")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    return model, device, class_names, transform, use_mediapipe


# ── MediaPipe detection (case 2+) ─────────────────────────────────────────────

def detect_hands(image_rgb):
    import mediapipe as mp
    mp_hands   = mp.solutions.hands
    h, w, _    = image_rgb.shape
    hands_result = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                pad   = 0.05
                x_min = max(0, int((min(x_coords) - pad) * w))
                y_min = max(0, int((min(y_coords) - pad) * h))
                x_max = min(w, int((max(x_coords) + pad) * w))
                y_max = min(h, int((max(y_coords) + pad) * h))

                width    = x_max - x_min
                height   = y_max - y_min
                max_side = max(width, height)
                center_x = x_min + width  // 2
                center_y = y_min + height // 2
                half     = int(max_side / 2)

                hands_result.append((
                    max(0, center_x - half),
                    max(0, center_y - half),
                    min(w, center_x + half),
                    min(h, center_y + half),
                ))

    return hands_result


# ── Classification ────────────────────────────────────────────────────────────

def classify(model, img, transform, device, class_names):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        idx   = probs.argmax().item()
    return class_names[idx], probs[idx].item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(test_dir, case):
    model, device, class_names, transform, use_mediapipe = load_case(case)

    
    total        = 0
    correct      = 0
    no_detection = 0
    wrong        = []
    confusion    = defaultdict(lambda: defaultdict(int))
    class_total  = defaultdict(int)
    class_correct= defaultdict(int)

    test_path        = Path(test_dir)
    image_extensions = ["*.jpg", "*.png", "*.jpeg"]


    # Count total images for ETA
    total_images = 0
    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        if class_folder.name not in class_names:
            continue
        for ext in image_extensions:
            total_images += len(list(class_folder.glob(ext)))



    print(f"Evaluating {case} on {total_images} images in '{test_dir}'")
    if use_mediapipe:
        print("MediaPipe hand detection: ON")
    else:
        print("MediaPipe hand detection: OFF (full image used)")
    print()

    start_time = time.time()
    
    # Walk through every class folder for images
    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        true_label = class_folder.name
        if true_label not in class_names:
            print(f"  Warning: folder '{true_label}' not in class names, skipping.")
            continue

        images = []
        for ext in image_extensions:
            images.extend(class_folder.glob(ext))
        images = sorted(images)

        # Loop through each image in the class folder
        for img_path in images:
            img = Image.open(img_path).convert("RGB")

            total += 1
            class_total[true_label] += 1

            # Progress + ETA
            # This is for ETA and is updated for every image
            elapsed   = time.time() - start_time
            rate      = total / elapsed if elapsed > 0 else 0
            remaining = (total_images - total) / rate if rate > 0 else 0
            mins, secs = divmod(int(remaining), 60)
            print(f"  {total}/{total_images}  |  {rate:.1f} img/s  |  ETA: {mins}m {secs}s    ", end="\r")

            # Here we try to detect the hand first (if case 2), and if no hand is found, we count it as a "no detection" error.
            if use_mediapipe:
                img_np = np.array(img)
                boxes  = detect_hands(img_np)

                if not boxes:
                    no_detection += 1
                    confusion[true_label]["NO_DETECTION"] += 1
                    wrong.append({
                        "file": str(img_path), "true": true_label,
                        "predicted": "NO_DETECTION", "confidence": 0.0,
                        "reason": "mediapipe_no_detection"
                    })
                    continue

                x_min, y_min, x_max, y_max = boxes[0]
                img_to_classify = img.crop((x_min, y_min, x_max, y_max))
            else:
                # Case 1: classify the full image directly
                img_to_classify = img

            # Now we classify the cropped hand (or full image for case 1)
            predicted, confidence = classify(model, img_to_classify, transform, device, class_names)
            confusion[true_label][predicted] += 1

            # Update correct/wrong counts
            if predicted == true_label:
                correct += 1
                class_correct[true_label] += 1
            else:
                wrong.append({
                    "file": str(img_path), "true": true_label,
                    "predicted": predicted, "confidence": confidence,
                    "reason": "misclassification"
                })

    # ── Results ───────────────────────────────────────────────────────────────

    detectable          = total - no_detection
    accuracy_overall    = correct / total      if total      > 0 else 0
    accuracy_detectable = correct / detectable if detectable > 0 else 0
    elapsed_total       = time.time() - start_time
    mins_total, secs_total = divmod(int(elapsed_total), 60)

    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS  [{case}]")
    print(f"{'='*50}")
    print(f"Total images:          {total}")
    print(f"Time elapsed:          {mins_total}m {secs_total}s")
    if use_mediapipe:
        print(f"No hand detected:      {no_detection}  ({no_detection/total:.1%})")
        print(f"Detectable images:     {detectable}")
    print(f"Correct predictions:   {correct}")
    print(f"")
    print(f"Accuracy (all images): {accuracy_overall:.1%}")
    if use_mediapipe:
        print(f"Accuracy (detected):   {accuracy_detectable:.1%}  (only where hand was found)")

    print(f"\n{'='*50}")
    print(f"PER-CLASS ACCURACY (all images)")
    print(f"{'='*50}")
    # cls = class name, n = total images of that class, c = correct predictions for that class, acc = accuracy for that class
    for cls in class_names:
        n   = class_total[cls]
        c   = class_correct[cls]
        acc = c / n if n > 0 else 0
        print(f"  {cls}:  {c:>4} / {n:<4}  ({acc:.1%})")

    # If using MediaPipe (case 2), also show accuracy for detected images only (excluding no-detection cases), since no-detection is a separate issue from misclassification.
    if use_mediapipe:
        print(f"\n{'='*50}")
        print(f"PER-CLASS ACCURACY (detected only)")
        print(f"{'='*50}")
        for cls in class_names:
            detected = class_total[cls] - sum(
                1 for w in wrong
                if w["true"] == cls and w["reason"] == "mediapipe_no_detection"
            )
            c   = class_correct[cls]
            acc = c / detected if detected > 0 else 0
            print(f"  {cls}:  {c:>4} / {detected:<4}  ({acc:.1%})")

    # Confusion matrix (is a bit wierd to read especially with case 1 since its so wide but very useful otherwise)
    print(f"\n{'='*50}")
    print(f"CONFUSION MATRIX  (rows=true, cols=predicted)")
    print(f"{'='*50}")
    col_headers = class_names + (["NO_DETECTION"] if use_mediapipe else [])
    print(f"{'':>6}" + "".join(f"{h:>14}" for h in col_headers))
    for true_cls in class_names:
        row = f"{true_cls:>6}"
        for pred_cls in col_headers:
            row += f"{confusion[true_cls][pred_cls]:>14}"
        print(row)


    # Print mistakes (first 15 misclassifications and first 10 no-detections, good for debugging and understanding common failure cases)
    misclassified = [w for w in wrong if w["reason"] == "misclassification"]
    undetected    = [w for w in wrong if w["reason"] == "mediapipe_no_detection"]
    print(f"\n{'='*50}")
    print(f"MISTAKES")
    print(f"{'='*50}")
    if misclassified:
        print(f"\nMisclassifications ({len(misclassified)} total, first 15):")
        for w in misclassified[:15]:
            print(f"  true={w['true']:<4} predicted={w['predicted']:<4} conf={w['confidence']:.1%}  {w['file']}")
    if undetected:
        print(f"\nNo detection ({len(undetected)} total, first 10):")
        for w in undetected[:10]:
            print(f"  true={w['true']:<4}  {w['file']}")


if __name__ == "__main__":
    # This is the entry point for evaluation, it takes a test directory and a case name as arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test directory")
    parser.add_argument("test_dir", type=str, help="Path to test directory (subfolders = class names)")
    parser.add_argument("--case",   type=str, default="case1", help="Which case to evaluate: case1, case2, ...")
    args = parser.parse_args()
    evaluate(args.test_dir, args.case)