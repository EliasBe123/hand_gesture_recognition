"""
Preprocess the HGR dataset: use MediaPipe Hands to detect the hand center
in each image and save normalized (x, y) coordinates to a CSV file.

Coordinates are normalized to [0, 1] relative to image dimensions,
so they work regardless of image size.

Usage:
    python -m src.data.preprocess
"""

import os
import sys
import csv

import mediapipe as mp
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_case2v2 import HGR_TRAIN_DIR, HGR_TEST_DIR, DATA_DIR


OUTPUT_CSV_TRAIN = os.path.join(DATA_DIR, "hgr_train_labels.csv")
OUTPUT_CSV_TEST = os.path.join(DATA_DIR, "hgr_test_labels.csv")


def get_hand_center(image_rgb):
    """
    Detect the hand in an RGB numpy array using MediaPipe.
    Returns normalized (cx, cy) of the hand center, or None if no hand found.
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            # Center = mean of all 21 landmark positions (already normalized 0-1)
            cx = np.mean([lm.x for lm in landmarks])
            cy = np.mean([lm.y for lm in landmarks])
            return cx, cy
    return None


def process_directory(root_dir, output_csv):
    """
    Walk through an ImageFolder-style directory, detect hand centers,
    and write results to a CSV: filepath, class, center_x, center_y
    """
    rows = []
    skipped = 0
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    print(f"Processing {root_dir} ...")
    print(f"Classes: {classes}")

    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"  {cls_name}: {len(files)} images", end="", flush=True)

        cls_skipped = 0
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            img = Image.open(fpath).convert("RGB")
            img_np = np.array(img)

            result = get_hand_center(img_np)
            if result:
                cx, cy = result
                # Store path relative to the root_dir for portability
                rel_path = os.path.join(cls_name, fname)
                rows.append([rel_path, cls_name, f"{cx:.6f}", f"{cy:.6f}"])
            else:
                cls_skipped += 1

        skipped += cls_skipped
        print(f"  (skipped {cls_skipped} — no hand detected)")

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "center_x", "center_y"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} labels to {output_csv}")
    print(f"Skipped {skipped} images (no hand detected)\n")


def main():
    process_directory(HGR_TRAIN_DIR, OUTPUT_CSV_TRAIN)
    process_directory(HGR_TEST_DIR, OUTPUT_CSV_TEST)


if __name__ == "__main__":
    main()
