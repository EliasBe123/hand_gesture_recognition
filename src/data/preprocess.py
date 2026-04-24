"""
Preprocess the HGR dataset:

1. Label: Use MediaPipe Hands to detect the hand center in each image
   and save normalized (x, y) coordinates to a CSV file.

2. Build movement sequences: Go through labeled images, group by gesture class,
   sort by hand position, and extract sequences of 5 frames where the hand
   moves consistently in one direction (N/S/E/W). Saves a new dataset with
   direction classes and a CSV of sequences.

Usage:
    python -m src.data.preprocess              # run both steps
    python -m src.data.preprocess --label-only  # only step 1
    python -m src.data.preprocess --seq-only    # only step 2 (requires labels CSV)
"""

import os
import sys
import csv
import shutil
import argparse
from itertools import combinations

import mediapipe as mp
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_case2v2 import HGR_TRAIN_DIR, HGR_TEST_DIR, DATA_DIR


# --- Labeling outputs ---
OUTPUT_CSV_TRAIN = os.path.join(DATA_DIR, "hgr_train_labels.csv")
OUTPUT_CSV_TEST = os.path.join(DATA_DIR, "hgr_test_labels.csv")

# --- Movement sequence outputs ---
SEQ_LENGTH = 5                          # frames per sequence
MIN_MOVEMENT = 0.05                     # minimum total displacement (normalized) to count as movement
MOVEMENT_DATASET_DIR = os.path.join(DATA_DIR, "hgr_movement")
MOVEMENT_CSV = os.path.join(DATA_DIR, "hgr_movement_sequences.csv")

DIRECTIONS = {
    "E": (1, 0),   # center_x increases
    "W": (-1, 0),  # center_x decreases
    "S": (0, 1),   # center_y increases (down in image coords)
    "N": (0, -1),  # center_y decreases (up in image coords)
}


# ============================================================
# Step 1: Label hand centers with MediaPipe
# ============================================================

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
                rel_path = os.path.join(cls_name, fname)
                rows.append([rel_path, cls_name, f"{cx:.6f}", f"{cy:.6f}"])
            else:
                cls_skipped += 1

        skipped += cls_skipped
        print(f"  (skipped {cls_skipped} — no hand detected)")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "center_x", "center_y"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} labels to {output_csv}")
    print(f"Skipped {skipped} images (no hand detected)\n")


# ============================================================
# Step 2: Build movement sequences from labeled data
# ============================================================

def load_labels(csv_path):
    """Load the labels CSV into a list of dicts."""
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "filepath": row["filepath"],
                "class": row["class"],
                "cx": float(row["center_x"]),
                "cy": float(row["center_y"]),
            })
    return entries


def classify_direction(points):
    """
    Given a list of (cx, cy) tuples forming a sequence, determine if the
    movement is consistently N/S/E/W.
    Returns the direction string or None if movement is ambiguous or too small.
    """
    dx = points[-1][0] - points[0][0]  # total x displacement
    dy = points[-1][1] - points[0][1]  # total y displacement

    # Check all intermediate steps move in the same general direction
    for i in range(1, len(points)):
        step_dx = points[i][0] - points[i - 1][0]
        step_dy = points[i][1] - points[i - 1][1]
        # Each step should agree with overall direction (no backtracking)
        if dx != 0 and (step_dx * dx) < 0:
            return None
        if dy != 0 and (step_dy * dy) < 0:
            return None

    abs_dx, abs_dy = abs(dx), abs(dy)

    # Must have enough total displacement
    if max(abs_dx, abs_dy) < MIN_MOVEMENT:
        return None

    # Dominant axis determines direction
    if abs_dx > abs_dy:
        return "E" if dx > 0 else "W"
    else:
        return "S" if dy > 0 else "N"


def build_movement_sequences(labels_csv, source_dir):
    """
    From labeled images, find sequences of SEQ_LENGTH images within the same
    gesture class that show consistent hand movement in one direction.

    Strategy:
    - Group images by gesture class
    - Sort by center_x → find E/W sequences (sliding window)
    - Sort by center_y → find N/S sequences (sliding window)
    - Copy images into data/hgr_movement/{N,S,E,W}/ with correlated names
    - Write a CSV with all sequence paths + direction
    """
    entries = load_labels(labels_csv)

    # Group by gesture class
    by_class = {}
    for e in entries:
        by_class.setdefault(e["class"], []).append(e)

    sequences = []  # list of (direction, [entry1, ..., entry5])

    for cls_name, images in by_class.items():
        # --- E/W sequences: sort by center_x ---
        sorted_x = sorted(images, key=lambda e: e["cx"])
        for i in range(len(sorted_x) - SEQ_LENGTH + 1):
            window = sorted_x[i:i + SEQ_LENGTH]
            points = [(e["cx"], e["cy"]) for e in window]
            direction = classify_direction(points)
            if direction in ("E", "W"):
                sequences.append((direction, window))

        # --- N/S sequences: sort by center_y ---
        sorted_y = sorted(images, key=lambda e: e["cy"])
        for i in range(len(sorted_y) - SEQ_LENGTH + 1):
            window = sorted_y[i:i + SEQ_LENGTH]
            points = [(e["cx"], e["cy"]) for e in window]
            direction = classify_direction(points)
            if direction in ("N", "S"):
                sequences.append((direction, window))

    print(f"Found {len(sequences)} movement sequences:")
    counts = {d: 0 for d in DIRECTIONS}
    for d, _ in sequences:
        counts[d] += 1
    for d, c in counts.items():
        print(f"  {d}: {c}")

    # --- Create the movement dataset ---
    # Clear old output
    if os.path.exists(MOVEMENT_DATASET_DIR):
        shutil.rmtree(MOVEMENT_DATASET_DIR)

    for d in DIRECTIONS:
        os.makedirs(os.path.join(MOVEMENT_DATASET_DIR, d), exist_ok=True)

    csv_rows = []
    seq_counter = {d: 0 for d in DIRECTIONS}

    for direction, window in sequences:
        seq_id = seq_counter[direction]
        seq_counter[direction] += 1
        dir_folder = os.path.join(MOVEMENT_DATASET_DIR, direction)

        paths = []
        for frame_idx, entry in enumerate(window):
            # Naming: {seq_id}_image{frame_idx}.png
            new_name = f"{seq_id}_image{frame_idx}.png"
            src_path = os.path.join(source_dir, entry["filepath"])
            dst_path = os.path.join(dir_folder, new_name)
            shutil.copy2(src_path, dst_path)
            paths.append(os.path.join(direction, new_name))

        csv_rows.append(paths)

    # Write sequences CSV
    with open(MOVEMENT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"frame_{i}" for i in range(SEQ_LENGTH)]
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"\nMovement dataset saved to {MOVEMENT_DATASET_DIR}")
    print(f"Sequences CSV saved to {MOVEMENT_CSV}")
    print(f"Total sequences: {len(csv_rows)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess HGR dataset")
    parser.add_argument("--label-only", action="store_true", help="Only run MediaPipe labeling")
    parser.add_argument("--seq-only", action="store_true", help="Only build movement sequences (requires labels CSV)")
    args = parser.parse_args()

    if args.seq_only:
        build_movement_sequences(OUTPUT_CSV_TRAIN, HGR_TRAIN_DIR)
        return

    if args.label_only:
        process_directory(HGR_TRAIN_DIR, OUTPUT_CSV_TRAIN)
        process_directory(HGR_TEST_DIR, OUTPUT_CSV_TEST)
        return

    # Default: run both
    process_directory(HGR_TRAIN_DIR, OUTPUT_CSV_TRAIN)
    process_directory(HGR_TEST_DIR, OUTPUT_CSV_TEST)
    build_movement_sequences(OUTPUT_CSV_TRAIN, HGR_TRAIN_DIR)


if __name__ == "__main__":
    main()
