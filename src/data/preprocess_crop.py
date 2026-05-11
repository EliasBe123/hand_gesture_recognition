import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from tqdm import tqdm
import argparse

# How much padding to add around the hand bounding box (% of box size)
PADDING = 0.2

def make_square_crop(img_np, x_min, y_min, x_max, y_max):
    """
    Expands the bounding box to a square using padding,
    then crops — no stretching, no distortion.
    """
    h, w = img_np.shape[:2]
    box_w = x_max - x_min
    box_h = y_max - y_min

    # Add padding around the box
    pad_w = int(box_w * PADDING)
    pad_h = int(box_h * PADDING)
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(w, x_max + pad_w)
    y_max = min(h, y_max + pad_h)

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Make it square by expanding the shorter side
    if box_w > box_h:
        diff = box_w - box_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + diff // 2)
    else:
        diff = box_h - box_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + diff // 2)

    return img_np[y_min:y_max, x_min:x_max]


def crop_hands_in_dir(input_dir, output_dir):
    mp_hands = mp.solutions.hands

    class_folders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    skipped = 0
    saved = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        for cls in class_folders:
            cls_in  = os.path.join(input_dir, cls)
            cls_out = os.path.join(output_dir, cls)
            os.makedirs(cls_out, exist_ok=True)

            files = [f for f in os.listdir(cls_in) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            for fname in tqdm(files, desc=cls):
                img_path = os.path.join(cls_in, fname)
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if not results.multi_hand_landmarks:
                    # No hand detected — skip this image
                    skipped += 1
                    continue

                # Use first detected hand
                landmarks = results.multi_hand_landmarks[0]
                h, w = img_rgb.shape[:2]
                x_coords = [lm.x * w for lm in landmarks.landmark]
                y_coords = [lm.y * h for lm in landmarks.landmark]

                x_min = int(min(x_coords))
                y_min = int(min(y_coords))
                x_max = int(max(x_coords))
                y_max = int(max(y_coords))

                # Square crop with padding
                crop = make_square_crop(img_rgb, x_min, y_min, x_max, y_max)

                if crop.size == 0:
                    skipped += 1
                    continue

                # Save as PNG
                out_path = os.path.join(cls_out, os.path.splitext(fname)[0] + ".png")
                Image.fromarray(crop).save(out_path)
                saved += 1

    print(f"\nDone. Saved: {saved}  Skipped (no hand detected): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop hands from dataset using MediaPipe")
    parser.add_argument("--input",  type=str, required=True, help="Input dataset root (with class subfolders)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for cropped images")
    args = parser.parse_args()

    print(f"Processing: {args.input} → {args.output}")
    crop_hands_in_dir(args.input, args.output)