import torch
import numpy as np
import cv2
import sys
import os
import time
import argparse
from PIL import Image

# Add project root to path so both src.* imports and the sibling import work
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# ── Re-use everything from the still-image script ────────────────────────────
# predict_case6_100.py lives in the same src/inference/ folder.
# Importing it gives us: transform, detect_hands, classify_crop,
# apply_adaptive_threshold_for_display, CLASS_NAMES, etc.
from predict_case6_100 import (
    transform,
    detect_hands,
    classify_crop,
    apply_adaptive_threshold_for_display,
    CLASS_NAMES,
)
from src.models.cnn_case2v2 import HandGestureCNN_Case2
from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE6, DEVICE

# ── Colours for each class label (BGR for OpenCV) ────────────────────────────
LABEL_COLORS = {
    "A": (255, 255,   0),   # cyan
    "F": (0, 255,   0),   # green
    "L": (  0, 0, 255),   # amber
    "Y": (255,   0, 255),   # magenta
}


def load_model(device):
    model = HandGestureCNN_Case2()
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH_CASE6, map_location=device, weights_only=True)
    )
    model.eval()
    model.to(device)
    return model


def draw_overlay(frame, boxes, predictions, next_inference_in, fps_setting):
    """Draw bounding boxes, labels, prob bars, and countdown on the BGR frame."""
    h, w = frame.shape[:2]

    for (x_min, y_min, x_max, y_max), (label, conf, probs) in zip(boxes, predictions):
        color = LABEL_COLORS.get(label, (255, 255, 255))

        # Bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Label badge above box
        text        = f"{label}  {conf:.0%}"
        font        = cv2.FONT_HERSHEY_DUPLEX
        font_scale  = 0.9
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        badge_x1 = x_min
        badge_y1 = max(0, y_min - th - 14)
        badge_x2 = x_min + tw + 10
        badge_y2 = y_min
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            frame, text,
            (badge_x1 + 5, badge_y2 - 6),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

        # Mini probability bar chart to the right of the bounding box
        bar_x   = x_max + 8
        bar_w   = 90
        bar_h   = 18
        padding = 4
        for i, name in enumerate(CLASS_NAMES):
            p    = probs[i].item()
            by   = y_min + i * (bar_h + padding)
            blen = int(p * bar_w)
            bc   = LABEL_COLORS.get(name, (200, 200, 200))
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (40, 40, 40), -1)
            cv2.rectangle(frame, (bar_x, by), (bar_x + blen,  by + bar_h), bc, -1)
            cv2.putText(
                frame, f"{name} {p:.0%}",
                (bar_x + 3, by + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

    # Countdown progress bar at the bottom
    # interval     = 1.0 / fps_setting
    # elapsed_frac = 1.0 - min(next_inference_in / interval, 1.0)
    # bar_total_w  = w - 20
    # bar_filled   = int(elapsed_frac * bar_total_w)
    # bar_y_pos    = h - 18
    # cv2.rectangle(frame, (10, bar_y_pos), (10 + bar_total_w, bar_y_pos + 10), (50, 50, 50), -1)
    # cv2.rectangle(frame, (10, bar_y_pos), (10 + bar_filled,  bar_y_pos + 10), (0, 200, 80), -1)
    # cv2.putText(
    #     frame,
    #     f"Next inference in {next_inference_in:.1f}s  |  {fps_setting} FPS  |  Q = quit",
    #     (10, bar_y_pos - 5),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    # )

    # Model info top-left
    # cv2.putText(
    #     frame, "Hand Gesture CNN  |  A  F  L  Y",
    #     (10, 22),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
    # )


def run_live(fps: float = 1.0, camera_index: int = 0):
    """
    Main loop — identical pipeline to predict_case6_100.predict():
      1. Resize full frame to 100x100   (matches training resolution)
      2. detect_hands()                 (same function as still-image script)
      3. classify_crop()                (same function as still-image script)
      4. Draw overlay on display frame
    """
    device   = torch.device(DEVICE)
    model    = load_model(device)
    interval = 1.0 / fps

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}.")
        sys.exit(1)

    print(f"Camera opened (index {camera_index}). "
          f"Inference every {interval:.1f}s ({fps} FPS).")
    print("Press Q or Esc in the window to quit.")

    last_inference_time = 0.0
    last_boxes          = []
    last_predictions    = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        now         = time.time()
        time_since  = now - last_inference_time
        next_inf_in = max(0.0, interval - time_since)

        # ── Inference tick ────────────────────────────────────────────────────
        if time_since >= interval:
            last_inference_time = now

            # Same first step as predict_case6_100.predict()
            small_bgr = cv2.resize(frame_bgr, (100, 100), interpolation=cv2.INTER_LANCZOS4)
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

            boxes = detect_hands(small_rgb)      # imported from predict_case6_100

            predictions = []
            if boxes:
                pil_img = Image.fromarray(small_rgb)
                for (x_min, y_min, x_max, y_max) in boxes:
                    crop = pil_img.crop((x_min, y_min, x_max, y_max))
                    crop = crop.convert("L").convert("RGB")
                    predictions.append(
                        classify_crop(model, crop, device)   # imported
                    )
                for i, ((x1, y1, x2, y2), (lbl, conf, _)) in \
                        enumerate(zip(boxes, predictions)):
                    print(f"  Hand {i+1}: {lbl}  ({conf:.1%})  @ ({x1},{y1})-({x2},{y2})")
            else:
                print("  No hands detected.")

            # Scale 100x100 boxes up to display-frame coordinates
            dh, dw = frame_bgr.shape[:2]
            sx, sy = dw / 100.0, dh / 100.0
            last_boxes = [
                (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                for (x1, y1, x2, y2) in boxes
            ]
            last_predictions = predictions

        # ── Draw & show ───────────────────────────────────────────────────────
        display = frame_bgr.copy()
        draw_overlay(display, last_boxes, last_predictions, next_inf_in, fps)
        cv2.imshow("Hand Gesture -- Live", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live camera hand-gesture classifier")
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Inference rate in FPS. Use 0.5 for one prediction every 2s. (default: 1.0)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0 = built-in Mac front camera)"
    )
    args = parser.parse_args()

    if args.fps <= 0:
        print("ERROR: --fps must be a positive number.")
        sys.exit(1)

    run_live(fps=args.fps, camera_index=args.camera)