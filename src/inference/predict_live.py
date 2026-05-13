import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import numpy as np
import cv2
import sys
import os
import time
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn_case2v2 import HandGestureCNN_Case2
from src.utils.config_case2v2 import BEST_MODEL_PATH_CASE6, IMG_SIZE, NUM_CLASSES, DEVICE

CLASS_NAMES = ["A", "F", "L", "Y"]

# ── Colours for each class label ────────────────────────────────────────────
LABEL_COLORS = {
    "A": (0,   220, 255),   # cyan
    "F": (0,   255, 100),   # green
    "L": (255, 180,   0),   # amber
    "Y": (220,   0, 255),   # magenta
}


class AdaptiveThreshold:
    """Must stay in sync with loader_case4_blackwhite.py."""
    def __call__(self, x):
        np_img = (x[0].numpy() * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(np_img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_float = torch.from_numpy(binary / 255.0).float()
        return binary_float.unsqueeze(0).repeat(3, 1, 1)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    AdaptiveThreshold(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_model(device):
    model = HandGestureCNN_Case2()
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH_CASE6, map_location=device, weights_only=True)
    )
    model.eval()
    model.to(device)
    return model


def detect_hands(image_rgb):
    """MediaPipe hand detection → list of square bounding boxes in pixel coords."""
    mp_hands = mp.solutions.hands
    h, w, _ = image_rgb.shape
    boxes = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                xs = [lm.x for lm in lms.landmark]
                ys = [lm.y for lm in lms.landmark]
                pad = 0.05
                x_min = max(0, int((min(xs) - pad) * w))
                y_min = max(0, int((min(ys) - pad) * h))
                x_max = min(w, int((max(xs) + pad) * w))
                y_max = min(h, int((max(ys) + pad) * h))
                # Square crop centred on the hand
                side    = max(x_max - x_min, y_max - y_min)
                cx      = (x_min + x_max) // 2
                cy      = (y_min + y_max) // 2
                half    = side // 2
                boxes.append((
                    max(0, cx - half), max(0, cy - half),
                    min(w, cx + half), min(h, cy + half)
                ))
    return boxes


def classify_crop(model, crop_pil, device):
    """Return (label, confidence, all_probs) for a PIL crop."""
    tensor = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return CLASS_NAMES[idx], probs[idx].item(), probs


def draw_overlay(frame, boxes, predictions, next_inference_in, fps_setting):
    """
    Draw bounding boxes + labels on the BGR frame in-place.
    Also draws a countdown bar and FPS indicator.
    """
    h, w = frame.shape[:2]

    for (x_min, y_min, x_max, y_max), (label, conf, probs) in zip(boxes, predictions):
        color = LABEL_COLORS.get(label, (255, 255, 255))

        # Bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Label badge
        text        = f"{label}  {conf:.0%}"
        font        = cv2.FONT_HERSHEY_DUPLEX
        font_scale  = 0.9
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        badge_x1    = x_min
        badge_y1    = max(0, y_min - th - 14)
        badge_x2    = x_min + tw + 10
        badge_y2    = y_min

        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            frame, text,
            (badge_x1 + 5, badge_y2 - 6),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

        # Mini probability bar chart (top-right of bounding box)
        bar_x   = x_max + 8
        bar_y   = y_min
        bar_w   = 90
        bar_h   = 18
        padding = 4
        for i, name in enumerate(CLASS_NAMES):
            p    = probs[i].item()
            by   = bar_y + i * (bar_h + padding)
            blen = int(p * bar_w)
            bc   = LABEL_COLORS.get(name, (200, 200, 200))
            # background
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (40, 40, 40), -1)
            # filled bar
            cv2.rectangle(frame, (bar_x, by), (bar_x + blen, by + bar_h), bc, -1)
            # label
            cv2.putText(
                frame, f"{name} {p:.0%}",
                (bar_x + 3, by + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

    # ── Countdown progress bar (bottom of frame) ─────────────────────────────
    interval      = 1.0 / fps_setting
    elapsed_frac  = 1.0 - min(next_inference_in / interval, 1.0)
    bar_total_w   = w - 20
    bar_filled    = int(elapsed_frac * bar_total_w)
    bar_y_pos     = h - 18

    cv2.rectangle(frame, (10, bar_y_pos), (10 + bar_total_w, bar_y_pos + 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, bar_y_pos), (10 + bar_filled,  bar_y_pos + 10), (0, 200, 80), -1)

    countdown_text = f"Next inference in {next_inference_in:.1f}s  |  {fps_setting} FPS  |  Q = quit"
    cv2.putText(
        frame, countdown_text,
        (10, bar_y_pos - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    )

    # ── Model info (top-left corner) ─────────────────────────────────────────
    cv2.putText(
        frame, "Hand Gesture CNN  |  A  F  L  Y",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
    )


def run_live(fps: float = 1.0, camera_index: int = 0):
    """Main loop: capture → resize to 100×100 → detect → classify → display."""
    device   = torch.device(DEVICE)
    model    = load_model(device)
    interval = 1.0 / fps

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}.")
        sys.exit(1)

    print(f"Camera opened. Running at {fps} FPS (inference every {interval:.1f}s).")
    print("Press Q in the window to quit.")

    last_inference_time = 0.0
    last_boxes          = []
    last_predictions    = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        now            = time.time()
        time_since     = now - last_inference_time
        next_inf_in    = max(0.0, interval - time_since)

        # ── Inference tick ────────────────────────────────────────────────
        if time_since >= interval:
            last_inference_time = now

            # Resize full frame to 100×100 (matches training resolution)
            small_bgr = cv2.resize(frame_bgr, (100, 100), interpolation=cv2.INTER_LANCZOS4)
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

            boxes = detect_hands(small_rgb)

            predictions = []
            if boxes:
                pil_img = Image.fromarray(small_rgb)
                for (x_min, y_min, x_max, y_max) in boxes:
                    crop = pil_img.crop((x_min, y_min, x_max, y_max))
                    crop = crop.convert("L").convert("RGB")
                    predictions.append(classify_crop(model, crop, device))

                for i, ((x_min, y_min, x_max, y_max), (label, conf, _)) in \
                        enumerate(zip(boxes, predictions)):
                    print(f"  Hand {i+1}: {label}  ({conf:.1%})  @ "
                          f"({x_min},{y_min})–({x_max},{y_max})")
            else:
                print("  No hands detected.")

            # Scale boxes back to the *display* frame size for drawing
            dh, dw = frame_bgr.shape[:2]
            sx, sy = dw / 100.0, dh / 100.0
            last_boxes = [
                (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                for (x1, y1, x2, y2) in boxes
            ]
            last_predictions = predictions

        # ── Draw overlay on the full-resolution display frame ─────────────
        display = frame_bgr.copy()
        draw_overlay(display, last_boxes, last_predictions, next_inf_in, fps)

        cv2.imshow("Hand Gesture — Live", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live camera hand-gesture classifier"
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Inference rate in FPS. Use 0.5 for one prediction every 2 s. (default: 1.0)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0)"
    )
    args = parser.parse_args()

    if args.fps <= 0:
        print("ERROR: --fps must be a positive number.")
        sys.exit(1)

    run_live(fps=args.fps, camera_index=args.camera)