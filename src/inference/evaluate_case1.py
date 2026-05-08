import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn import HandGestureCNN
from src.utils.config import BEST_MODEL_PATH, IMG_SIZE, NUM_CLASSES, DEVICE

CLASS_NAMES = sorted([str(i) for i in range(NUM_CLASSES)])

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def evaluate(test_dir):
    # Load model once
    device = torch.device(DEVICE)
    model = HandGestureCNN()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    wrong = []

    # Walk through every class folder
    test_path = Path(test_dir)
    for class_folder in sorted(test_path.iterdir()):
        if not class_folder.is_dir():
            continue
        true_label = class_folder.name  # folder name is the true class

        for img_path in sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))):
            img = Image.open(img_path)
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                predicted_idx = probs.argmax().item()
                confidence = probs[predicted_idx].item()
                predicted = CLASS_NAMES[predicted_idx]

            total += 1
            if predicted == true_label:
                correct += 1
            else:
                wrong.append({
                    "file": str(img_path),
                    "true": true_label,
                    "predicted": predicted,
                    "confidence": confidence
                })

            # Progress indicator
            if total % 50 == 0:
                print(f"  Processed {total} images...", end="\r")

    # Results
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*40}")
    print(f"Total images:  {total}")
    print(f"Correct:       {correct}")
    print(f"Wrong:         {len(wrong)}")
    print(f"Accuracy:      {accuracy:.1%}")
    print(f"{'='*40}\n")

    if wrong:
        print(f"First 10 mistakes:")
        for w in wrong[:10]:
            print(f"  true={w['true']:<5} predicted={w['predicted']:<5} conf={w['confidence']:.1%}  {w['file']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str, help="Path to test directory")
    args = parser.parse_args()
    evaluate(args.test_dir)