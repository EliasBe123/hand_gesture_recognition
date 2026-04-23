import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn_case2 import HandGestureCNN_Case2
from src.utils.config_case2 import NUM_CLASSES

# --- Config ---
MODEL_PATH = "models/bestmodel_case2.pth"
DATA_DIR   = "data/hgr/train"        # folder containing one subfolder per class
IMG_SIZE   = 100               # must match training (RGB 100x100)

# Read class names from the data folder (alphabetical = same order as training)
CLASS_NAMES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

# --- Transform (must match training) ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # standard RGB normalisation
])

def predict(image_path):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureCNN_Case2()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()

    predicted_class = CLASS_NAMES[predicted_idx]

    # Try to get the true label from the folder name in the path
    true_label = None
    for cls in CLASS_NAMES:
        if cls in image_path:
            true_label = cls
            break

    # Print results
    print(f"\nImage:      {image_path}")
    if true_label:
        correct = "✓" if true_label == predicted_class else "✗"
        print(f"True label: {true_label}")
        print(f"Prediction: {predicted_class}  {correct}")
    else:
        print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.1%}\n")

    # Top 3 predictions
    top3 = probabilities.topk(3)
    print("Top 3:")
    for prob, idx in zip(top3.values, top3.indices):
        print(f"  {CLASS_NAMES[idx]:<30} {prob.item():.1%}")

    # Plot
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    title = f"Predicted: {predicted_class}\nConfidence: {confidence:.1%}"
    if true_label:
        title += f"\nTrue: {true_label}"
    plt.title(title, color="green" if true_label == predicted_class else "red")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict gesture from image (Case 2)")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    predict(args.image_path)