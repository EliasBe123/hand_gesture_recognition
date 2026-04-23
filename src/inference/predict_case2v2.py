import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
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

def predict(image_path1, image_path2):
    device = torch.device(DEVICE)
    model = HandGestureCNN_Case2()
    model.load_state_dict(torch.load(BEST_MODEL_PATH_CASE2, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    # Load and preprocess both images
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")
    t1 = transform(img1)
    t2 = transform(img2)

    # Stack into (1, 2, C, H, W) — batch of 1 pair
    pair_tensor = torch.stack([t1, t2], dim=0).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(pair_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()

    predicted_class = CLASS_NAMES[predicted_idx]

    # Show result
    print(f"\nImage 1:    {image_path1}")
    print(f"Image 2:    {image_path2}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.1%}\n")

    # Show top predictions
    top3 = probabilities.topk(min(3, NUM_CLASSES))
    print("Top predictions:")
    for prob, idx in zip(top3.values, top3.indices):
        print(f"  {CLASS_NAMES[idx]:<20} {prob.item():.1%}")

    # Plot both images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img1)
    ax1.set_title("Frame 1")
    ax1.axis("off")
    ax2.imshow(img2)
    ax2.set_title("Frame 2")
    ax2.axis("off")
    fig.suptitle(f"Predicted: {predicted_class}  ({confidence:.1%})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict gesture from an image pair (Case 2)")
    parser.add_argument("image1", type=str, help="Path to the first image")
    parser.add_argument("image2", type=str, help="Path to the second image")
    args = parser.parse_args()
    predict(args.image1, args.image2)
