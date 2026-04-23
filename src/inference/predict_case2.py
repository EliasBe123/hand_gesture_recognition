import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.models.cnn_case2 import HandGestureCNN_Case2
from src.utils.config_case2 import BEST_MODEL_PATH_CASE2, IMG_SIZE, NUM_CLASSES, DEVICE

# Class names — matches ImageFolder alphabetical ordering of A, F, L, Y
CLASS_NAMES = ["A", "F", "L", "Y"]

# Transform (matches training — RGB, no grayscale)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict(image_path):
    device = torch.device(DEVICE)
    model = HandGestureCNN_Case2()
    model.load_state_dict(torch.load(BEST_MODEL_PATH_CASE2, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()

    predicted_class = CLASS_NAMES[predicted_idx]

    # Show result
    print(f"\nImage:      {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.1%}\n")

    # Show top 3 predictions
    top3 = probabilities.topk(min(3, NUM_CLASSES))
    print("Top predictions:")
    for prob, idx in zip(top3.values, top3.indices):
        print(f"  {CLASS_NAMES[idx]:<20} {prob.item():.1%}")

    # Plot the image with prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.1%}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict gesture from image (Case 2)")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    predict(args.image_path)
