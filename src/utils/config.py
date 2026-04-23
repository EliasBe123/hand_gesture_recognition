import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_TRAIN_DIR = os.path.join(DATA_DIR, "raw", "train", "train")
RAW_TEST_DIR = os.path.join(DATA_DIR, "raw", "test", "test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "bestmodel.pth")

# Data
IMG_SIZE = 50
NUM_CLASSES = 20
BATCH_SIZE = 64
VAL_SPLIT = 0.15  # fraction of training data used for validation

# Training
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = "cpu"  # change to "mps" (macOS 12.3+) or "cuda" if available
