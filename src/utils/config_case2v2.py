import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HGR_TRAIN_DIR = os.path.join(DATA_DIR, "hgr", "train")
HGR_TEST_DIR = os.path.join(DATA_DIR, "hgr", "multi_user_test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH_CASE2 = os.path.join(MODEL_DIR, "bestmodel_case2.pth")

# Data
IMG_SIZE = 100       # HGR images are 100x100
NUM_CHANNELS = 3     # RGB
NUM_CLASSES = 4      # A, F, L, Y
BATCH_SIZE = 64
VAL_SPLIT = 0.15

# Training
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = "cpu"  # change to "mps" (macOS 12.3+) or "cuda" if available
