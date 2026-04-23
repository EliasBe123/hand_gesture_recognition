# Hand Gesture Recognition

CNN-based hand gesture recognition with two dataset options.

## Setup

```bash
git clone https://github.com/EliasBe123/hand_gesture_recognition.git
cd hand_gesture_recognition
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision kagglehub numpy<2 pillow matplotlib
```

## Datasets

### Case 1 — 20-class grayscale gestures (50×50)

```bash
python -c "
import kagglehub, shutil, os
path = kagglehub.dataset_download('aryarishabh/hand-gesture-recognition-dataset')
print('Downloaded to:', path)
shutil.copytree(path, 'data/raw', dirs_exist_ok=True)
"
```

Train:
```bash
python -m src.training.train
```

### Case 2 — 4-class RGB gestures (100×100, A/F/L/Y)

```bash
python -c "
import kagglehub, shutil, os
path = kagglehub.dataset_download('joelbaptista/hand-gestures-for-human-robot-interaction')
src = os.path.join(path, 'HGR dataset')
shutil.copytree(src, 'data/hgr', dirs_exist_ok=True)
"
```

Train:
```bash
python -m src.training.train_case2v2
```

## Inference

Predict on an image (case 1 model):
```bash
python -m src.inference.predict path/to/image.jpg
```

## Project Structure

```
├── data/
│   ├── raw/              # Case 1 dataset (20 classes, grayscale)
│   └── hgr/              # Case 2 dataset (4 classes, RGB)
├── models/
│   ├── bestmodel.pth     # Best model from case 1
│   └── bestmodel_case2.pth
├── src/
│   ├── data/
│   │   ├── loader.py             # DataLoader for case 1
│   │   └── loader_case2v2.py     # DataLoader for case 2 (paired)
│   ├── models/
│   │   ├── cnn.py                 # CNN for case 1 (1×50×50, 20 classes)
│   │   └── cnn_case2v2.py         # CNN+LSTM for case 2 (3×100×100, 4 classes)
│   ├── training/
│   │   ├── train.py               # Training script for case 1
│   │   └── train_case2v2.py       # Training script for case 2
│   ├── inference/
│   │   ├── predict.py
│   │   └── predict_case2v2.py
│   └── utils/
│       ├── config.py              # Config for case 1
│       └── config_case2v2.py      # Config for case 2
└── README.md
```