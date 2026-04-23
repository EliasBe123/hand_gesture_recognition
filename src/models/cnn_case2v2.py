import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config_case2v2 import NUM_CLASSES, NUM_CHANNELS


class CNNFeatureExtractor(nn.Module):
    """CNN backbone that extracts a feature vector from a single frame."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)

        # After 4 pool layers: 100 -> 50 -> 25 -> 12 -> 6
        self.feature_dim = 256 * 6 * 6

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)  # (batch, feature_dim)
        return x


class HandGestureCNN_Case2(nn.Module):
    """CNN + LSTM: extracts features from each frame, then classifies the pair."""

    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        feature_dim = self.cnn.feature_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # x shape: (batch, seq_len=2, C, H, W)
        batch_size, seq_len, C, H, W = x.shape

        # Extract features for each frame
        x = x.view(batch_size * seq_len, C, H, W)  # merge batch & seq
        features = self.cnn(x)                       # (batch*seq, feature_dim)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq, feature_dim)

        # LSTM over the 2-frame sequence
        lstm_out, _ = self.lstm(features)            # (batch, seq, 256)
        last_hidden = lstm_out[:, -1, :]             # take last timestep

        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
