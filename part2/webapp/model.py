"""model.py — CNN 结构与权重加载

结构必须与 cnn_training.py 中的 ImprovedCNN 完全一致，否则 load_state_dict 会报 key 不匹配。
"""

import os

import torch
import torch.nn as nn

# webapp/ 的上一级就是 part2/，权重文件放在那里
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(_HERE), "best_mnist_cnn.pth")


class ImprovedCNN(nn.Module):
    """改进的 CNN：BatchNorm + Dropout + 四层卷积"""

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        return self.fc2(x)


def load_model(weights_path=None):
    """返回 (model, device)，model 已进入 eval 模式并放到可用设备上"""
    path = weights_path or DEFAULT_WEIGHTS
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到权重文件：{path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device
