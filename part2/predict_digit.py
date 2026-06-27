"""
predict_digit.py — 使用训练好的 CNN 模型识别手写数字图片

用法：
    python predict_digit.py <图片路径>
    例如: python predict_digit.py my_digit.png （首先要将目录切换到part2/）

要求：
    - 图片最好是白底黑字的 28×28 左右的手写数字
    - 支持 PNG/JPG/BMP 等常见格式
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# 锚定到脚本所在目录，确保默认模型路径正确
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# 配置中文字体（Windows 优先）
_CN_FONTS = ["Microsoft YaHei", "SimHei", "KaiTi", "FangSong", "SimSun"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_cn_font = next((f for f in _CN_FONTS if f in _available), None)
if _cn_font:
    plt.rcParams["font.sans-serif"] = [_cn_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 1. 模型定义（必须与 test.py 中完全一致）
# ---------------------------------------------------------------------------
class ImprovedCNN(nn.Module):
    """改进的CNN模型：添加BatchNorm + Dropout + 更深的卷积层"""
    def __init__(self, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

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
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 2. 图像预处理（与训练时一致）
# ---------------------------------------------------------------------------
def preprocess_image(image_path):
    """
    预处理输入图片：
      1. 转为灰度图
      2. 缩放到 28×28
      3. 颜色反转（白底黑字 → 黑底白字，与 MNIST 一致）
      4. 归一化 (mean=0.1307, std=0.3081，与训练时一致)
    """
    img = Image.open(image_path).convert("L")          # 灰度图
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # 缩放到 28×28

    # 显示预处理后的图像（供用户确认）
    img_array = np.array(img, dtype=np.float32)

    # 判断是否需要反转颜色
    # MNIST 是黑底白字，大多数手写图片是白底黑字
    # 如果边缘像素较亮（白底），则反转
    edge_mean = (img_array[0, :].mean() + img_array[-1, :].mean() +
                 img_array[:, 0].mean() + img_array[:, -1].mean()) / 4
    if edge_mean > 127:  # 白底 → 需要反转
        img_array = 255.0 - img_array

    # 归一化（与训练时一致）
    img_array = img_array / 255.0
    img_array = (img_array - 0.1307) / 0.3081

    # 转为 tensor: (1, 1, 28, 28)
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, img_array


# ---------------------------------------------------------------------------
# 3. 预测
# ---------------------------------------------------------------------------
def predict(image_path, model_path="best_mnist_cnn.pth"):
    """加载模型并预测图片中的数字"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ImprovedCNN().to(device)
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 test.py 训练模型，或确保 best_mnist_cnn.pth 在当前目录")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 预处理图片
    tensor, processed_img = preprocess_image(image_path)
    tensor = tensor.to(device)

    # 推理
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_digit = predicted.item()
    confidence_pct = confidence.item() * 100

    # ---------- 输出结果 ----------
    print("=" * 50)
    print(f"预测结果: 数字 ** {predicted_digit} **")
    print(f"置信度:   {confidence_pct:.2f}%")
    print("-" * 50)
    print("各类别概率:")
    probs = probabilities.cpu().numpy().flatten()
    for digit, prob in sorted(enumerate(probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {digit}: {bar} {prob:.4f} ({prob*100:.1f}%)")
    print("=" * 50)

    # ---------- 可视化 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 左图：预处理后的图像
    ax1.imshow(processed_img, cmap="gray")
    ax1.set_title(f"预处理后的图像 (28×28)", fontsize=12)
    ax1.axis("off")

    # 右图：概率柱状图
    digits = list(range(10))
    colors = ["#4CAF50" if d == predicted_digit else "#BDBDBD" for d in digits]
    ax2.bar(digits, probs, color=colors, edgecolor="black")
    ax2.set_title(f"各类别概率分布", fontsize=12)
    ax2.set_xlabel("数字类别")
    ax2.set_ylabel("概率")
    ax2.set_xticks(digits)
    ax2.set_ylim(0, 1.05)

    # 在最高柱上标注预测结果
    ax2.annotate(
        f"预测: {predicted_digit}\n置信度: {confidence_pct:.1f}%",
        xy=(predicted_digit, confidence.item()),
        xytext=(predicted_digit + 1.5, max(0, confidence.item() - 0.15)),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=11, color="red", fontweight="bold",
        verticalalignment="top",
    )

    fig.suptitle("手写数字识别结果", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return predicted_digit, confidence_pct


# ---------------------------------------------------------------------------
# 4. 主入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict_digit.py <图片路径>")
        print("示例: python predict_digit.py my_digit.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"[错误] 图片文件不存在: {image_path}")
        sys.exit(1)

    predict(image_path)
