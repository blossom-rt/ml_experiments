"""
load_mnist.py — 加载并读取 MNIST 数据集

功能：
  1. 从 data/MNIST/raw/ 读取 IDX 格式文件
  2. 优先使用 idx2numpy，未安装则回退到 struct 手动解析
  3. 输出训练集/测试集的形状、标签数量
  4. Matplotlib 可视化前几张图像及对应标签
  5. 统计标签分布
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

# 锚定到脚本所在目录，与 test.py 下载的 data/ 保持一致
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# 0. 配置 Matplotlib 中文字体（Windows 优先使用系统中文字体）
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei",   # 微软雅黑
    "SimHei",            # 黑体
    "KaiTi",             # 楷体
    "FangSong",          # 仿宋
    "SimSun",            # 宋体
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "AR PL UMing CN",
]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_cn_font = None
for _f in _CN_FONT_CANDIDATES:
    if _f in _available:
        _cn_font = _f
        break

if _cn_font:
    plt.rcParams["font.sans-serif"] = [_cn_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[字体] 已启用中文字体: {_cn_font}")
else:
    print("[字体] 未找到中文字体，图表标题将使用英文")

# 1. 配置路径
DATA_DIR = os.path.join("data", "MNIST", "raw")

TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES  = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS  = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

# 2. 读取函数：优先 idx2numpy，回退 struct
def load_with_idx2numpy(filepath):
    """使用 idx2numpy 加载 IDX 文件"""
    import idx2numpy
    return idx2numpy.convert_from_file(filepath)


def load_with_struct(filepath):
    """使用 struct 手动解析 IDX 文件格式

    IDX 文件格式:
      - 魔数: 4 字节（前 2 字节为 0，第 3 字节为数据类型，第 4 字节为维度数）
      - 各维度大小: 每个维度 4 字节（大端序 int32）
      - 数据: 按大端序存储
    """
    with open(filepath, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        # magic 的高 2 字节为 0，低 2 字节: [数据类型(1B), 维度数(1B)]
        data_type = (magic >> 8) & 0xFF  # 0x08=uint8, 0x09=int8, 0x0B=int16, 0x0C=int32, 0x0D=float32, 0x0E=float64
        num_dims  = magic & 0xFF

        # 读取各维度大小
        dims = []
        for _ in range(num_dims):
            dims.append(struct.unpack(">I", f.read(4))[0])

        # 读取数据
        raw = f.read()

    # 根据数据类型解析
    dtype_map = {
        0x08: ("B", np.uint8),    # unsigned byte
        0x09: ("b", np.int8),     # signed byte
        0x0B: (">h", np.int16),   # short (2 bytes)
        0x0C: (">i", np.int32),   # int (4 bytes)
        0x0D: (">f", np.float32), # float (4 bytes)
        0x0E: (">d", np.float64), # double (8 bytes)
    }

    if data_type not in dtype_map:
        raise ValueError(f"不支持的数据类型: 0x{data_type:02X}")

    fmt_char, np_dtype = dtype_map[data_type]
    elem_size = struct.calcsize(fmt_char.replace(">", ""))
    total = 1
    for d in dims:
        total *= d

    expected_size = total * elem_size
    if len(raw) != expected_size:
        raise ValueError(f"文件大小不匹配: 期望 {expected_size} 字节，实际 {len(raw)} 字节")

    fmt = f">{fmt_char * total}"
    data = struct.unpack(fmt, raw)
    arr = np.array(data, dtype=np_dtype).reshape(dims)
    return arr


def load_idx(filepath):
    """智能加载 IDX 文件：优先 idx2numpy，失败则 struct"""
    try:
        return load_with_idx2numpy(filepath)
    except ImportError:
        print(f"[提示] idx2numpy 未安装，使用 struct 手动解析: {os.path.basename(filepath)}")
        return load_with_struct(filepath)


# 3. 主流程
def main():
    print("=" * 60)
    print("MNIST 数据集加载与可视化")
    print("=" * 60)

    # 检查文件是否存在
    for fpath in [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"文件不存在: {fpath}")

    # 加载数据
    print("\n正在加载训练图像...")
    X_train = load_idx(TRAIN_IMAGES)
    print("正在加载训练标签...")
    y_train = load_idx(TRAIN_LABELS)

    print("正在加载测试图像...")
    X_test = load_idx(TEST_IMAGES)
    print("正在加载测试标签...")
    y_test = load_idx(TEST_LABELS)

    # 输出形状与数量
    print("\n" + "-" * 40)
    print(f"训练集图像形状: {X_train.shape}   (样本数 × 高度 × 宽度)")
    print(f"训练集标签数量: {len(y_train)}    标签形状: {y_train.shape}")
    print(f"测试集图像形状: {X_test.shape}   (样本数 × 高度 × 宽度)")
    print(f"测试集标签数量: {len(y_test)}    标签形状: {y_test.shape}")
    print("-" * 40)

    # 标签分布统计
    print("\n训练集标签分布:")
    train_counter = Counter(y_train.tolist())
    for label in sorted(train_counter):
        print(f"  数字 {label}: {train_counter[label]:>5} 个  ({train_counter[label] / len(y_train) * 100:.1f}%)")

    print("\n测试集标签分布:")
    test_counter = Counter(y_test.tolist())
    for label in sorted(test_counter):
        print(f"  数字 {label}: {test_counter[label]:>5} 个  ({test_counter[label] / len(y_test) * 100:.1f}%)")

    # 可视化前几张图像
    num_show = 10  # 显示前 10 张
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(num_show):
        axes[i].imshow(X_train[i], cmap="gray")
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")

    fig.suptitle("MNIST 训练集前 10 张图像", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # 额外：标签分布柱状图
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels_list = sorted(train_counter)
    train_counts = [train_counter[l] for l in labels_list]
    test_counts  = [test_counter[l] for l in labels_list]

    ax1.bar(labels_list, train_counts, color="steelblue", edgecolor="black")
    ax1.set_title("训练集标签分布", fontsize=13, fontweight="bold")
    ax1.set_xlabel("数字类别")
    ax1.set_ylabel("样本数量")
    ax1.set_xticks(labels_list)

    ax2.bar(labels_list, test_counts, color="coral", edgecolor="black")
    ax2.set_title("测试集标签分布", fontsize=13, fontweight="bold")
    ax2.set_xlabel("数字类别")
    ax2.set_ylabel("样本数量")
    ax2.set_xticks(labels_list)

    fig2.suptitle("MNIST 标签分布对比", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n完成！")


if __name__ == "__main__":
    main()
