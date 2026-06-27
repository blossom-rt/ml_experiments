# ML Experiments

两个实验：传统分类器用于油气管道腐蚀预测，以及基于 CNN 的手写数字识别（MNIST）。

---

## 安装依赖

```bash
pip install -r requirements.txt
```

如需 GPU 训练（实验二），请从 [pytorch.org](https://pytorch.org) 安装带 CUDA 的 PyTorch。

> 所有脚本均会切换工作目录到自身所在位置，从任意路径运行均可。
> MNIST 数据集在首次运行 `cnn_training.py` 时会自动下载。若因网络限制下载失败，可手动从 [Google Cloud Storage](https://storage.googleapis.com/cvdf-datasets/mnist/) 下载 4 个 IDX gzip 文件，解压到 `part2/data/MNIST/raw/` 即可。

---

## 实验一：管道腐蚀缺陷预测（`part1/`）

比较 5 种 scikit-learn 分类器在调参前后的表现，任务为油气管道腐蚀缺陷二分类。

**分类器**：SVM（RBF 核）、随机森林、K 近邻、梯度提升、决策树

**流程**：
1. 加载数据 → 标签二值化（阈值 0.211）→ MinMax 归一化 → 70/30 分层划分
2. 用默认参数训练 5 个模型
3. 对每个模型执行 GridSearchCV（穷举网格，5 折交叉验证）
4. 用最优参数重建并重新训练模型
5. 评估指标：准确率、精确率、召回率、F1 值（加权平均 + 按类别）、ROC-AUC
6. 生成 12 张以上对比图表（混淆矩阵、ROC/PR 曲线、雷达图、特征重要性等）

**运行**：
```bash
cd part1
python traditional_machine_learning_models.py
```

**数据集**（`part1/generated_dataset.csv`）：10,291 条样本，8 个特征 — 井口温度、井口压力、天然气/石油/水产率、BSW 含水率、CO₂ 摩尔分数、气体密度。目标值为腐蚀速率（CR）。

---

## 实验二：MNIST 手写数字识别（`part2/`）

改进的 CNN 模型在 MNIST 上训练手写数字识别，附带独立的推理脚本。

### 模型结构

```
ImprovedCNN（871K 参数）
├─ 卷积块 1：Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout(0.25)
├─ 卷积块 2：Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout(0.25)
└─ 分类器：   FC(3136→256) → BN → ReLU → Dropout(0.5) → FC(256→10)
```

### 训练

- 数据增强：随机旋转（±10°）+ 随机仿射平移（±10%）
- 优化器：Adam（lr=0.001）+ ReduceLROnPlateau 调度器（factor=0.5, patience=2）
- 批次大小：64，轮数：10
- TensorBoard 日志记录（`runs/mnist_cnn_improved/`）
- 保存最佳检查点为 `best_mnist_cnn.pth`

**运行**：
```bash
cd part2
python cnn_training.py
```

### 推理

对任意手写数字图片进行预测：

```bash
cd part2
python predict_digit.py path/to/your_digit.png
```

脚本自动完成：
- 转为灰度图，缩放到 28×28
- 若图片为白底黑字则自动反色（转为 MNIST 的黑底白字格式）
- 用 MNIST 统计量（mean=0.1307, std=0.3081）归一化
- 显示预测结果及概率柱状图

### 数据加载器

`load_mnist.py` 读取原始 MNIST IDX 文件（`data/MNIST/raw/`），支持 `idx2numpy`（可选，未安装则自动回退到 `struct` 手动解析），并可视化样本图像及标签分布。
