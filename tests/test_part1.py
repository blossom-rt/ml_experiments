"""part1 传统机器学习流程测试

覆盖：数据集形状、二值化阈值、加载预处理划分、evaluate_model 指标正确性、
per-class 指标参数化。
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 导入 part1 模块（模块内会 os.chdir 到自身目录）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'part1'))
import traditional_machine_learning_models as tml


def test_dataset_shape():
    df = pd.read_csv(tml.DATA_PATH)
    assert df.shape == (10292, 9)
    assert df.isnull().sum().sum() == 0


def test_binarization_threshold():
    y_raw = np.array([0.0, 0.2, 0.211, 0.212, 1.0])
    y = np.where(y_raw <= tml.BINARIZE_THRESHOLD, 0, 1)
    assert list(y) == [0, 0, 0, 1, 1]


def test_load_and_preprocess():
    X_train, X_test, y_train, y_test, feature_cols, scaler = tml.load_and_preprocess()
    total = len(X_train) + len(X_test)
    assert total == 10292
    # 10292 * 0.30 = 3087.6，sklearn 取整为 3088（random_state=42 确定）
    assert len(X_test) == 3088
    assert len(X_train) == 7204
    assert len(feature_cols) == 8
    # MinMax 归一化后落在 [0, 1]
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0


def test_evaluate_model_metrics():
    X_train, X_test, y_train, y_test, _, _ = tml.load_and_preprocess()
    # 小样本快速训练
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_train), size=500, replace=False)
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X_train[idx], y_train[idx])

    metrics, y_pred, y_prob = tml.evaluate_model(clf, X_test, y_test)

    # 所有指标落在 [0, 1]
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k} 越界: {v}"

    # per-class 指标键名存在（默认 Low/High）
    for k in ['Precision_Low', 'Recall_Low', 'F1_Low',
              'Precision_High', 'Recall_High', 'F1_High']:
        assert k in metrics, f"缺少键 {k}"

    # 简化后的 ROC-AUC 与直接调用 roc_auc_score 一致
    assert abs(metrics['ROC-AUC'] - roc_auc_score(y_test, y_prob)) < 1e-12


def test_evaluate_model_parameterized_labels():
    """per-class 指标键名应随 class_names 参数变化，而非硬编码"""

    class Dummy:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)) * 0.8, np.ones(len(X)) * 0.2])

    y = np.array([0, 1, 0, 1])
    metrics, _, _ = tml.evaluate_model(
        Dummy(), np.zeros((4, 2)), y,
        class_labels=(0, 1), class_names=('A', 'B')
    )
    assert 'Precision_A' in metrics
    assert 'Recall_B' in metrics
    assert 'Precision_Low' not in metrics  # 不再硬编码 Low/High