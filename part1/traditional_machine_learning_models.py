"""
- SVM, Random Forest, KNN, Gradient Boosting, Decision Tree
- 调参前 / 调参后 对比
- 可视化输出
"""

import os
import warnings
import pickle
import time
import numpy as np

# 基于脚本所在目录锚定所有路径，无论从何处运行都能正确解析
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)  # 切换工作目录到脚本所在位置
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

# 全局配置
RANDOM_STATE = 42
TEST_SIZE = 0.30          # 论文使用 70:30 划分
BINARIZE_THRESHOLD = 0.211  # 论文标签二值化阈值
DATA_PATH = 'generated_dataset.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

# 颜色方案
MODEL_COLORS = {
    'SVM': '#e74c3c',
    'RF':  '#2ecc71',
    'KNN': '#3498db',
    'GB':  '#f39c12',
    'DT':  '#9b59b6',
}


def print_section(title):
    """打印分隔标题"""
    print()
    print(f"  {title}")
    print()


# 1. 数据加载与预处理

def load_and_preprocess():
    """加载数据，执行标签二值化、Min-Max归一化、70:30划分"""
    print_section("1. 数据加载与预处理")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] 数据集形状: {df.shape}")
    print(f"[INFO] 列名: {list(df.columns)}")

    # 确定特征列和目标列
    target_col = 'CR-corrosion defect'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y_raw = df[target_col].values

    # 标签二值化
    # CR <= 0.211 → 0 (low), CR > 0.211 → 1 (high)
    y = np.where(y_raw <= BINARIZE_THRESHOLD, 0, 1)

    n_low = np.sum(y == 0)
    n_high = np.sum(y == 1)
    print(f"[INFO] 二值化后类别分布: Low(0) = {n_low}, High(1) = {n_high}")
    print(f"[INFO] 比例: Low={n_low/len(y)*100:.1f}%, High={n_high/len(y)*100:.1f}%")

    # Min-Max 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 70:30 训练/测试划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] 训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # 保存scaler和特征列名（方便外部加载模型后知晓输入顺序）
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)

    return X_train, X_test, y_train, y_test, feature_cols, scaler


# 2. 定义模型

def get_models_default():
    """获取默认参数的5种模型（调参前）"""
    return {
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'RF':  RandomForestClassifier(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(),
        'GB':  GradientBoostingClassifier(random_state=RANDOM_STATE),
        'DT':  DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


def build_tuned_models(best_params):
    """根据 GridSearchCV 搜索结果动态构建调参后模型，替代硬编码参数"""
    models = {}

    # SVM
    svm_p = best_params.get('SVM', {})
    models['SVM'] = SVC(
        C=svm_p.get('C', 1.0),
        gamma=svm_p.get('gamma', 'scale'),
        kernel=svm_p.get('kernel', 'rbf'),
        probability=True, random_state=RANDOM_STATE
    )

    # Random Forest
    rf_p = best_params.get('RF', {})
    models['RF'] = RandomForestClassifier(
        n_estimators=rf_p.get('n_estimators', 100),
        max_depth=rf_p.get('max_depth', None),
        max_features=rf_p.get('max_features', 'sqrt'),
        bootstrap=rf_p.get('bootstrap', True),
        random_state=RANDOM_STATE
    )

    # KNN
    knn_p = best_params.get('KNN', {})
    models['KNN'] = KNeighborsClassifier(
        n_neighbors=knn_p.get('n_neighbors', 5),
        weights=knn_p.get('weights', 'uniform'),
        metric=knn_p.get('metric', 'minkowski')
    )

    # Gradient Boosting
    gb_p = best_params.get('GB', {})
    models['GB'] = GradientBoostingClassifier(
        n_estimators=gb_p.get('n_estimators', 100),
        max_depth=gb_p.get('max_depth', 3),
        max_features=gb_p.get('max_features', None),
        learning_rate=gb_p.get('learning_rate', 0.1),
        random_state=RANDOM_STATE
    )

    # Decision Tree
    dt_p = best_params.get('DT', {})
    models['DT'] = DecisionTreeClassifier(
        criterion=dt_p.get('criterion', 'gini'),
        max_depth=dt_p.get('max_depth', None),
        min_samples_split=dt_p.get('min_samples_split', 2),
        min_samples_leaf=dt_p.get('min_samples_leaf', 1),
        random_state=RANDOM_STATE
    )

    return models


# 3. 模型训练与评估

def evaluate_model(model, X_test, y_test):
    """计算所有评估指标（含加权平均和 per-class）"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'Accuracy':   accuracy_score(y_test, y_pred),
        'Precision':  precision_score(y_test, y_pred, average='weighted'),
        'Recall':     recall_score(y_test, y_pred, average='weighted'),
        'F1-Score':   f1_score(y_test, y_pred, average='weighted'),
        # Per-class 指标（class 0 = Low corrosion, class 1 = High corrosion）
        'Precision_Low':  precision_score(y_test, y_pred, average=None, labels=[0])[0],
        'Recall_Low':     recall_score(y_test, y_pred, average=None, labels=[0])[0],
        'F1_Low':         f1_score(y_test, y_pred, average=None, labels=[0])[0],
        'Precision_High': precision_score(y_test, y_pred, average=None, labels=[1])[0],
        'Recall_High':    recall_score(y_test, y_pred, average=None, labels=[1])[0],
        'F1_High':        f1_score(y_test, y_pred, average=None, labels=[1])[0],
    }

    if y_prob is not None:
        # Micro-average ROC-AUC
        y_bin = label_binarize(y_test, classes=[0, 1])
        metrics['ROC-AUC'] = roc_auc_score(y_bin, y_prob, average='micro')
    else:
        metrics['ROC-AUC'] = np.nan

    return metrics, y_pred, y_prob


def train_and_evaluate_models(models_dict, X_train, X_test, y_train, y_test, stage_name):
    """批量训练和评估模型，返回结果、模型、预测值和训练耗时"""
    results = {}
    models_trained = {}
    y_preds = {}
    y_probs = {}
    train_times = {}

    for name, model in models_dict.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        train_times[name] = elapsed
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        models_trained[name] = model
        y_preds[name] = y_pred
        y_probs[name] = y_prob

        print(f"  [{stage_name}] {name:5s} | Acc={metrics['Accuracy']:.4f} "
              f"| Prec={metrics['Precision']:.4f} | Rec={metrics['Recall']:.4f} "
              f"| F1={metrics['F1-Score']:.4f} | AUC={metrics['ROC-AUC']:.4f} "
              f"| Time={elapsed:.2f}s")

    return results, models_trained, y_preds, y_probs, train_times


# 4. GridSearchCV 超参数调优

def run_grid_search(X_train, y_train):
    """对每个模型执行GridSearchCV — 使用穷举网格搜索 + 5折交叉验证"""
    print_section("4. GridSearchCV 超参数调优 (穷举网格 + 5折CV)")

    param_grids = {
        'SVM': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear'],
            }
        },
        'RF': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 50],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
            }
        },
        'GB': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [15, 50, 100, 200],
                'max_depth': [3, 5, None],
                'max_features': ['sqrt', 'log2', None],
                'learning_rate': [0.01, 0.1, 0.2],
            }
        },
        'DT': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 50, 100, 150],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        },
    }

    best_params_all = {}
    best_scores_all = {}

    for name, cfg in param_grids.items():
        # 统计参数组合数
        n_combinations = 1
        for v in cfg['params'].values():
            n_combinations *= len(v)
        print(f"\n  [GridSearchCV] {name} ... ({n_combinations} combinations × 5-fold CV)")
        t0 = time.time()

        # 使用 GridSearchCV（穷举网格搜索）+ 5折交叉验证
        # n_jobs=-1 表示使用所有CPU核心
        search = GridSearchCV(
            cfg['model'], cfg['params'],
            cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)

        elapsed = time.time() - t0
        best_params_all[name] = search.best_params_
        best_scores_all[name] = search.best_score_
        print(f"    最优参数: {search.best_params_}")
        print(f"    最佳CV得分: {search.best_score_:.4f} | 耗时: {elapsed:.1f}s")

    return best_params_all, best_scores_all


# 5. 可视化函数
def plot_results_table(results_dict, stage_name, filename):
    """绘制结果对比表格图"""
    df = pd.DataFrame(results_dict).T
    df = df.round(4)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    col_labels = ['Model'] + list(df.columns)
    cell_text = []
    for model_name, row in df.iterrows():
        cell_text.append([model_name] + [f'{v:.4f}' for v in row.values])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # 高亮最优行
    best_model = df['Accuracy'].idxmax()
    for key, cell in table.get_celld().items():
        if key[1] == 0 and cell.get_text().get_text() == best_model:
            cell.set_facecolor('#b7e4c7')
        cell.set_edgecolor('#dee2e6')

    ax.set_title(f'{stage_name} - Results Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_confusion_matrices(models_trained, X_test, y_test, stage_name, filename):
    """绘制所有模型混淆矩阵"""
    n_models = len(models_trained)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(models_trained.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'High'],
                    yticklabels=['Low', 'High'],
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name} ({stage_name})', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    # 隐藏多余子图
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{stage_name} - Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_roc_auc_all(y_probs_dict, y_test, filename):
    """绘制所有模型ROC-AUC曲线对比"""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = list(MODEL_COLORS.values())

    for idx, (name, y_prob) in enumerate(y_probs_dict.items()):
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{name} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC-AUC Curves Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_precision_recall_all(y_probs_dict, y_test, filename):
    """绘制所有模型Precision-Recall曲线对比"""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = list(MODEL_COLORS.values())

    for idx, (name, y_prob) in enumerate(y_probs_dict.items()):
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, color=colors[idx % len(colors)], lw=2,
                label=f'{name}')

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_accuracy_comparison(results_before, results_after, filename):
    """绘制调参前后准确率对比柱状图"""
    models = list(results_before.keys())
    before_acc = [results_before[m]['Accuracy'] * 100 for m in models]
    after_acc = [results_after[m]['Accuracy'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before_acc, width, label='Before Tuning',
                   color='#74b9ff', edgecolor='white')
    bars2 = ax.bar(x + width/2, after_acc, width, label='After Tuning',
                   color='#e17055', edgecolor='white')

    # 在柱上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Comparison: Before vs After Tuning', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_feature_importance_rf(rf_model, feature_cols, filename):
    """绘制Random Forest特征重要性图"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(feature_cols)))
    ax.barh(range(len(feature_cols)), importances[indices], color=colors)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Random Forest - Feature Importance', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    # 标注数值
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_feature_importance_dt(dt_model, feature_cols, filename):
    """绘制Decision Tree特征重要性图"""
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(feature_cols)))
    ax.barh(range(len(feature_cols)), importances[indices], color=colors)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Decision Tree - Feature Importance', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_correlation_heatmap(df, feature_cols, y, filename):
    """绘制特征间相关性热力图"""
    df_plot = df[feature_cols].copy()
    df_plot['Target'] = y
    corr = df_plot.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


def plot_pairplot(df, feature_cols, y, filename):
    """绘制特征配对分布图（pairplot），按腐蚀类别着色"""
    # 采样以减少绘图时间
    sample_size = min(2000, len(df))
    df_sample = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()
    df_sample['Corrosion'] = y[df_sample.index]
    df_sample['Corrosion'] = df_sample['Corrosion'].map({0: 'Low', 1: 'High'})

    # 选择关键特征列
    key_features = feature_cols + ['Corrosion']
    df_plot = df_sample[key_features]

    try:
        g = sns.pairplot(df_plot, hue='Corrosion', diag_kind='kde',
                         palette={'Low': '#3498db', 'High': '#e74c3c'},
                         plot_kws={'alpha': 0.4, 's': 15})
        g.fig.suptitle('Feature Pair Plot (Sampled 2000)', fontsize=14, fontweight='bold', y=1.01)
        g.fig.set_size_inches(16, 14)
        g.fig.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {filename}")
    except Exception as e:
        print(f"  [WARN] Pairplot generation failed: {e}")


def plot_metric_radar(results_before, results_after, filename):
    """绘制调参前后指标雷达图"""
    models = list(results_before.keys())
    metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    n_metrics = len(metrics_keys)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        before_vals = [results_before[model_name][m] for m in metrics_keys]
        after_vals = [results_after[model_name][m] for m in metrics_keys]
        before_vals += before_vals[:1]
        after_vals += after_vals[:1]

        ax.plot(angles, before_vals, 'o-', linewidth=2, label='Before', color='#74b9ff')
        ax.fill(angles, before_vals, alpha=0.1, color='#74b9ff')
        ax.plot(angles, after_vals, 'o-', linewidth=2, label='After', color='#e17055')
        ax.fill(angles, after_vals, alpha=0.1, color='#e17055')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_keys, fontsize=9)
        ax.set_ylim(0.8, 1.0)
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', fontsize=8)

    # 隐藏多余子图
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Metrics Radar Chart: Before vs After Tuning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {filename}")


# 6. 保存结果

def save_results_csv(results_dict, filename):
    """保存评估结果到CSV"""
    df = pd.DataFrame(results_dict).T
    df.to_csv(os.path.join(RESULTS_DIR, filename))
    print(f"  [SAVED] {filename}")


def save_models(models_dict):
    """保存每个模型到磁盘（.pkl）"""
    for name, model in models_dict.items():
        fname = f"{name.lower()}_model.pkl"
        with open(os.path.join(MODELS_DIR, fname), 'wb') as f:
            pickle.dump(model, f)
        print(f"  [SAVED] models/{fname}")


def save_best_params(best_params, best_scores, filename):
    """保存GridSearchCV最优参数和CV得分"""
    with open(os.path.join(RESULTS_DIR, filename), 'w', encoding='utf-8') as f:
        f.write("GridSearchCV Best Parameters\n")
        f.write("\n")
        for name, params in best_params.items():
            f.write(f"[{name}]\n")
            cv_score = best_scores.get(name, 0.0)
            f.write(f"  Best CV Score (accuracy): {cv_score:.4f}\n")
            for k, v in params.items():
                f.write(f"  {k} = {v}\n")
            f.write("\n")
    print(f"  [SAVED] {filename}")


# Main

def main():
    print("  论文复现实验 - Anomaly Detection for Oil & Gas Pipelines")
    print("  阶段一：传统机器学习（SVM, RF, KNN, GB, DT）")

    # 1. 数据预处理
    X_train, X_test, y_train, y_test, feature_cols, scaler = load_and_preprocess()

    # 2. 调参前训练与评估
    print_section("2. 调参前模型训练与评估")
    models_default = get_models_default()
    results_before, trained_before, preds_before, probs_before, times_before = train_and_evaluate_models(
        models_default, X_train, X_test, y_train, y_test, "BEFORE"
    )

    # 3. 保存调参前结果
    save_results_csv(results_before, 'results_before_tuning.csv')
    plot_results_table(results_before, 'Before Tuning', 'comparison_table_before.png')

    # 4. GridSearchCV调参
    t_total_start = time.time()
    best_params, best_scores = run_grid_search(X_train, y_train)
    save_best_params(best_params, best_scores, 'best_params.txt')

    # 5. 调参后训练与评估
    print_section("5. 调参后模型训练与评估（使用 GridSearchCV 搜索结果）")
    models_tuned = build_tuned_models(best_params)
    results_after, trained_after, preds_after, probs_after, times_after = train_and_evaluate_models(
        models_tuned, X_train, X_test, y_train, y_test, "AFTER"
    )

    # 6. 保存调参后结果
    save_results_csv(results_after, 'results_after_tuning.csv')
    plot_results_table(results_after, 'After Tuning', 'comparison_table_after.png')

    # 7. 保存模型文件
    print_section("7. 保存模型文件")
    save_models(trained_after)

    # 8. 可视化
    print_section("8. 生成可视化图表")

    # 8.1 混淆矩阵
    plot_confusion_matrices(trained_before, X_test, y_test, 'Before Tuning', 'confusion_matrix_before.png')
    plot_confusion_matrices(trained_after, X_test, y_test, 'After Tuning', 'confusion_matrix_after.png')

    # 8.2 ROC-AUC 曲线
    plot_roc_auc_all(probs_before, y_test, 'roc_auc_before.png')
    plot_roc_auc_all(probs_after, y_test, 'roc_auc_after.png')

    # 8.3 Precision-Recall 曲线
    plot_precision_recall_all(probs_before, y_test, 'precision_recall_before.png')
    plot_precision_recall_all(probs_after, y_test, 'precision_recall_after.png')

    # 8.4 准确率对比柱状图
    plot_accuracy_comparison(results_before, results_after, 'accuracy_comparison.png')

    # 8.5 指标雷达图
    plot_metric_radar(results_before, results_after, 'radar_metrics.png')

    # 8.6 特征重要性 (RF & DT)
    if 'RF' in trained_after:
        plot_feature_importance_rf(trained_after['RF'], feature_cols, 'feature_importance_rf.png')
    if 'DT' in trained_after:
        plot_feature_importance_dt(trained_after['DT'], feature_cols, 'feature_importance_dt.png')

    # 8.7 相关性热力图
    df = pd.read_csv(DATA_PATH)
    target_col = 'CR-corrosion defect'
    feature_cols_list = [c for c in df.columns if c != target_col]
    y_labels = np.where(df[target_col].values <= BINARIZE_THRESHOLD, 0, 1)
    plot_correlation_heatmap(df, feature_cols_list, y_labels, 'correlation_heatmap.png')

    # 8.8 Pairplot
    plot_pairplot(df, feature_cols_list, y_labels, 'pairplot_features.png')

    # 9. 最终汇总
    print_section("9. 最终结果汇总")
    print("\n  调参前 (Table 3)")
    df_before = pd.DataFrame(results_before).T
    print(df_before.to_string())

    print("\n  调参后 (Table 4)")
    df_after = pd.DataFrame(results_after).T
    print(df_after.to_string())

    print("\n  GridSearchCV 最佳CV得分")
    for name, score in best_scores.items():
        print(f"  {name:5s}: CV Accuracy = {score:.4f}")

    print("\n  训练耗时 (s)")
    for name in models_default.keys():
        print(f"  {name:5s}: Before={times_before.get(name, 0):.2f}s  |  After={times_after.get(name, 0):.2f}s")

    t_total = time.time() - t_total_start
    print()
    print(f"  总耗时: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  实验完成！所有结果已保存到 '{RESULTS_DIR}/' 和 '{MODELS_DIR}/'")
    print()


if __name__ == '__main__':
    main()