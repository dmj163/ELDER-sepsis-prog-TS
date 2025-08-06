import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, \
    mean_squared_log_error,mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculate_classification_metrics(y_true, y_prob):
    """计算二分类或多分类指标"""
    y_pred = (y_prob > 0.5).astype(int) if y_prob.ndim == 1 else np.argmax(y_prob, axis=1)
    y_true_np = y_true # 假设已经是 numpy array

    metrics = {}
    try:
        metrics['auc'] = roc_auc_score(y_true_np, y_prob if y_prob.ndim == 1 else y_prob[:, 1])
    except ValueError:
        metrics['auc'] = float('nan') # 处理只有一个类别的特殊情况
    metrics['accuracy'] = accuracy_score(y_true_np, y_pred)
    metrics['f1'] = f1_score(y_true_np, y_pred, average='binary' if y_prob.ndim == 1 else 'macro')
    metrics['precision'] = precision_score(y_true_np, y_pred, average='binary' if y_prob.ndim == 1 else 'macro')
    metrics['recall'] = recall_score(y_true_np, y_pred, average='binary' if y_prob.ndim == 1 else 'macro')
    # 特异性需要手动计算 (TN / (TN + FP))
    # 这里简化，假设是二分类
    if y_prob.ndim == 1:
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred, labels=[0,1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['specificity'] = float('nan') # 多分类特异性计算复杂，暂时不实现
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """计算回归指标"""
    metrics = {}
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    # 添加msle
    metrics['msle'] = mean_squared_log_error(y_true, y_pred)
    return metrics

def plot_roc_curve(y_true, y_prob):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_precision_recall_curve(y_true, y_prob):
    """绘制PR曲线"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
