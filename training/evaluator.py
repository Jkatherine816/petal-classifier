"""
training/evaluator.py - 模型评估

产出：混淆矩阵图、分类报告（csv/json）、各类别准确率条形图、置信度分布图。
"""

import os
import json
import logging

import numpy as np
import torch
from core import mpl_setup as _mpl_setup  # noqa: F401  配置 matplotlib 中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, data_loader, device, classes, out_dir=None):
    """
    在 data_loader 上评估模型。

    返回 metrics：
        {accuracy, precision, recall, f1, confusion_matrix,
         per_class: {类别: {accuracy, precision, recall, f1, support}},
         predictions: [{index, true, pred, confidence}], }
    若提供 out_dir，同时保存图表与报告文件。
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = float(accuracy_score(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(len(classes))))

    # 每类指标
    per_class = {}
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(classes))),
        zero_division=0)
    for i, cls in enumerate(classes):
        mask = all_labels == i
        per_class[cls] = {
            'accuracy': float(accuracy_score(all_labels[mask],
                                             all_preds[mask])) if mask.any() else 0.0,
            'precision': float(p_c[i]), 'recall': float(r_c[i]),
            'f1': float(f_c[i]), 'support': int(s_c[i]),
        }

    predictions = [{
        'index': i,
        'true': classes[int(t)],
        'pred': classes[int(p)],
        'confidence': float(all_probs[i, int(p)]),
    } for i, (t, p) in enumerate(zip(all_labels, all_preds))]

    metrics = {
        'accuracy': acc, 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1),
        'confusion_matrix': cm.tolist(), 'per_class': per_class,
        'predictions': predictions,
    }

    logger.info('评估完成: acc=%.4f f1=%.4f 样本数=%d', acc, f1, len(all_labels))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _save_artifacts(metrics, classes, all_probs, all_labels, all_preds,
                        out_dir)
    return metrics


# ---------------------------------------------------------------- 图表
def _save_artifacts(metrics, classes, all_probs, all_labels, all_preds, out_dir):
    cm = np.array(metrics['confusion_matrix'])

    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('混淆矩阵')
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, classes, rotation=45, ha='right')
    ax.set_yticks(ticks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 每类准确率条形图
    names = list(metrics['per_class'].keys())
    accs = [metrics['per_class'][c]['accuracy'] for c in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, accs, color='#45B7D1')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('准确率')
    ax.set_title('各类别准确率')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_class_accuracy.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 置信度分布（正确 vs 错误）
    correct_conf = [all_probs[i, all_preds[i]] for i in range(len(all_preds))
                    if all_preds[i] == all_labels[i]]
    wrong_conf = [all_probs[i, all_preds[i]] for i in range(len(all_preds))
                  if all_preds[i] != all_labels[i]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    if correct_conf:
        ax.hist(correct_conf, bins=bins, alpha=0.7, label=f'正确 ({len(correct_conf)})',
                color='#4ECDC4')
    if wrong_conf:
        ax.hist(wrong_conf, bins=bins, alpha=0.7, label=f'错误 ({len(wrong_conf)})',
                color='#FF6B6B')
    ax.set_xlabel('置信度')
    ax.set_ylabel('样本数')
    ax.set_title('预测置信度分布')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confidence_distribution.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 报告文件
    with open(os.path.join(out_dir, 'evaluation_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    import csv
    with open(os.path.join(out_dir, 'classification_report.csv'), 'w',
              encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['类别', '准确率', '精确率', '召回率', 'F1', '样本数'])
        for cls, m in metrics['per_class'].items():
            writer.writerow([cls, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                             f"{m['recall']:.4f}", f"{m['f1']:.4f}", m['support']])
        writer.writerow(['总体', f"{metrics['accuracy']:.4f}",
                         f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
                         f"{metrics['f1']:.4f}",
                         sum(m['support'] for m in metrics['per_class'].values())])

    # 错分样本清单（供 GUI 错分浏览器使用）
    wrong = [p for p in metrics['predictions'] if p['true'] != p['pred']]
    with open(os.path.join(out_dir, 'misclassified.json'), 'w',
              encoding='utf-8') as f:
        json.dump(wrong, f, indent=2, ensure_ascii=False)

    logger.info('评估产物已保存到 %s', out_dir)


def plot_training_curves(history, out_path):
    """绘制训练曲线（loss / acc / lr）。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history['train_loss'], label='训练')
    axes[0].plot(history['val_loss'], label='验证')
    axes[0].set_title('损失曲线'); axes[0].set_xlabel('轮次'); axes[0].legend()

    axes[1].plot(history['train_acc'], label='训练')
    axes[1].plot(history['val_acc'], label='验证')
    axes[1].set_title('准确率曲线'); axes[1].set_xlabel('轮次')
    axes[1].set_ylabel('准确率 (%)'); axes[1].legend()

    axes[2].plot(history['learning_rate'])
    axes[2].set_title('学习率'); axes[2].set_xlabel('轮次')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path
