"""
training/trainer.py - 统一训练器（baseline / 微调共用）

训练循环与脚本解耦：通过回调对接 GUI 或命令行。
    on_epoch_end(epoch, metrics, history)  - 每个 epoch 结束（GUI 实时曲线）
    on_log(message)                        - 普通日志（默认走 logging）
    should_stop()                          - 返回 True 则当前 epoch 结束后停止

修复与增强：
    - 类别权重直接由 create_dataloaders 返回（不再依赖 Subset.labels 这种取不到的属性）；
    - 支持混合精度训练（AMP，仅 CUDA 时启用）；
    - 微调：仅兼容同构 checkpoint，按模块冻结（替代旧版按参数数量比例冻结）；
    - best/final checkpoint 均为自描述格式。
"""

import os
import copy
import logging

import numpy as np
import torch
import torch.nn as nn

from models.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, device=None, criterion=None, optimizer=None,
                 scheduler=None, amp=False, on_log=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = bool(amp) and self.device.type == 'cuda'
        self.on_log = on_log or (lambda msg: logger.info(msg))
        self.scaler = torch.amp.GradScaler('cuda') if self.amp else None

    # ------------------------------------------------------------ 单轮
    def _run_epoch(self, loader, train=True):
        model = self.model
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=self.amp):
                    outputs, _ = model(inputs)
                    loss = self.criterion(outputs, labels)

                if train:
                    if self.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)

    # ------------------------------------------------------------ 训练主循环
    def fit(self, train_loader, val_loader, epochs, save_dir=None,
            model_type=None, model_config=None, preprocess=None,
            classes=None, on_epoch_end=None, should_stop=None):
        """
        返回：(history, best_metrics, stopped_early)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        history = {'train_loss': [], 'train_acc': [],
                   'val_loss': [], 'val_acc': [], 'learning_rate': []}
        best_val_acc = -1.0
        best_metrics = {}
        stopped = False

        has_val = val_loader is not None and len(val_loader.dataset) > 0

        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            if has_val:
                val_loss, val_acc = self._run_epoch(val_loader, train=False)
            else:
                val_loss, val_acc = float('nan'), float('nan')
                # 无验证集时按训练精度保存最佳
                val_acc_for_best = train_acc

            if self.scheduler is not None:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if has_val else train_loss)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(lr)

            metrics = {'epoch': epoch + 1, 'train_loss': train_loss,
                       'train_acc': train_acc, 'val_loss': val_loss,
                       'val_acc': val_acc, 'lr': lr}
            self.on_log(
                f'Epoch {epoch + 1}/{epochs}  '
                f'训练 loss={train_loss:.4f} acc={train_acc:.2f}%  '
                f'验证 loss={val_loss:.4f} acc={val_acc:.2f}%  lr={lr:.2e}')

            select_acc = val_acc if has_val else train_acc
            if select_acc > best_val_acc:
                best_val_acc = select_acc
                best_metrics = dict(metrics)
                if save_dir:
                    save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth'),
                        self.model, model_type, model_config,
                        classes=classes, preprocess=preprocess,
                        metrics={'best_val_acc': best_val_acc, 'epoch': epoch + 1},
                        history=history, epoch=epoch + 1)

            if on_epoch_end:
                on_epoch_end(epoch, metrics, copy.deepcopy(history))

            if should_stop and should_stop():
                stopped = True
                self.on_log(f'收到停止请求，在第 {epoch + 1} 轮结束训练')
                break

        if save_dir:
            save_checkpoint(
                os.path.join(save_dir, 'final_model.pth'),
                self.model, model_type, model_config,
                classes=classes, preprocess=preprocess,
                metrics={'best_val_acc': best_val_acc},
                history=history, epoch=len(history['train_loss']))

        return history, best_metrics, stopped


# ------------------------------------------------------------ 便捷构造函数
def make_criterion(class_weights=None, device='cpu', label_smoothing=0.1):
    """
    构造交叉熵损失。label_smoothing 防止对"参数指纹"过度自信，
    改善跨参数泛化时的置信度校准（0 = 关闭）。
    """
    weight = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def make_optimizer(model, lr=1e-4, weight_decay=1e-4, only_trainable=True):
    params = (p for p in model.parameters()
              if (p.requires_grad or not only_trainable))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, kind='plateau', epochs=50):
    if kind == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=epochs)
    return None


def load_pretrained_compatible(model, checkpoint_path, device):
    """
    加载同构预训练权重（微调用）。只接受 shape 与名称都匹配的参数，
    返回加载成功的参数比例；完全不兼容时返回 0 并给出警告。

    注意：旧版流程把 baseline DenseNet 的权重"加载"进 EnhancedDenseNet，
    两者参数名几乎不重叠，实际等于带着 70% 随机冻结权重训练 —— 已废弃该用法。
    """
    info = load_checkpoint(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    compatible = {k: v for k, v in info['state_dict'].items()
                  if k in model_dict and model_dict[k].shape == v.shape}
    ratio = len(compatible) / max(len(model_dict), 1)
    if compatible:
        model_dict.update(compatible)
        model.load_state_dict(model_dict)
    if ratio < 0.5:
        logger.warning('预训练权重与当前模型兼容度仅 %.0f%%，'
                       '微调应使用同构模型的 checkpoint', ratio * 100)
    return ratio
