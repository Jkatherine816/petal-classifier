"""
core/dataset.py - 数据集与 DataLoader

目录约定：data_root/class_<类别>/*.png
修复：旧版 Subset.labels 从未生效（Subset 无 labels 属性），类权重恒为 None；
本版 PetalDataset 提供真正的 labels 属性，compute_class_weights 正常工作。
"""

import os
import logging

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from .constants import (CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
                        CLASS_DIR_PREFIX, DEFAULT_INPUT_SIZE,
                        NORM_MEAN, NORM_STD)

logger = logging.getLogger(__name__)

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


class PetalDataset(Dataset):
    """单花瓣图像数据集（懒加载，支持任意图片总量）"""

    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []      # [(path, label_idx)]
        self._scan()

    def _scan(self):
        for cls_name in CLASSES:
            cls_dir = os.path.join(self.data_root,
                                   f'{CLASS_DIR_PREFIX}{cls_name}')
            if not os.path.isdir(cls_dir):
                continue
            label = CLASS_TO_IDX[cls_name]
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith(IMG_EXTS):
                    self.samples.append((os.path.join(cls_dir, fn), label))
        dist = self.class_distribution()
        logger.info('数据集加载: %d 个样本, 分布=%s', len(self.samples), dist)

    @property
    def labels(self):
        """修复点：旧版 Subset.labels 从未生效，这里提供真实标签数组"""
        return [s[1] for s in self.samples]

    def class_distribution(self):
        counts = np.bincount([s[1] for s in self.samples],
                             minlength=len(CLASSES))
        return {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))
                if counts[i] > 0}

    def get_path(self, idx):
        return self.samples[idx][0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_train_transform(input_size=DEFAULT_INPUT_SIZE, aug_level='medium',
                          mean=NORM_MEAN, std=NORM_STD):
    """训练用数据增强。注意：ToTensor/Normalize 必须在最后（PIL 增强先于张量化）。"""
    ops = [transforms.Resize((input_size, input_size))]
    if aug_level in ('light', 'medium', 'heavy'):
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomVerticalFlip())
    if aug_level in ('medium', 'heavy'):
        ops.append(transforms.RandomRotation(15))
        ops.append(transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)))
    if aug_level == 'heavy':
        ops.append(transforms.RandomRotation(15))
        ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def build_eval_transform(input_size=DEFAULT_INPUT_SIZE,
                         mean=NORM_MEAN, std=NORM_STD):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def compute_class_weights(labels):
    """逆频率归一化类权重（mean=1），用于 CrossEntropyLoss"""
    counts = np.bincount(np.asarray(labels), minlength=len(CLASSES)) \
        .astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)


def create_dataloaders(data_root, batch_size=32, val_split=0.2,
                       aug_level='medium', input_size=DEFAULT_INPUT_SIZE,
                       mean=NORM_MEAN, std=NORM_STD, seed=42,
                       num_workers=2, use_class_weights=True):
    """
    创建训练/验证 DataLoader（分层划分）。

    返回：
        train_loader, val_loader, info
        info = {num_classes, classes, class_to_idx, idx_to_class,
                class_weights(tensor), class_counts, total_samples,
                train_size, val_size, preprocess:{input_size, mean, std}}
    """
    train_tf = build_train_transform(input_size, aug_level, mean, std)
    eval_tf = build_eval_transform(input_size, mean, std)

    ds_train = PetalDataset(data_root, transform=train_tf)
    ds_eval = PetalDataset(data_root, transform=eval_tf)
    assert len(ds_train) == len(ds_eval)

    labels = np.array(ds_train.labels)
    n = len(labels)
    if n == 0:
        raise ValueError(f'数据集为空: {data_root}（需要 class_<类别>/ 目录结构）')

    # 分层划分
    from sklearn.model_selection import StratifiedShuffleSplit
    if val_split > 0 and n >= 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split,
                                     random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(n), labels))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        train_idx, val_idx = perm, np.array([], dtype=int)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(ds_train, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generator,
                              pin_memory=True)
    val_loader = DataLoader(Subset(ds_eval, val_idx),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    class_weights, class_counts = compute_class_weights(labels[train_idx])

    info = {
        'num_classes': len(CLASSES),
        'classes': list(CLASSES),
        'class_to_idx': dict(CLASS_TO_IDX),
        'idx_to_class': dict(IDX_TO_CLASS),
        'class_weights': class_weights if use_class_weights else None,
        'class_counts': class_counts.tolist(),
        'total_samples': n,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'preprocess': {'input_size': input_size, 'mean': list(mean),
                       'std': list(std)},
    }
    return train_loader, val_loader, info


def create_predict_loader(data_root, input_size=DEFAULT_INPUT_SIZE,
                          mean=NORM_MEAN, std=NORM_STD,
                          batch_size=64, num_workers=0):
    """评估/预测用 loader（无增强）。返回 (loader, dataset)。"""
    ds = PetalDataset(data_root,
                      transform=build_eval_transform(input_size, mean, std))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    return loader, ds
