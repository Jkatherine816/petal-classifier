"""
core/dataset.py - 数据集与数据加载

相对旧版 utils/dataset.py 的修复：
    - 删除无法运行的 _collect_from_six_in_one（类名/变量未定义）——
      训练数据统一走"拆分落盘 -> 按类目录加载"单一路径；
    - 图像不在 __init__ 全部读入内存，改为惰性加载（存路径，__getitem__ 再读）；
    - 提供 labels 属性，类别权重/分层划分都能正确使用（旧版 Subset 取不到 labels）；
    - 数据增强作用于 PIL 图像，ToTensor/Normalize 永远在最后（旧版顺序颠倒）；
    - 训练/验证划分改为分层抽样（stratified），且训练/验证使用不同 transform。
"""

import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T

from .constants import (
    CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, CLASS_DIR_PREFIX,
    DEFAULT_INPUT_SIZE, NORM_MEAN, NORM_STD,
)

logger = logging.getLogger(__name__)

IMG_EXTS = ('.png', '.jpg', '.jpeg')


class PetalDataset(Dataset):
    """单花瓣图像数据集（按 class_<类别>/ 目录组织，惰性加载）"""

    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.classes = list(CLASSES)
        self.class_to_idx = dict(CLASS_TO_IDX)
        self.idx_to_class = dict(IDX_TO_CLASS)

        self.samples = []   # [(path, label_idx)]
        self._scan()

        logger.info('数据集加载: %d 个样本, 分布=%s',
                    len(self.samples), self.class_distribution())

    def _scan(self):
        for cls in self.classes:
            class_dir = os.path.join(self.data_root, f'{CLASS_DIR_PREFIX}{cls}')
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMG_EXTS):
                    self.samples.append((os.path.join(class_dir, fname),
                                         self.class_to_idx[cls]))

    @property
    def labels(self):
        """全部样本的标签索引（供类别权重/分层划分使用）"""
        return [label for _, label in self.samples]

    def class_distribution(self):
        dist = {}
        for _, label in self.samples:
            name = self.idx_to_class[label]
            dist[name] = dist.get(name, 0) + 1
        return dist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_path(self, idx):
        return self.samples[idx][0]


# ---------------------------------------------------------------- 变换
def build_train_transform(input_size=DEFAULT_INPUT_SIZE, aug_level='medium',
                          mean=NORM_MEAN, std=NORM_STD):
    """
    训练变换：先 PIL 增强，再 ToTensor + Normalize，最后张量级增强。

    注意：花瓣是稀疏点阵图（不是密集自然图像），增强必须克制——
    过强的 RandomErasing 会把有效点整块抹掉，过强的 RandomResizedCrop
    会切掉花瓣尖端的圆心锚点，二者都会破坏标签信息导致欠拟合。
    点密度漂移已由渲染端域随机化覆盖，此处 RandomErasing 仅保留小比例。
    """
    aug_level = (aug_level or 'none').lower()
    pil_ops = []
    tensor_ops = []
    if aug_level == 'light':
        pil_ops = [
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        tensor_ops = []
    elif aug_level == 'medium':
        pil_ops = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
        ]
        tensor_ops = [T.RandomErasing(p=0.1, scale=(0.02, 0.06))]
    elif aug_level == 'heavy':
        pil_ops = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
        ]
        tensor_ops = [T.RandomErasing(p=0.2, scale=(0.02, 0.12))]
    return T.Compose(pil_ops + [
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ] + tensor_ops)


def build_eval_transform(input_size=DEFAULT_INPUT_SIZE,
                         mean=NORM_MEAN, std=NORM_STD):
    """评估/预测变换：不做任何随机增强。"""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def compute_class_weights(labels, num_classes=len(CLASSES)):
    """由标签列表计算类别权重（逆频率，归一化使均值为 1）。"""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0          # 防止除零
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.FloatTensor(weights), counts.astype(int)


# ---------------------------------------------------------------- DataLoader
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
    logger.info('训练集 %d / 验证集 %d，类别计数 %s',
                len(train_idx), len(val_idx), class_counts.tolist())
    return train_loader, val_loader, info


def create_predict_loader(data_root, batch_size=32,
                          input_size=DEFAULT_INPUT_SIZE,
                          mean=NORM_MEAN, std=NORM_STD, num_workers=2):
    """预测/评估用：全量数据集 + 评估变换。"""
    ds = PetalDataset(data_root, transform=build_eval_transform(
        input_size, mean, std))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    return loader, ds
