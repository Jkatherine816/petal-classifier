"""
core/signal_generator.py - 7 类时序信号生成器

修复与改进：
- 使用 np.random.default_rng（可复现，seed 可控）
- staircase 量化改为向量化（旧版 Python 循环极慢）
- 输出目录按 class_<类别>/ 组织，配套索引加载器
"""

import os
import json
import logging

import numpy as np

from .constants import CLASSES, CLASS_DIR_PREFIX

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self, length=50000, sampling_rate=20, seed=None):
        self.length = int(length)
        self.fs = sampling_rate
        self.rng = np.random.default_rng(seed)
        self.t = np.arange(self.length) / self.fs

    # ------------------------------------------------------------ 基线
    def _base_wave(self):
        """多谐波叠加 + 轻噪声的类振动基线"""
        sig = np.zeros(self.length)
        for _ in range(self.rng.integers(2, 5)):
            f = self.rng.uniform(0.5, self.fs / 4)
            amp = self.rng.uniform(0.5, 1.5)
            phase = self.rng.uniform(0, 2 * np.pi)
            sig += amp * np.sin(2 * np.pi * f * self.t + phase)
        sig += self.rng.normal(0, 0.05, self.length)
        return sig

    # ------------------------------------------------------------ 7 类
    def gen_normal(self):
        return self._base_wave()

    def gen_outlier(self):
        sig = self._base_wave()
        n_out = self.rng.integers(3, 12)
        idx = self.rng.choice(self.length, n_out, replace=False)
        sig[idx] += self.rng.uniform(3, 8, n_out) * \
            self.rng.choice([-1, 1], n_out)
        return sig

    def gen_constant(self):
        sig = self._base_wave()
        start = self.rng.integers(self.length // 4, self.length // 2)
        span = self.rng.integers(self.length // 8, self.length // 3)
        level = float(sig[start])
        sig[start:start + span] = level
        return sig

    def gen_trend(self):
        sig = self._base_wave()
        slope = self.rng.uniform(-4, 4)
        if abs(slope) < 0.5:
            slope = 0.5 * np.sign(slope if slope != 0 else 1)
        sig += slope * np.linspace(0, 1, self.length)
        return sig

    def gen_missing(self):
        sig = self._base_wave()
        n_gaps = self.rng.integers(2, 6)
        for _ in range(n_gaps):
            start = self.rng.integers(0, self.length - 200)
            span = self.rng.integers(50, 200)
            sig[start:start + span] = 0.0
        return sig

    def gen_bias(self):
        sig = self._base_wave()
        sig += self.rng.uniform(1.5, 4.0) * self.rng.choice([-1, 1])
        return sig

    def gen_staircase(self):
        """阶梯：向量化量化（修复旧版逐点循环的性能问题）"""
        sig = self._base_wave()
        n_levels = self.rng.integers(4, 9)
        lo, hi = sig.min(), sig.max()
        if hi - lo < 1e-12:
            return sig
        levels = np.linspace(lo, hi, n_levels)
        idx = np.searchsorted(levels, sig)
        idx = np.clip(idx - 1, 0, n_levels - 1)
        return levels[idx]

    # ------------------------------------------------------------ 数据集
    GENERATORS = {
        'normal': gen_normal, 'outlier': gen_outlier,
        'constant': gen_constant, 'trend': gen_trend,
        'missing': gen_missing, 'bias': gen_bias,
        'staircase': gen_staircase,
    }

    def generate_one(self, cls_name):
        return self.GENERATORS[cls_name](self)

    def generate_dataset(self, output_dir, num_samples_per_class,
                         progress_cb=None):
        """生成完整数据集到 output_dir/class_<类别>/，返回统计 dict"""
        stats = {'per_class': {}, 'total': 0,
                 'length': self.length, 'sampling_rate': self.fs}
        total = len(CLASSES) * num_samples_per_class
        done = 0
        for cls in CLASSES:
            cls_dir = os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{cls}')
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(num_samples_per_class):
                sig = self.generate_one(cls)
                np.save(os.path.join(cls_dir, f'{cls}_{i:05d}.npy'), sig)
                done += 1
                if progress_cb:
                    progress_cb(int(done / total * 100),
                                f'{cls} {i + 1}/{num_samples_per_class}')
            stats['per_class'][cls] = num_samples_per_class
            stats['total'] += num_samples_per_class
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info('数据集生成完成: %s', stats)
        return stats


# ---------------------------------------------------------------- 加载工具
def load_npy(path):
    return np.load(path)


class DatasetIndex:
    """class_<类别>/ 目录索引，支持随机抽样（渲染训练图用）"""

    def __init__(self, data_root, seed=None):
        self.data_root = data_root
        self.rng = np.random.default_rng(seed)
        self.files = {}
        self.last_choice_path = None
        for cls in CLASSES:
            cls_dir = os.path.join(data_root, f'{CLASS_DIR_PREFIX}{cls}')
            if os.path.isdir(cls_dir):
                self.files[cls] = sorted(
                    os.path.join(cls_dir, fn)
                    for fn in os.listdir(cls_dir) if fn.endswith('.npy'))
            else:
                self.files[cls] = []

    def random_choice(self, cls):
        pool = self.files.get(cls) or []
        if not pool:
            raise ValueError(f'类别 {cls} 没有可用样本: {self.data_root}')
        self.last_choice_path = pool[self.rng.integers(len(pool))]
        return load_npy(self.last_choice_path)

    def all_items(self):
        items = []
        for cls in CLASSES:
            for path in self.files.get(cls, []):
                items.append((cls, path))
        return items


def load_dataset_index(data_root, seed=None):
    return DatasetIndex(data_root, seed=seed)
