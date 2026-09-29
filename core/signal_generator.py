"""
core/signal_generator.py - 时序信号生成器

生成 7 类时序信号（normal + 6 种异常），保存为 .npy 并按 class_<类别> 目录组织。
相对旧版 data_generator.py 的修复：
    - 不再每次调用 np.random.seed()（原实现会重置全局种子、无法复现），
      改用 np.random.Generator，可传入固定 seed 完全复现。
    - 支持进度回调（GUI 用）。
"""

import os
import logging

import numpy as np

from .constants import (
    CLASSES, CLASS_DIR_PREFIX,
    DEFAULT_SIGNAL_LENGTH, DEFAULT_SAMPLING_RATE,
)

logger = logging.getLogger(__name__)


class SignalGenerator:
    """时序信号生成器（可复现）"""

    def __init__(self, length=DEFAULT_SIGNAL_LENGTH,
                 sampling_rate=DEFAULT_SAMPLING_RATE, seed=None):
        """
        参数：
            length: 信号长度 L
            sampling_rate: 采样频率 Fs
            seed: 随机种子（None 则不可复现）
        """
        self.L = int(length)
        self.Fs = sampling_rate
        self.t = np.arange(0, self.L) / self.Fs
        self.rng = np.random.default_rng(seed)
        self.classes = list(CLASSES)

    # ------------------------------------------------------------ 各类信号
    def gen_normal(self):
        """正常信号：基带多频正弦叠加 + 多载波调制 + 噪声"""
        rng = self.rng
        n_carriers = rng.integers(3, 7)
        k_base = rng.integers(6, 11)
        baseband = np.zeros(self.L)

        for _ in range(k_base):
            f_j = rng.random() * 0.2 + 0.1
            a_j = rng.random() * 0.9 + 0.1
            phi_j = rng.random() * 2 * np.pi - np.pi
            baseband += a_j * np.sin(2 * np.pi * f_j * self.t + phi_j)

        signal = np.zeros(self.L)
        for _ in range(n_carriers):
            f_i = rng.random() * 3
            phi_i = rng.random() * 2 * np.pi - np.pi
            signal += baseband * np.sin(2 * np.pi * f_i * self.t + phi_i)

        # 按随机信噪比加噪
        sp = np.var(signal)
        snr = rng.random() * 0.9 + 0.1
        noise_power = sp / (10 ** (snr / 10))
        noise = np.sqrt(noise_power) * rng.standard_normal(self.L)
        return signal + noise

    def gen_outlier(self, normal_signal):
        """离群点：随机位置叠加 ±(8~20)σ 的尖峰"""
        rng = self.rng
        sigma = np.std(normal_signal)
        num_outliers = int(round(rng.random() * 0.001 * self.L + 50))
        indices = rng.choice(self.L, num_outliers, replace=False)
        magnitude = rng.random() * 12 + 8

        mode = rng.random()
        if mode <= 0.3:
            signs = np.sign(rng.standard_normal(num_outliers))   # 随机方向
        elif mode <= 0.65:
            signs = np.ones(num_outliers)                        # 全部偏大
        else:
            signs = -np.ones(num_outliers)                       # 全部偏小

        out = normal_signal.copy()
        out[indices] += signs * magnitude * sigma
        return out

    def gen_constant(self):
        """常数信号：几个常数水平循环平铺 + 少量跳变点"""
        rng = self.rng
        n = rng.integers(4, 7)
        constants = rng.random(n) * rng.random()
        out = np.tile(constants, int(np.ceil(self.L / n)))[:self.L].copy()

        jump_prob = rng.random() * 0.1 + 0.2
        num_jumps = max(1, int(round(jump_prob * self.L)))
        jump_indices = rng.choice(self.L, num_jumps, replace=False)
        out[jump_indices] *= (1 + rng.standard_normal(num_jumps) * 0.1)
        return out

    def gen_trend(self, normal_signal):
        """趋势信号：正常信号叠加线性漂移"""
        rng = self.rng
        t_trend = np.linspace(0, 3600, len(normal_signal))
        signal_range = np.max(normal_signal) - np.min(normal_signal)
        limit = (0.1 * signal_range) / 3600

        if rng.random() < 0.5:
            k = -2 + (2 - limit) * rng.random()
        else:
            k = limit + (2 - limit) * rng.random()
        return normal_signal + k * t_trend

    def gen_missing(self, normal_signal):
        """缺失信号：连续一段置零（30%~70% 长度）"""
        rng = self.rng
        fraction = rng.random() * 0.4 + 0.3
        length = int(round(fraction * self.L))
        start = int(rng.integers(0, self.L - length + 1))
        out = normal_signal.copy()
        out[start:start + length] = 0
        return out

    def gen_bias(self, normal_signal):
        """偏置信号：连续一段整体抬升 (4~5)σ"""
        rng = self.rng
        fraction = rng.random() * 0.4 + 0.3
        length = int(round(fraction * self.L))
        start = int(rng.integers(0, self.L - length + 1))
        magnitude = rng.integers(4, 6)
        sigma = np.std(normal_signal)
        out = normal_signal.copy()
        out[start:start + length] += magnitude * sigma
        return out

    def gen_staircase(self, normal_signal):
        """阶梯信号：分段量化到若干电平"""
        rng = self.rng
        num_segments = int(rng.integers(2, 4))
        split_points = np.sort(
            rng.choice(round(self.L * 0.8), num_segments - 1, replace=False)
            + round(self.L * 0.1)
        )
        segments = np.concatenate(([0], split_points, [self.L]))
        levels_per_segment = rng.integers(15, 26, num_segments)
        threshold_ratio = 0.10

        out = normal_signal.copy()
        for seg in range(num_segments):
            s, e = int(segments[seg]), int(segments[seg + 1])
            seg_data = out[s:e]
            seg_min, seg_max = np.min(seg_data), np.max(seg_data)
            seg_range = seg_max - seg_min
            if seg_range <= 0:
                continue
            levels = np.linspace(seg_min + seg_range * 0.1,
                                 seg_max - seg_range * 0.1,
                                 levels_per_segment[seg])
            # 向最近电平吸附（仅当距离足够近）
            idx = np.argmin(np.abs(seg_data[:, None] - levels[None, :]), axis=1)
            nearest = levels[idx]
            close = np.abs(seg_data - nearest) < seg_range * threshold_ratio
            seg_data[close] = nearest[close]
            out[s:e] = seg_data
        return out

    # ------------------------------------------------------------ 批量生成
    def generate_one_set(self):
        """以一条正常信号为基础，生成全部 7 类信号，返回 {类别: ndarray}"""
        normal = self.gen_normal()
        return {
            'normal': normal,
            'outlier': self.gen_outlier(normal),
            'constant': self.gen_constant(),
            'trend': self.gen_trend(normal),
            'missing': self.gen_missing(normal),
            'bias': self.gen_bias(normal),
            'staircase': self.gen_staircase(normal),
        }

    def generate_dataset(self, output_dir, num_samples_per_class=100,
                         progress_cb=None):
        """
        批量生成数据集。

        参数：
            output_dir: 输出目录（自动创建 class_<类别> 子目录）
            num_samples_per_class: 每类样本数
            progress_cb: 可选回调 progress_cb(percent:int, message:str)
        返回：
            stats: {类别: 样本数}
        """
        os.makedirs(output_dir, exist_ok=True)
        for cls in self.classes:
            os.makedirs(os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{cls}'),
                        exist_ok=True)

        logger.info('开始生成数据：每类 %d 条，长度 %d，采样率 %s',
                    num_samples_per_class, self.L, self.Fs)

        for i in range(num_samples_per_class):
            signals = self.generate_one_set()
            for cls, sig in signals.items():
                path = os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{cls}',
                                    f'{cls}_{i:03d}.npy')
                np.save(path, sig)

            if progress_cb:
                progress_cb(int((i + 1) / num_samples_per_class * 100),
                            f'已生成 {i + 1}/{num_samples_per_class} 组')
            elif (i + 1) % 10 == 0:
                logger.info('生成进度: %d/%d', i + 1, num_samples_per_class)

        stats = {}
        for cls in self.classes:
            d = os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{cls}')
            stats[cls] = len([f for f in os.listdir(d) if f.endswith('.npy')])

        logger.info('数据生成完成: %s', stats)
        return stats


def load_npy(filepath):
    """加载 .npy 并展平为一维；失败返回 None。"""
    try:
        data = np.load(filepath)
        if data.ndim > 1:
            data = data.flatten()
        return data if len(data) > 0 else None
    except Exception as exc:
        logger.warning('加载 npy 失败 %s: %s', filepath, exc)
        return None


def load_dataset_index(data_root):
    """扫描数据目录，返回 {类别: [文件路径, ...]}（按文件名排序）。"""
    index = {}
    for cls in CLASSES:
        class_dir = os.path.join(data_root, f'{CLASS_DIR_PREFIX}{cls}')
        if os.path.isdir(class_dir):
            files = sorted(os.path.join(class_dir, f)
                           for f in os.listdir(class_dir) if f.endswith('.npy'))
            index[cls] = files
    return index
