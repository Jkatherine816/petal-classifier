"""
core/sdp_renderer.py - SDP 六合一图像渲染器

相对旧版 image_generator(_fixed).py 的修复：
    - 渲染改为纯数据驱动：删除按类别施加的"特征增强"（原实现给 outlier 加噪、
      constant 压扁、trend 加坡度、missing 稀疏采样），这些增强在模式B预测时
      不存在，造成训练/预测分布不一致。
    - 训练/预测统一单色渲染（修复颜色捷径）；彩色仅用于人工预览。
    - 模式B 超过 6 个文件不再静默丢弃，自动分组成多张六合一图。
    - 元数据 petals 数组始终写满 6 项，字段齐全。
    - 角度/颜色/类别定义全部来自 core.constants。
    - 新增快速路径 render_single_petals：信号直接渲染为单花瓣训练图，
      与拆分路径几何一致（圆心锚点 + 居中裁剪）。
"""

import os
import json
import logging

import numpy as np
from core import mpl_setup as _mpl_setup  # noqa: F401  配置 matplotlib 中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .constants import (
    CLASSES, CLASS_TO_IDX, SECTOR_ANGLES, SECTOR_CENTERS, POSITION_TO_TYPE,
    CLASS_COLORS, MONO_COLOR, DEFAULT_IMAGE_SIZE, DEFAULT_SEGMENT_LENGTH,
    DEFAULT_TAU, DEFAULT_DPI, METADATA_SUFFIX)
from .signal_generator import load_npy, load_dataset_index
from .image_utils import crop_and_square

logger = logging.getLogger(__name__)


class SDPRenderer:
    """SDP 极坐标渲染器"""

    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, dpi=DEFAULT_DPI,
                 segment_length=DEFAULT_SEGMENT_LENGTH, tau=DEFAULT_TAU,
                 seed=None):
        self.image_size = image_size
        self.dpi = dpi
        self.segment_length = segment_length
        self.tau = tau
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ SDP 几何
    def preprocess(self, data):
        """取前 segment_length 点并归一化到 [0,1]（常数序列保护）"""
        data = np.asarray(data, dtype=np.float64)[:self.segment_length]
        lo, hi = data.min(), data.max()
        if hi - lo < 1e-12:
            return np.full_like(data, 0.5)
        return (data - lo) / (hi - lo)

    def petal_points(self, data_norm):
        """
        SDP 映射：第 i 点 -> 角度 theta_i = i/N*2π + 抖动，
        半径 r_i = data_norm[i]（对称：r 与 1-r 两点）
        纯几何函数，jitter 来自 self.rng 保证可复现。
        """
        n = len(data_norm)
        base = np.arange(n) / n * 2 * np.pi
        jitter = self.rng.uniform(-0.002, 0.002, n)
        theta = base + jitter
        r = data_norm
        # 对称点模式：每个样本两个点
        thetas = np.concatenate([theta, theta])
        rs = np.concatenate([r, 1.0 - r])
        return thetas, rs

    def signal_to_petal(self, data):
        return self.petal_points(self.preprocess(data))

    # ------------------------------------------------------------ 渲染
    def _render_polar(self, point_sets, output_path):
        """
        渲染 6 瓣到一张图。point_sets: list[6] of (theta, r, color)
        theta=0 正东、逆时针（matplotlib polar 默认）。
        """
        px = self.image_size
        fig = plt.figure(figsize=(px / self.dpi, px / self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)          # 逆时针
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        for idx, (lo, hi) in enumerate(SECTOR_ANGLES):
            if idx >= len(point_sets) or point_sets[idx] is None:
                continue
            theta, r, color = point_sets[idx]
            ax.scatter(theta, r, s=0.3, c=color, marker='.', linewidths=0)
        ax.plot(0, 0, 'ko', markersize=2)   # 圆心锚点（拆分器定位用）
        ax.set_ylim(0, 1.05)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(output_path, dpi=self.dpi)
        plt.close(fig)

    def _write_metadata(self, path, petals, mode, extra=None):
        """petals 永远写满 6 项（缺位补 None），字段齐全"""
        padded = list(petals)[:6] + [None] * max(0, 6 - len(petals))
        meta = {
            'mode': mode,
            'image_size': self.image_size,
            'segment_length': self.segment_length,
            'tau': self.tau,
            'sector_angles': [list(x) for x in SECTOR_ANGLES],
            'petals': padded,
        }
        if extra:
            meta.update(extra)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------ 模式A：训练图
    def generate_mode_A(self, data_root, output_dir, num_images,
                        colorful=False, progress_cb=None):
        """
        生成训练图：一半 train_fixed_*（位置 i 放第 i 种异常），
        一半 train_normal_*（6 瓣全正常）。返回生成文件列表。
        """
        os.makedirs(output_dir, exist_ok=True)
        index = load_dataset_index(data_root)
        made = []
        n_fixed = num_images // 2
        n_normal = num_images - n_fixed
        total = n_fixed + n_normal

        for k in range(n_fixed):
            point_sets = []
            petals_meta = []
            for pos in range(6):
                cls = POSITION_TO_TYPE[pos]
                data = index.random_choice(cls)
                color = CLASS_COLORS[cls] if colorful else MONO_COLOR
                t, r = self.signal_to_petal(data)
                point_sets.append((t, r, color))
                petals_meta.append({
                    'position': pos, 'source_class': cls, 'true_label': cls,
                    'source_file': index.last_choice_path,
                })
            name = f'train_fixed_{k:05d}'
            png = os.path.join(output_dir, name + '.png')
            self._render_polar(point_sets, png)
            self._write_metadata(
                os.path.join(output_dir, name + METADATA_SUFFIX),
                petals_meta, 'A_fixed')
            made.append(png)
            if progress_cb:
                progress_cb(int((k + 1) / total * 100),
                            f'固定异常图 {k + 1}/{n_fixed}')

        for k in range(n_normal):
            point_sets = []
            petals_meta = []
            for pos in range(6):
                data = index.random_choice('normal')
                color = CLASS_COLORS['normal'] if colorful else MONO_COLOR
                t, r = self.signal_to_petal(data)
                point_sets.append((t, r, color))
                petals_meta.append({
                    'position': pos, 'source_class': 'normal',
                    'true_label': 'normal',
                    'source_file': index.last_choice_path,
                })
            name = f'train_normal_{k:05d}'
            png = os.path.join(output_dir, name + '.png')
            self._render_polar(point_sets, png)
            self._write_metadata(
                os.path.join(output_dir, name + METADATA_SUFFIX),
                petals_meta, 'A_normal')
            made.append(png)
            if progress_cb:
                progress_cb(int((n_fixed + k + 1) / total * 100),
                            f'全正常图 {k + 1}/{n_normal}')
        return made

    # ------------------------------------------------------------ 模式B：预测图
    def generate_mode_B(self, data_folder, output_dir, colorful=False,
                        progress_cb=None):
        """
        把文件夹里的 .npy 按每 6 个一组渲染成 predict_XXX.png。
        返回 [(png_path, metadata), ...]。超过 6 个自动分组，不再丢弃。
        """
        os.makedirs(output_dir, exist_ok=True)
        files = sorted(fn for fn in os.listdir(data_folder)
                       if fn.endswith('.npy'))
        results = []
        if not files:
            return results
        groups = [files[i:i + 6] for i in range(0, len(files), 6)]
        for gi, group in enumerate(groups):
            point_sets = [None] * 6
            petals_meta = []
            for pos, fn in enumerate(group):
                data = load_npy(os.path.join(data_folder, fn))
                t, r = self.signal_to_petal(data)
                point_sets[pos] = (t, r, MONO_COLOR)
                petals_meta.append({
                    'position': pos, 'source_file': fn, 'true_label': None,
                })
            name = f'predict_{gi:03d}'
            png = os.path.join(output_dir, name + '.png')
            self._render_polar(point_sets, png)
            meta_path = os.path.join(output_dir, name + METADATA_SUFFIX)
            self._write_metadata(meta_path, petals_meta, 'B')
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            results.append((png, meta))
            if progress_cb:
                progress_cb(int((gi + 1) / len(groups) * 100),
                            f'预测图 {gi + 1}/{len(groups)}')
        return results

    # ------------------------------------------------------------ 快速路径
    def render_single_petal(self, data, output_path, canvas_size=224):
        """
        信号 -> 单花瓣训练图（快速路径）。
        渲染到与拆分器一致的标准扇区 (60,120)，带圆心锚点，再居中裁剪，
        保证与 split 路径的几何完全一致（方向/位置/尺度）。
        """
        data_norm = self.preprocess(data)
        n = len(data_norm)
        lo, hi = np.radians(60), np.radians(120)
        theta = lo + np.arange(n) / n * (hi - lo)
        jitter = self.rng.uniform(-0.002, 0.002, n)
        theta = theta + jitter
        r = data_norm

        px = canvas_size
        fig = plt.figure(figsize=(px / self.dpi, px / self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.scatter(np.concatenate([theta, theta]),
                   np.concatenate([r, 1.0 - r]),
                   s=0.3, c=MONO_COLOR, marker='.', linewidths=0)
        ax.plot(0, 0, 'ko', markersize=2)      # 圆心锚点：保证裁剪居中
        ax.set_ylim(0, 1.05)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        petal = crop_and_square(buf, DEFAULT_INPUT_SIZE_FINAL)
        petal.save(output_path)
        return output_path

    def render_single_petals(self, data_root, output_dir, progress_cb=None):
        """批量快速路径：数据集目录 -> class_ 目录单花瓣图。返回统计。"""
        index = load_dataset_index(data_root)
        stats = {'total': 0, 'per_class': {c: 0 for c in CLASSES}}
        items = index.all_items()
        for i, (cls, path) in enumerate(items):
            data = load_npy(path)
            cls_dir = os.path.join(output_dir, f'class_{cls}')
            os.makedirs(cls_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(cls_dir, f'{stem}_{cls}.png')
            self.render_single_petal(data, out)
            stats['total'] += 1
            stats['per_class'][cls] += 1
            if progress_cb:
                progress_cb(int((i + 1) / len(items) * 100),
                            f'{i + 1}/{len(items)} {cls}')
        logger.info('快速路径渲染完成: %s', stats)
        return stats


# 快速路径最终输出尺寸（与拆分器 output_size 一致）
DEFAULT_INPUT_SIZE_FINAL = 64
