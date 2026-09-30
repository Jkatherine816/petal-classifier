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
      跳过"六合一 -> 拆分"往返（可选，详见 GUI 拆分页开关）。
    - 域随机化（randomize=True，训练推荐）：逐花瓣随机采样
      segment_length / tau / jitter，防止模型记忆"参数指纹"（点密度/纹理），
      被迫学习对参数不变的形状特征。采样值写入元数据 render_params 可追溯。
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
    CLASSES, CLASS_DIR_PREFIX, METADATA_SUFFIX,
    SECTOR_ANGLES, POSITION_TO_TYPE, CLASS_COLORS, MONO_COLOR,
    DEFAULT_IMAGE_SIZE, DEFAULT_DPI, DEFAULT_SEGMENT_LENGTH, DEFAULT_TAU,
    DEFAULT_PETAL_SIZE, DEFAULT_JITTER,
    RAND_SEGMENT_RANGE, RAND_TAU_RANGE, RAND_JITTER_RANGE,
)
from .signal_generator import load_npy, load_dataset_index
from .image_utils import crop_and_square

logger = logging.getLogger(__name__)


class SDPRenderer:
    """SDP 极坐标图像渲染器"""

    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, dpi=DEFAULT_DPI,
                 segment_length=DEFAULT_SEGMENT_LENGTH, tau=DEFAULT_TAU,
                 jitter=DEFAULT_JITTER, seed=None, randomize=False):
        self.image_size = image_size
        self.dpi = dpi
        self.segment_length = segment_length
        self.tau = tau
        self.jitter = jitter
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ 域随机化
    def _sample_params(self):
        """
        采样一组渲染参数。randomize=True 时在配置范围内随机（训练用）；
        否则返回固定实例参数（预测/评估用，保证与 checkpoint 同分布）。
        """
        if not self.randomize:
            return {'segment_length': self.segment_length,
                    'tau': self.tau, 'jitter': self.jitter}
        return {
            'segment_length': int(self.rng.integers(
                RAND_SEGMENT_RANGE[0], RAND_SEGMENT_RANGE[1] + 1)),
            'tau': int(self.rng.integers(
                RAND_TAU_RANGE[0], RAND_TAU_RANGE[1] + 1)),
            'jitter': float(self.rng.uniform(
                RAND_JITTER_RANGE[0], RAND_JITTER_RANGE[1])),
        }

    # ------------------------------------------------------------ 数据 -> 点集
    def preprocess(self, data, segment_length=None, tau=None):
        """
        时序数据 -> (radius, y)。
        截取 segment_length 段，归一化到 [0,1]，按 tau 构造 SDP 延迟对。
        """
        if data is None or len(data) == 0:
            return None, None

        seg = segment_length or self.segment_length
        t = self.tau if tau is None else tau

        segment = data[:seg] if len(data) > seg else data
        d_min, d_max = np.min(segment), np.max(segment)
        norm = (segment - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(segment)

        if len(norm) <= t:
            return None, None
        y = norm[t:]
        radius = norm[:len(y)]
        if len(radius) <= 10:
            return None, None
        return radius, y

    def petal_points(self, radius, y, angle_range, jitter=None):
        """
        (radius, y) -> 扇形内的 (theta_rad, r)。纯几何映射，不做任何类别相关增强。
        角度由 y 的归一化值映射到扇形中部 80% 范围，并叠加抖动（幅度可调）。
        """
        if radius is None or y is None or len(radius) < 10:
            return None, None

        j = self.jitter if jitter is None else jitter

        if np.max(y) > np.min(y):
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        else:
            y_norm = np.zeros_like(y)

        start, end = angle_range
        width = end - start
        theta_deg = start + width * 0.1 + y_norm * width * 0.8
        theta_deg += self.rng.normal(0, j, len(theta_deg))
        theta_deg = np.clip(theta_deg, start, end)
        return np.deg2rad(theta_deg), np.clip(radius, 0, 1)

    def signal_to_petal(self, data, angle_range, params=None):
        """
        单条信号 -> (theta, r)；数据无效时返回 (None, None)。
        params 可指定 {'segment_length','tau','jitter'}，缺省用实例参数。
        """
        p = params or {'segment_length': self.segment_length,
                       'tau': self.tau, 'jitter': self.jitter}
        radius, y = self.preprocess(data, p.get('segment_length'), p.get('tau'))
        return self.petal_points(radius, y, angle_range, p.get('jitter'))

    # ------------------------------------------------------------ 极坐标绘图
    def _render_polar(self, point_sets, output_path):
        """
        绘制极坐标散点图（透明背景、无坐标轴）。
        point_sets: [(theta_rad, r, color_hex)]，None 项自动跳过。
        """
        fig, ax = plt.subplots(
            figsize=(self.image_size / self.dpi, self.image_size / self.dpi),
            dpi=self.dpi, subplot_kw={'projection': 'polar'})
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        for theta, r, color in point_sets:
            if theta is not None and r is not None:
                ax.scatter(theta, r, s=1.8, alpha=0.6,
                           color=color, edgecolors='none', zorder=3)

        ax.set_rmax(1.1)
        ax.set_rticks([])
        ax.set_xticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.plot(0, 0, 'ko', markersize=3, alpha=0.8, zorder=4)

        plt.tight_layout(pad=0)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    pad_inches=0, transparent=True)
        plt.close(fig)
        return True

    @staticmethod
    def _write_metadata(output_path, metadata):
        """写元数据；petals 数组强制补齐到 6 项。"""
        metadata = dict(metadata)
        petals = list(metadata.get('petals', []))
        while len(petals) < 6:
            pos = len(petals)
            petals.append({'position': pos, 'data_file': None,
                           'true_label': None, 'has_data': False,
                           'angle_range': list(SECTOR_ANGLES[pos])})
        metadata['petals'] = petals[:6]
        meta_path = output_path.replace('.png', METADATA_SUFFIX)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return meta_path

    def _color_for(self, label, colorful):
        return CLASS_COLORS.get(label, MONO_COLOR) if colorful else MONO_COLOR

    # ------------------------------------------------------------ 模式A：训练图
    def generate_mode_A(self, data_root, output_dir, num_images=500,
                        colorful=False, progress_cb=None):
        """
        模式A：生成训练图像（固定位置）。
        前半为"六扇形各放一种异常"，后半为"六花瓣全正常"。
        colorful=False（默认，训练用）；True 仅用于人工预览/回归测试。
        randomize=True 时逐花瓣随机采样渲染参数（域随机化，防参数指纹捷径），
        每个花瓣实际使用的参数写入元数据 render_params 字段。
        """
        os.makedirs(output_dir, exist_ok=True)
        data_index = load_dataset_index(data_root)

        n_anomaly = num_images // 2
        n_normal = num_images - n_anomaly
        total = max(num_images, 1)
        done = 0

        def tick(msg):
            nonlocal done
            done += 1
            if progress_cb:
                progress_cb(int(done / total * 100), msg)

        logger.info('模式A：生成 %d 张固定异常图 + %d 张全正常图（%s，域随机化=%s）',
                    n_anomaly, n_normal,
                    '彩色预览' if colorful else '单色', self.randomize)

        for img_idx in range(n_anomaly):
            point_sets, petals_meta = [], []
            for sector_idx in range(6):
                ptype = POSITION_TO_TYPE[sector_idx]
                files = data_index.get(ptype, [])
                theta, r = (None, None)
                data_file = None
                params = self._sample_params()
                if files:
                    data_file = files[int(self.rng.integers(0, len(files)))]
                    theta, r = self.signal_to_petal(load_npy(data_file),
                                                    SECTOR_ANGLES[sector_idx],
                                                    params)
                has_data = theta is not None
                point_sets.append((theta, r, self._color_for(ptype, colorful)))
                petals_meta.append({
                    'position': sector_idx, 'data_file': data_file,
                    'true_label': ptype if has_data else None,
                    'has_data': has_data,
                    'angle_range': list(SECTOR_ANGLES[sector_idx]),
                    'render_params': params,
                })

            out_path = os.path.join(output_dir, f'train_fixed_{img_idx:04d}.png')
            self._render_polar(point_sets, out_path)
            self._write_metadata(out_path, {
                'generation_mode': 'A', 'image_type': 'fixed_anomalies',
                'colorful': colorful, 'randomize': self.randomize,
                'petals': petals_meta})
            tick(f'固定异常图 {img_idx + 1}/{n_anomaly}')

        normal_files = data_index.get('normal', [])
        for img_idx in range(n_normal):
            point_sets, petals_meta = [], []
            for sector_idx in range(6):
                theta, r = (None, None)
                data_file = None
                params = self._sample_params()
                if normal_files:
                    data_file = normal_files[int(self.rng.integers(0, len(normal_files)))]
                    theta, r = self.signal_to_petal(load_npy(data_file),
                                                    SECTOR_ANGLES[sector_idx],
                                                    params)
                has_data = theta is not None
                point_sets.append((theta, r, self._color_for('normal', colorful)))
                petals_meta.append({
                    'position': sector_idx, 'data_file': data_file,
                    'true_label': 'normal' if has_data else None,
                    'has_data': has_data,
                    'angle_range': list(SECTOR_ANGLES[sector_idx]),
                    'render_params': params,
                })

            out_path = os.path.join(output_dir, f'train_normal_{img_idx:04d}.png')
            self._render_polar(point_sets, out_path)
            self._write_metadata(out_path, {
                'generation_mode': 'A', 'image_type': 'all_normal',
                'colorful': colorful, 'randomize': self.randomize,
                'petals': petals_meta})
            tick(f'全正常图 {img_idx + 1}/{n_normal}')

        logger.info('模式A完成：%d 张图像 -> %s', num_images, output_dir)
        return output_dir

    # ------------------------------------------------------------ 模式B：预测图
    def generate_mode_B(self, data_folder, output_dir, colorful=False,
                        progress_cb=None):
        """
        模式B：把文件夹内的 .npy 按文件名顺序分配进扇形，每 6 个一组。
        超过 6 个文件自动分组成多张图（旧版会静默丢弃）。
        预测图永远使用固定渲染参数（实例参数，由 Predictor 从 checkpoint 恢复），
        不做域随机化。

        返回：[(image_path, metadata), ...]
        """
        os.makedirs(output_dir, exist_ok=True)
        npy_files = sorted(f for f in os.listdir(data_folder) if f.endswith('.npy'))
        if not npy_files:
            logger.warning('模式B：%s 中没有 .npy 文件', data_folder)
            return []

        fixed_params = {'segment_length': self.segment_length,
                        'tau': self.tau, 'jitter': self.jitter}
        results = []
        n_groups = (len(npy_files) + 5) // 6
        for g in range(n_groups):
            group = npy_files[g * 6:(g + 1) * 6]
            point_sets, petals_meta = [], []
            for sector_idx in range(6):
                if sector_idx < len(group):
                    data_file = os.path.join(data_folder, group[sector_idx])
                    theta, r = self.signal_to_petal(load_npy(data_file),
                                                    SECTOR_ANGLES[sector_idx],
                                                    fixed_params)
                    has_data = theta is not None
                else:
                    data_file, theta, r, has_data = None, None, None, False
                point_sets.append((theta, r, self._color_for('normal', colorful)))
                petals_meta.append({
                    'position': sector_idx, 'data_file': data_file,
                    'true_label': None, 'has_data': has_data,
                    'angle_range': list(SECTOR_ANGLES[sector_idx]),
                    'render_params': fixed_params,
                })

            out_path = os.path.join(output_dir, f'predict_{g:03d}.png')
            self._render_polar(point_sets, out_path)
            metadata = {'generation_mode': 'B', 'colorful': colorful,
                        'group_index': g, 'petals': petals_meta}
            self._write_metadata(out_path, metadata)
            results.append((out_path, metadata))
            if progress_cb:
                progress_cb(int((g + 1) / n_groups * 100),
                            f'预测图 {g + 1}/{n_groups}')

        logger.info('模式B完成：%d 个文件 -> %d 张图像', len(npy_files), len(results))
        return results

    # ------------------------------------------------------------ 快速路径：单花瓣
    def render_single_petal(self, data, output_path=None,
                            canvas_size=DEFAULT_IMAGE_SIZE, params=None):
        """
        快速路径：把一条信号直接渲染为"已摆正"的单花瓣 RGBA 图像。
        使用规范扇形 (60°,120°)（中心 90°，正上方），绘制圆心锚点（与六合一图
        保持一致，作为花瓣尖端的定位基准），再按统一几何规范裁剪缩放，
        输出与拆分器完全一致。
        randomize=True 时随机采样渲染参数（训练用域随机化）。

        返回：(petal_rgba 或 None, output_path 或 None)
        """
        p = params or self._sample_params()
        theta, r = self.signal_to_petal(data, (60, 120), p)
        if theta is None:
            return None, None

        fig, ax = plt.subplots(figsize=(canvas_size / self.dpi, canvas_size / self.dpi),
                               dpi=self.dpi, subplot_kw={'projection': 'polar'})
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.scatter(theta, r, s=1.8, alpha=0.6, color=MONO_COLOR,
                   edgecolors='none', zorder=3)
        ax.set_rmax(1.1)
        ax.set_rticks([]), ax.set_xticks([]), ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.plot(0, 0, 'ko', markersize=3, alpha=0.8, zorder=4)  # 圆心锚点
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        rgba = buf[:, :, :4].copy()  # (H, W, 4)

        petal, ok = crop_and_square(rgba, DEFAULT_PETAL_SIZE)
        if not ok:
            return None, None

        if output_path:
            from PIL import Image
            Image.fromarray(petal, 'RGBA').save(output_path)
        return petal, output_path

    def render_single_petals(self, data_root, output_dir, progress_cb=None):
        """
        快速路径批量版：数据目录 -> 按类别组织的单花瓣图像目录。
        目录结构与拆分器输出一致（class_<类别>/*.png），可直接用于训练。

        返回：{类别: 生成数量}
        """
        os.makedirs(output_dir, exist_ok=True)
        data_index = load_dataset_index(data_root)
        total = sum(len(v) for v in data_index.values())
        done = 0
        stats = {}

        for cls in CLASSES:
            files = data_index.get(cls, [])
            class_dir = os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{cls}')
            os.makedirs(class_dir, exist_ok=True)
            count = 0
            for fpath in files:
                stem = os.path.splitext(os.path.basename(fpath))[0]
                out_path = os.path.join(class_dir, f'{stem}_petal.png')
                petal, _ = self.render_single_petal(load_npy(fpath), out_path)
                if petal is not None:
                    count += 1
                done += 1
                if progress_cb and total:
                    progress_cb(int(done / total * 100),
                                f'{cls}: {count}/{len(files)}')
            stats[cls] = count

        logger.info('快速路径完成：%s -> %s（域随机化=%s）',
                    stats, output_dir, self.randomize)
        return stats
