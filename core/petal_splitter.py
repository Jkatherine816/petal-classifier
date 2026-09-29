"""
core/petal_splitter.py - 花瓣拆分器（镜像问题已修复）

修复要点（相对旧版 petal_splitter.py / petal_splitter_fixed.py）：
    旧版用 arctan2(dy, dx) 测量像素角度，忽略了图像 y 轴向下，
    导致 pos i 实际抓到 pos 5-i 的花瓣（镜像错位）；
    "fixed" 版把角度测量反过来、同时又把扇形列表反转，两者抵消，修复无效。
    本版：角度测量改为 arctan2(-dy, dx)，扇形列表保持 constants 中的定义不动。

    回归验证见 tests/test_pipeline.py（颜色自校验 + 摆正方向校验）。
"""

import os
import json
import glob
import logging

import numpy as np
import cv2
from PIL import Image

from .constants import (
    CLASSES, CLASS_DIR_PREFIX, METADATA_SUFFIX,
    SECTOR_ANGLES, SECTOR_CENTERS, POSITION_TO_TYPE,
    DEFAULT_PETAL_SIZE,
)
from .image_utils import crop_and_square, rgba_to_rgb, to_gray_rgb

logger = logging.getLogger(__name__)

# 判定扇形"有数据"的最少非透明像素数
MIN_PIXELS = 50


class PetalSplitter:
    """六合一图像 -> 6 个摆正的单花瓣图像"""

    def __init__(self, output_size=DEFAULT_PETAL_SIZE, debug=False):
        self.output_size = output_size
        self.debug = debug

    # ------------------------------------------------------------ 几何
    @staticmethod
    def sector_mask(image_shape, sector_idx):
        """
        生成扇形掩码。角度按数学坐标测量：angle = degrees(arctan2(-dy, dx)) % 360。
        """
        height, width = image_shape[:2]
        cx, cy = width // 2, height // 2

        y_coords, x_coords = np.ogrid[:height, :width]
        dx = x_coords - cx
        dy = y_coords - cy

        angles = np.degrees(np.arctan2(-dy, dx))   # 关键修复：-dy
        angles = np.mod(angles, 360.0)

        start, end = SECTOR_ANGLES[sector_idx]
        if start < end:
            angle_mask = (angles >= start) & (angles <= end)
        else:  # 跨 0°（本定义下不会发生，保留防御）
            angle_mask = (angles >= start) | (angles <= end)

        distances = np.sqrt(dx ** 2 + dy ** 2)
        circle_mask = distances <= min(width, height) // 2
        return angle_mask & circle_mask

    def extract_and_align(self, image_rgba, sector_idx):
        """
        提取扇形区域并摆正（扇形中心线旋转到正上方），再裁剪缩放为正方形。

        返回：(petal_rgba, has_data)
        """
        if image_rgba is None:
            return None, False

        height, width = image_rgba.shape[:2]
        mask = self.sector_mask((height, width), sector_idx)

        sector_image = np.zeros((height, width, 4), dtype=np.uint8)
        for c in range(4):
            sector_image[:, :, c] = image_rgba[:, :, c] * mask.astype(np.uint8)

        alpha = sector_image[:, :, 3]
        if int(np.sum(alpha > 10)) <= MIN_PIXELS:
            return None, False

        # 摆正：cv2 正角度 = 数学正方向旋转，把扇形中心转到 90°（正上方）
        rotation_angle = 90.0 - SECTOR_CENTERS[sector_idx]
        rot_mat = cv2.getRotationMatrix2D((width // 2, height // 2),
                                          rotation_angle, 1.0)
        rotated = cv2.warpAffine(
            sector_image, rot_mat, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        return crop_and_square(rotated, self.output_size)

    # ------------------------------------------------------------ 标签
    @staticmethod
    def load_metadata(metadata_path):
        if not metadata_path or not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc:
            logger.warning('加载元数据失败 %s: %s', metadata_path, exc)
            return None

    @staticmethod
    def get_label(sector_idx, metadata):
        """
        确定花瓣标签：
            模式A：优先元数据 true_label；缺失时按 image_type 回退
                   （all_normal -> normal；fixed_anomalies -> 位置映射）。
            模式B / 无元数据：返回 None（未知，供预测流程）。
        """
        if not metadata:
            return None

        petals = metadata.get('petals') or []
        if sector_idx < len(petals):
            info = petals[sector_idx] or {}
            true_label = info.get('true_label')
            if true_label and str(true_label).lower() not in ('none', 'null'):
                return true_label
            if info.get('has_data') is False:
                return None

        if metadata.get('generation_mode') == 'A':
            image_type = metadata.get('image_type')
            if image_type == 'all_normal':
                return 'normal'
            if image_type == 'fixed_anomalies':
                return POSITION_TO_TYPE.get(sector_idx)
        return None

    # ------------------------------------------------------------ 拆分
    def split(self, image_path, metadata_path=None, output_dir=None,
              strip_color=False):
        """
        拆分一张六合一图像。

        参数：
            image_path: 六合一图像路径
            metadata_path: 元数据路径（自动探测 <同名>_metadata.json）
            output_dir: 若提供，有标签的花瓣按 class_<标签>/ 落盘（RGB PNG）
            strip_color: True 时把花瓣转灰度再复制 3 通道（防颜色捷径的保险）
        返回：
            petals: 6 项列表 [{position, has_data, petal_image, label, saved_path}]
        """
        try:
            image = np.array(Image.open(image_path).convert('RGBA'))
        except Exception as exc:
            logger.error('加载图像失败 %s: %s', image_path, exc)
            return []

        if metadata_path is None:
            candidate = image_path.replace('.png', METADATA_SUFFIX)
            metadata_path = candidate if os.path.exists(candidate) else None
        metadata = self.load_metadata(metadata_path)

        if output_dir:
            for cls in CLASSES:
                os.makedirs(os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{cls}'),
                            exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        petals = []
        for sector_idx in range(6):
            petal_img, has_data = self.extract_and_align(image, sector_idx)
            entry = {'position': sector_idx, 'has_data': has_data,
                     'petal_image': None, 'label': None, 'saved_path': None}
            if has_data:
                label = self.get_label(sector_idx, metadata)
                entry['label'] = label
                entry['petal_image'] = petal_img

                if output_dir and label in CLASSES:
                    rgb = rgba_to_rgb(petal_img)
                    if strip_color:
                        rgb = to_gray_rgb(rgb)
                    class_dir = os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{label}')
                    filename = f'{base_name}_pos{sector_idx}_{label}.png'
                    save_path = os.path.join(class_dir, filename)
                    Image.fromarray(rgb, 'RGB').save(save_path)
                    entry['saved_path'] = save_path

                if self.debug:
                    logger.debug('扇形 %d -> 标签 %s', sector_idx, label)
            petals.append(entry)

        valid = sum(1 for p in petals if p['has_data'])
        logger.info('拆分 %s：%d 个有效花瓣', os.path.basename(image_path), valid)
        return petals

    def batch_split(self, input_dir, output_dir, strip_color=False,
                    progress_cb=None):
        """
        批量拆分目录下所有六合一图像（自动配对元数据）。

        返回：{类别: 样本数}（仅统计成功落盘的有标签花瓣）
        """
        os.makedirs(output_dir, exist_ok=True)
        image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        if not image_files:
            logger.warning('批量拆分：%s 中没有 PNG 图像', input_dir)
            return {}

        stats = {}
        for i, image_file in enumerate(image_files):
            petals = self.split(image_file, output_dir=output_dir,
                                strip_color=strip_color)
            for p in petals:
                if p['has_data'] and p['label'] in CLASSES and p['saved_path']:
                    stats[p['label']] = stats.get(p['label'], 0) + 1
            if progress_cb:
                progress_cb(int((i + 1) / len(image_files) * 100),
                            f'已拆分 {i + 1}/{len(image_files)}：'
                            f'{os.path.basename(image_file)}')

        stats_path = os.path.join(output_dir, 'split_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info('批量拆分完成：%s -> %s', stats, output_dir)
        return stats


def find_metadata_for(image_path):
    """按命名约定查找六合一图像对应的元数据文件。"""
    base = os.path.splitext(image_path)[0]
    for candidate in (base + METADATA_SUFFIX, base + '.json'):
        if os.path.exists(candidate):
            return candidate
    return None
