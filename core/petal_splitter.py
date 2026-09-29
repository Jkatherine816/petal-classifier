"""
core/petal_splitter.py - 六合一图像拆分器

把 224×224 六合一 SDP 图像拆成 6 个 64×64 单花瓣，旋转对齐到统一方向，
并按标签目录落盘（训练数据组织）。

【核心修复 - 扇形镜像 bug】
旧版两个拆分器在提取位置 i 时实际取到的是位置 5-i 的内容
（极坐标数学角度 vs 图像像素 y 轴向下导致的镜像）。
本版像素角度恢复严格使用：
    angles = degrees(arctan2(-dy, dx)) % 360
扇形表 SECTOR_ANGLES 保持 [(0,60),...,(300,360)] 不反转。
已加入像素级回归测试（tests/test_pipeline.py::TestSplitLabelConsistency）。

旋转对齐：把扇形中心方向转到正上方（90°）：
    rotation_angle = 90 - SECTOR_CENTERS[idx]
cv2.getRotationMatrix2D 的正角度 = 数学正方向（逆时针）旋转。
"""

import os
import json
import logging

import cv2
import numpy as np
from PIL import Image

from .constants import (CLASSES, SECTOR_ANGLES, SECTOR_CENTERS,
                        CLASS_DIR_PREFIX, METADATA_SUFFIX,
                        POSITION_TO_TYPE, DEFAULT_INPUT_SIZE)
from .image_utils import crop_and_square, to_gray_rgb

logger = logging.getLogger(__name__)


class PetalSplitter:
    def __init__(self, output_size=DEFAULT_INPUT_SIZE, debug=False):
        self.output_size = output_size
        self.debug = debug

    # ------------------------------------------------------------ 几何
    def sector_mask(self, shape, center, radius, angle_lo, angle_hi):
        """生成扇形掩膜（角度为数学约定：东 0° 逆时针；图像 y 轴向下需取负）"""
        h, w = shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        dx = xx - center[0]
        dy = yy - center[1]
        r = np.hypot(dx, dy)
        angles = np.degrees(np.arctan2(-dy, dx))   # 关键：-dy
        angles = np.mod(angles, 360.0)
        mask = (r <= radius) & (r > 3) & (angles >= angle_lo) & (angles < angle_hi)
        return mask

    def find_center_radius(self, gray):
        """圆心=暗色中心点，半径=前景最大距离"""
        h, w = gray.shape
        fg = gray < 240
        ys, xs = np.nonzero(fg)
        if len(xs) == 0:
            return (w / 2, h / 2), min(h, w) * 0.42
        # 圆心取图像几何中心附近最暗的点（中心锚点黑点）
        cy, cx = h / 2, w / 2
        d = np.hypot(xs - cx, ys - cy)
        inner = fg & (d < min(h, w) * 0.05)
        if inner.any():
            iy, ix = np.nonzero(inner)
            cx, cy = ix.mean(), iy.mean()
        radius = np.percentile(np.hypot(xs - cx, ys - cy), 99.5)
        return (cx, cy), radius

    def extract_and_align(self, img, mask, center, idx):
        """抠出扇形 idx 并旋转到统一方向（中心朝正上方）"""
        out = img.copy()
        out[~mask] = 255
        rotation_angle = 90.0 - SECTOR_CENTERS[idx]
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(out, M, (w, h), borderValue=(255, 255, 255))
        return rotated

    # ------------------------------------------------------------ 标签
    def get_label(self, idx, image_name, metadata=None):
        """
        优先级：metadata 的 true_label > 文件名规则（模式A）> None（模式B未知）
        """
        if metadata:
            petals = metadata.get('petals') or []
            if idx < len(petals) and petals[idx]:
                tl = petals[idx].get('true_label')
                if tl in CLASSES:
                    return tl
        if 'all_normal' in image_name or 'train_normal' in image_name:
            return 'normal'
        if 'fixed_anomal' in image_name or 'train_fixed' in image_name:
            return POSITION_TO_TYPE.get(idx)
        return None

    # ------------------------------------------------------------ 主流程
    def split(self, image_path, metadata_path=None, output_dir=None,
              strip_color=False):
        """
        拆分一张六合一图，返回 6 个 dict：
        {position, has_data, petal_image(RGB np), label, saved_path}
        """
        img = np.array(Image.open(image_path).convert('RGB'))
        if strip_color:
            img = to_gray_rgb(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        center, radius = self.find_center_radius(gray)

        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, encoding='utf-8') as f:
                metadata = json.load(f)

        base = os.path.splitext(os.path.basename(image_path))[0]
        results = []
        for idx, (lo, hi) in enumerate(SECTOR_ANGLES):
            mask = self.sector_mask(gray.shape, center, radius, lo, hi)
            has_data = bool((gray[mask] < 240).sum() > 50)
            aligned = self.extract_and_align(img, mask, center, idx)
            petal = crop_and_square(
                np.dstack([aligned, np.full(aligned.shape[:2], 255,
                                            dtype=np.uint8)]),
                self.output_size)
            label = self.get_label(idx, base, metadata)
            saved = None
            if output_dir and label:
                cls_dir = os.path.join(output_dir, f'{CLASS_DIR_PREFIX}{label}')
                os.makedirs(cls_dir, exist_ok=True)
                saved = os.path.join(cls_dir, f'{base}_pos{idx}_{label}.png')
                petal.save(saved)
            results.append({
                'position': idx,
                'has_data': has_data,
                'petal_image': np.array(petal),
                'label': label,
                'saved_path': saved,
            })
        return results

    def batch_split(self, input_dir, output_dir, strip_color=False,
                    progress_cb=None):
        """批量拆分目录下所有六合一图，写 split_stats.json"""
        os.makedirs(output_dir, exist_ok=True)
        names = [fn for fn in sorted(os.listdir(input_dir))
                 if fn.lower().endswith('.png')]
        stats = {'total_images': 0, 'total_petals': 0, 'labeled': 0,
                 'unlabeled': 0, 'per_class': {c: 0 for c in CLASSES}}
        for i, fn in enumerate(names):
            img_path = os.path.join(input_dir, fn)
            meta_path = self.find_metadata_for(img_path)
            results = self.split(img_path, meta_path, output_dir,
                                 strip_color=strip_color)
            stats['total_images'] += 1
            for r in results:
                if not r['has_data']:
                    continue
                stats['total_petals'] += 1
                if r['label']:
                    stats['labeled'] += 1
                    stats['per_class'][r['label']] += 1
                else:
                    stats['unlabeled'] += 1
            if progress_cb:
                progress_cb(int((i + 1) / len(names) * 100),
                            f'{i + 1}/{len(names)} {fn}')
        with open(os.path.join(output_dir, 'split_stats.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info('拆分完成: %s', stats)
        return stats

    @staticmethod
    def find_metadata_for(image_path):
        """六合一图对应的元数据文件路径（可能不存在）"""
        base = os.path.splitext(image_path)[0]
        cand = base + METADATA_SUFFIX
        return cand if os.path.exists(cand) else None
