"""
core/image_utils.py - 图像几何工具
"""

import numpy as np
from PIL import Image


def crop_and_square(image_rgba, output_size, margin=10):
    """
    按非透明像素包围盒裁剪并补白为正方形，缩放到 output_size。
    输入：RGBA numpy 数组 (H, W, 4)；输出：RGB PIL 图像。
    """
    alpha = image_rgba[..., 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        # 空图：返回白底
        return Image.new('RGB', (output_size, output_size), (255, 255, 255))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    cmin = max(0, cmin - margin)
    rmax = min(image_rgba.shape[0] - 1, rmax + margin)
    cmax = min(image_rgba.shape[1] - 1, cmax + margin)

    cropped = image_rgba[rmin:rmax + 1, cmin:cmax + 1, :3]
    h, w = cropped.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = cropped
    img = Image.fromarray(canvas)
    return img.resize((output_size, output_size), Image.LANCZOS)


def rgba_to_rgb(fig_array):
    """matplotlib buffer_rgba 的 (H,W,4) -> (H,W,3) 白底合成"""
    rgb = fig_array[..., :3].astype(np.float32)
    alpha = fig_array[..., 3:4].astype(np.float32) / 255.0
    return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)


def to_gray_rgb(img_array):
    """RGB -> 灰度 -> 三通道（保持模型输入通道数一致）"""
    gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def petal_centroid_direction(mask):
    """掩膜质心方向（度，数学角度约定：东 0° 逆时针），用于自检"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cy, cx = mask.shape[0] / 2, mask.shape[1] / 2
    dy = ys.mean() - cy
    dx = xs.mean() - cx
    return float(np.degrees(np.arctan2(-dy, dx)) % 360.0)
