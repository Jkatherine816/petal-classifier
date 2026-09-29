"""
core/image_utils.py - 花瓣图像的公共几何处理

拆分器与"快速路径"单花瓣渲染共用同一套裁剪/缩放逻辑，
保证两条路径产生的花瓣图像几何规范完全一致。
"""

import numpy as np
import cv2


def crop_and_square(image_rgba, output_size, margin=10, alpha_threshold=10):
    """
    将 RGBA 图像按 alpha 通道裁剪出有效区域，等比缩放后居中放入正方形画布。

    参数：
        image_rgba: (H, W, 4) uint8 图像
        output_size: 输出正方形边长
        margin: 裁剪边距（像素）
        alpha_threshold: 判定"有内容"的 alpha 阈值
    返回：
        (output_image, ok): (output_size, output_size, 4) 图像；无有效内容时 ok=False
    """
    if image_rgba is None:
        return None, False

    alpha = image_rgba[:, :, 3]
    non_zero = np.where(alpha > alpha_threshold)
    if len(non_zero[0]) == 0:
        return None, False

    h_img, w_img = image_rgba.shape[:2]
    y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
    x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])

    y_min = max(0, y_min - margin)
    y_max = min(h_img, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(w_img, x_max + margin)

    cropped = image_rgba[y_min:y_max, x_min:x_max]
    h, w = cropped.shape[:2]
    if h <= 0 or w <= 0:
        return None, False

    scale = min(output_size / h, output_size / w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    if scale != 1.0:
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cropped

    output = np.zeros((output_size, output_size, 4), dtype=np.uint8)
    y_off = (output_size - new_h) // 2
    x_off = (output_size - new_w) // 2
    output[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return output, True


def rgba_to_rgb(image_rgba, background=(0, 0, 0)):
    """RGBA -> RGB，透明区域以指定底色填充（默认黑底）。"""
    if image_rgba.shape[2] == 3:
        return image_rgba
    rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB)
    alpha = image_rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.zeros_like(rgb)
    bg[:, :] = background
    out = (rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha))
    return out.astype(np.uint8)


def to_gray_rgb(image_rgb):
    """RGB -> 灰度后再复制成 3 通道，彻底去除颜色信息（防颜色捷径的保险）。"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def petal_centroid_direction(image_rgba, alpha_threshold=10):
    """
    返回花瓣像素质心相对图像中心的方向向量 (dx, dy)，
    其中 dy 为数学方向（向上为正）。用于验证摆正效果。
    """
    alpha = image_rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0:
        return None
    h, w = alpha.shape
    cx, cy = w / 2.0, h / 2.0
    return float(xs.mean() - cx), float(cy - ys.mean())
