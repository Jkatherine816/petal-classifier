"""
inference/predictor.py - 统一预测器

baseline / 微调模型通用：预处理参数（归一化、输入尺寸）与模型结构
全部从自描述 checkpoint 恢复（修复旧版 predict.py 内容被覆盖、
baseline/微调预处理不一致的问题）。
"""

from core import mpl_setup as _mpl_setup  # noqa: F401  配置 matplotlib 中文字体

import os
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.constants import (
    CLASSES, CLASS_COLORS, SECTOR_CENTERS, METADATA_SUFFIX,
    DEFAULT_INPUT_SIZE, NORM_MEAN, NORM_STD,
)
from core.dataset import build_eval_transform
from core.petal_splitter import PetalSplitter, find_metadata_for
from core.sdp_renderer import SDPRenderer
from core.image_utils import rgba_to_rgb
from models.checkpoint import load_checkpoint, build_model_from_checkpoint
from models.registry import build_model

logger = logging.getLogger(__name__)


class Predictor:
    """统一预测器：加载任意自描述 checkpoint 进行预测"""

    def __init__(self, checkpoint_path, device='auto', model_type=None):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        info = load_checkpoint(checkpoint_path, map_location=self.device)
        if info['legacy']:
            if model_type is None:
                raise ValueError(
                    '旧版 checkpoint 缺少模型结构信息，请提供 model_type '
                    "('densenet' 或 'enhanced')")
            classes = info.get('classes') or CLASSES
            model, _ = build_model(model_type, len(classes))
            model.load_state_dict(info['state_dict'])
            model.to(self.device).eval()
            self.model = model
            self.classes = list(classes)
            preprocess = {'input_size': DEFAULT_INPUT_SIZE,
                          'mean': list(NORM_MEAN), 'std': list(NORM_STD)}
        else:
            self.model = build_model_from_checkpoint(info, self.device)
            self.classes = info.get('classes') or list(CLASSES)
            preprocess = info.get('preprocess') or {}

        self.model_type = info.get('model_type') or model_type
        self.checkpoint_path = checkpoint_path

        self.input_size = int(preprocess.get('input_size', DEFAULT_INPUT_SIZE))
        mean = tuple(preprocess.get('mean', NORM_MEAN))
        std = tuple(preprocess.get('std', NORM_STD))
        self.transform = build_eval_transform(self.input_size, mean, std)

        self.splitter = PetalSplitter(output_size=max(self.input_size, 64))
        logger.info('预测器就绪: %s (%s, 输入 %dx%d, 设备 %s)',
                    os.path.basename(checkpoint_path), self.model_type,
                    self.input_size, self.input_size, self.device)

    # ------------------------------------------------------------ 单花瓣
    def predict_petal(self, petal_image):
        """
        预测单个花瓣（PIL / RGBA ndarray / RGB ndarray）。
        返回 {predicted_class, confidence, probabilities}
        """
        if isinstance(petal_image, np.ndarray):
            arr = petal_image
            pil = Image.fromarray(rgba_to_rgb(arr) if arr.shape[2] == 4 else arr)
        else:
            pil = petal_image.convert('RGB')

        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(tensor)
            probs = F.softmax(output, dim=1)[0]
        conf, idx = probs.max(0)
        return {
            'predicted_class': self.classes[int(idx)],
            'confidence': float(conf),
            'probabilities': {cls: float(probs[i])
                              for i, cls in enumerate(self.classes)},
        }

    # ------------------------------------------------------------ 六合一图
    def predict_image(self, image_path, metadata_path=None):
        """
        预测一张六合一图像的所有花瓣。
        返回 [{position, predicted_class, confidence, probabilities, source_file}]
        """
        if metadata_path is None:
            metadata_path = find_metadata_for(image_path)
        metadata = PetalSplitter.load_metadata(metadata_path)
        source_files = {}
        if metadata:
            for p in metadata.get('petals', []):
                source_files[p.get('position')] = p.get('data_file')

        petals = self.splitter.split(image_path, metadata_path)
        results = []
        for entry in petals:
            if not entry['has_data']:
                continue
            r = self.predict_petal(entry['petal_image'])
            r['position'] = entry['position']
            src = source_files.get(entry['position'])
            r['source_file'] = os.path.basename(src) if src else None
            results.append(r)
            logger.info('  扇形 %d -> %s (%.3f)', entry['position'],
                        r['predicted_class'], r['confidence'])
        return results

    # ------------------------------------------------------------ 数据文件夹
    def predict_folder(self, data_folder, work_dir=None, progress_cb=None):
        """
        从 .npy 数据文件夹预测：渲染模式B 六合一图（自动分组）-> 拆分 -> 预测。

        返回 {'images': [{image_path, results}], 'summary': {...}}
        """
        work_dir = work_dir or os.path.join(data_folder, '_predict_images')
        renderer = SDPRenderer()
        image_list = renderer.generate_mode_B(data_folder, work_dir)
        if not image_list:
            return {'images': [], 'summary': {}}

        all_images = []
        for i, (image_path, _meta) in enumerate(image_list):
            results = self.predict_image(image_path)
            all_images.append({'image_path': image_path, 'results': results})
            if progress_cb:
                progress_cb(int((i + 1) / len(image_list) * 100),
                            f'已预测 {i + 1}/{len(image_list)} 张图')

        return {'images': all_images,
                'summary': self.summarize(all_images)}

    # ------------------------------------------------------------ 结果汇总/标注/导出
    def summarize(self, images_results):
        """汇总多个图像的预测结果。"""
        class_counts, total_conf = {}, {}
        n = 0
        for img in images_results:
            for r in img['results']:
                cls = r['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
                total_conf[cls] = total_conf.get(cls, 0.0) + r['confidence']
                n += 1
        return {
            'total_petals': n,
            'class_distribution': class_counts,
            'average_confidence': {c: total_conf[c] / class_counts[c]
                                   for c in class_counts},
        }

    def annotate_image(self, image_path, results, output_path):
        """在六合一图上叠加每个花瓣的预测标签与置信度。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        img = np.array(Image.open(image_path).convert('RGBA'))
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.42

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.imshow(img)
        ax.axis('off')

        for r in results:
            theta = np.deg2rad(SECTOR_CENTERS[r['position']])
            x = cx + radius * np.cos(theta)
            y = cy - radius * np.sin(theta)   # 图像 y 向下
            color = CLASS_COLORS.get(r['predicted_class'], '#CCCCCC')
            ax.text(x, y, f"{r['predicted_class']}\n{r['confidence']:.3f}",
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                              alpha=0.85, edgecolor='black', linewidth=1.5))

        ax.set_title(f'预测结果 - {os.path.basename(image_path)}')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    transparent=True)
        plt.close(fig)
        logger.info('标注图已保存: %s', output_path)
        return output_path

    @staticmethod
    def export_results(payload, out_dir, base_name='prediction_results'):
        """导出预测结果 json + csv。"""
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f'{base_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(out_dir, f'{base_name}.csv')
        import csv
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['图像', '扇形位置', '预测类别', '置信度', '来源文件'])
            for img in payload.get('images', []):
                for r in img['results']:
                    writer.writerow([
                        os.path.basename(img['image_path']), r['position'],
                        r['predicted_class'], f"{r['confidence']:.4f}",
                        r.get('source_file') or ''])
        logger.info('结果已导出: %s / %s', json_path, csv_path)
        return json_path, csv_path
