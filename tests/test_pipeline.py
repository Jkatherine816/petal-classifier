"""
tests/test_pipeline.py - 核心流水线回归测试

重点：角度/标签一致性（防镜像问题回归）、单色渲染、模式B分组、
快速路径与标准路径的几何一致性。

运行：pytest tests/ -v
"""

import os
import json

import numpy as np
import pytest

from core.signal_generator import SignalGenerator
from core.sdp_renderer import SDPRenderer
from core.petal_splitter import PetalSplitter
from core.constants import (POSITION_TO_TYPE, CLASS_COLORS, CLASSES,
                            SECTOR_CENTERS)
from core.image_utils import rgba_to_rgb


def _hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


@pytest.fixture(scope='module')
def tiny_dataset(tmp_path_factory):
    """小规模数据集（7 类 × 3 条）"""
    root = tmp_path_factory.mktemp('data')
    gen = SignalGenerator(length=3000, seed=42)
    stats = gen.generate_dataset(str(root), num_samples_per_class=3)
    assert all(v == 3 for v in stats.values())
    return str(root)


@pytest.fixture(scope='module')
def colorful_image(tiny_dataset, tmp_path_factory):
    """彩色模式渲染一张固定异常图（彩色仅用于颜色自校验）"""
    out = tmp_path_factory.mktemp('img')
    SDPRenderer(seed=0).generate_mode_A(tiny_dataset, str(out),
                                        num_images=2, colorful=True)
    return os.path.join(str(out), 'train_fixed_0000.png')


class TestSplitLabelConsistency:
    """镜像修复回归：拆分器提取的花瓣内容必须与元数据标签一致"""

    def test_labels_match_metadata(self, colorful_image):
        splitter = PetalSplitter()
        petals = splitter.split(colorful_image)
        assert len(petals) == 6
        for entry in petals:
            assert entry['has_data'], f"pos{entry['position']} 应有数据"
            assert entry['label'] == POSITION_TO_TYPE[entry['position']], \
                f"pos{entry['position']} 标签错误: {entry['label']}"

    def test_pixel_content_matches_label(self, colorful_image):
        """颜色自校验：按像素颜色判断实际内容类别，与标签比对。"""
        splitter = PetalSplitter()
        petals = splitter.split(colorful_image)
        for entry in petals:
            img = entry['petal_image']
            mask = img[:, :, 3] > 100
            mean_rgb = img[:, :, :3][mask].mean(axis=0)
            expected = _hex2rgb(CLASS_COLORS[POSITION_TO_TYPE[entry['position']]])
            assert all(abs(a - b) < 40 for a, b in zip(mean_rgb, expected)), \
                f"pos{entry['position']} 内容为镜像错位: {mean_rgb} vs {expected}"


class TestMonochromeDefault:
    """训练模式必须单色渲染（防颜色捷径）"""

    def test_mode_a_monochrome(self, tiny_dataset, tmp_path_factory):
        out = tmp_path_factory.mktemp('mono')
        SDPRenderer(seed=0).generate_mode_A(tiny_dataset, str(out),
                                            num_images=2, colorful=False)
        splitter = PetalSplitter()
        petals = splitter.split(os.path.join(str(out), 'train_fixed_0000.png'))
        for entry in petals:
            rgb = rgba_to_rgb(entry['petal_image'])
            # R=G=B（灰度），允许抗锯齿小误差
            spread = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)
            assert spread.max() <= 4, '训练图像不是单色，存在颜色泄漏'


class TestModeBGrouping:
    """模式B：>6 个文件自动分组，不再静默丢弃"""

    def test_grouping(self, tmp_path):
        folder = tmp_path / 'npy'
        folder.mkdir()
        rng = np.random.default_rng(0)
        for i in range(13):
            np.save(folder / f'sig_{i:02d}.npy', rng.standard_normal(3000))
        out = tmp_path / 'imgB'
        results = SDPRenderer(seed=0).generate_mode_B(str(folder), str(out))
        assert len(results) == 3
        # 元数据花瓣总数应覆盖全部 13 个文件
        total = 0
        for _path, meta in results:
            total += sum(1 for p in meta['petals'] if p['has_data'])
        assert total == 13


class TestFastPath:
    """快速路径：与标准路径几何一致（同信号、花瓣朝上）"""

    def test_single_petal_consistent(self, tiny_dataset, tmp_path):
        data = np.load(os.path.join(
            tiny_dataset, 'class_outlier', 'outlier_000.npy'))
        renderer = SDPRenderer(seed=0)
        petal_fast, _ = renderer.render_single_petal(data)
        assert petal_fast is not None and petal_fast.shape[2] == 4

        # 标准路径：同信号放扇形0再拆分
        theta, r = renderer.signal_to_petal(data, (0, 60))
        img_path = str(tmp_path / 'sector.png')
        renderer._render_polar([(theta, r, '#4A4A4A')], img_path)
        petal_split = PetalSplitter().split(img_path)[0]['petal_image']

        a = rgba_to_rgb(petal_fast).astype(np.float32).mean(axis=2)
        b = rgba_to_rgb(petal_split).astype(np.float32).mean(axis=2)
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
        assert corr > 0.5, f'快速路径与标准路径不一致: corr={corr:.3f}'

    def test_render_single_petals_stats(self, tiny_dataset, tmp_path):
        out = tmp_path / 'petals'
        stats = SDPRenderer(seed=0).render_single_petals(tiny_dataset, str(out))
        assert set(stats) == set(CLASSES)
        assert all(v == 3 for v in stats.values())
