"""
gui/pages/page_split.py - ③ 花瓣拆分页

两种路径：
    标准路径：六合一图像 -> 拆分摆正 -> 按类目录落盘；
    快速路径：npy 信号直接渲染单花瓣（跳过六合一往返）。
可视化：任选一张六合一图，实时拆分展示 6 个花瓣与标签（镜像修复效果肉眼可验），
并显示整个输出目录的标签分布。
"""

import os

import numpy as np
from core import mpl_setup as _mpl_setup  # noqa: F401  配置 matplotlib 中文字体
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QCheckBox, QPushButton, QSplitter,
                               QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                               QMessageBox)

from core.constants import CLASSES, CLASS_NAMES_ZH
from core.petal_splitter import PetalSplitter
from core.sdp_renderer import SDPRenderer
from .base_page import BasePage


class SplitPage(BasePage):
    TITLE = '③ 花瓣拆分 / 训练图生成'
    DESC = ('标准路径：把六合一图像拆成 6 个摆正的单花瓣并按标签落盘；'
            '快速路径：直接从信号渲染单花瓣，跳过六合一往返（更快、无拆分误差）。')

    def build_form(self):
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(['标准路径（拆分六合一图像）',
                                  '快速路径（信号直渲染单花瓣）'])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_change)
        self.form.addRow('处理路径', self.combo_mode)

        self.edit_in = self.add_dir_row(
            '输入目录', '', '标准路径=六合一图像目录；快速路径=npy 数据目录')

        self.chk_gray = QCheckBox('输出转为灰度（防颜色捷径保险，训练图建议开启）')
        self.chk_gray.setChecked(True)
        self.form.addRow('', self.chk_gray)

        self.edit_out = self.add_dir_row(
            '输出目录', self.ctx.default_path('petals'),
            'class_<类别>/ 结构的训练图像目录')

        # 预览控制行
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel('预览图像:'))
        self.combo_image = QComboBox()
        self.combo_image.currentIndexChanged.connect(self._preview_split)
        preview_row.addWidget(self.combo_image, stretch=1)
        btn_export = QPushButton('导出本步产物')
        btn_export.clicked.connect(self.on_export)
        preview_row.addWidget(btn_export)
        preview_widget = QWidget()
        preview_widget.setLayout(preview_row)

        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.result_layout.addWidget(preview_widget)
        self.result_layout.addWidget(self.canvas)

    # ------------------------------------------------------------
    def _on_mode_change(self, idx):
        fast = idx == 1
        default_in = self.ctx.data_dir if fast else self.ctx.image_dir
        if default_in:
            self.edit_in.setText(default_in)
        self.chk_gray.setEnabled(not fast)  # 快速路径本就单色

    def on_run(self):
        in_dir = self.edit_in.text().strip()
        out_dir = self.edit_out.text().strip()
        if not in_dir or not out_dir:
            return

        fast = self.combo_mode.currentIndex() == 1
        if fast:
            renderer = SDPRenderer()
            self.run_task(renderer.render_single_petals, in_dir, out_dir,
                          on_done=lambda stats: self._show_results(out_dir, stats))
        else:
            splitter = PetalSplitter()
            self.run_task(splitter.batch_split, in_dir, out_dir,
                          self.chk_gray.isChecked(),
                          on_done=lambda stats: self._show_results(out_dir, stats))

    # ------------------------------------------------------------ 结果展示
    def _show_results(self, out_dir, stats):
        self.ctx.petals_dir = out_dir
        self._draw_distribution(stats)

        # 标准路径下填充预览下拉框
        in_dir = self.edit_in.text().strip()
        if self.combo_mode.currentIndex() == 0 and os.path.isdir(in_dir):
            images = sorted(f for f in os.listdir(in_dir) if f.endswith('.png'))
            self.combo_image.blockSignals(True)
            self.combo_image.clear()
            self.combo_image.addItems(images)
            self.combo_image.blockSignals(False)
            if images:
                self._preview_split(0)

    def _draw_distribution(self, stats):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        names = [c for c in CLASSES if c in stats]
        counts = [stats[c] for c in names]
        bars = ax.bar([f'{c}\n{CLASS_NAMES_ZH[c]}' for c in names], counts,
                      color='#45B7D1')
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                    ha='center', va='bottom', fontsize=9)
        ax.set_title('输出目录标签分布')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _preview_split(self, idx):
        """实时拆分选中的六合一图，展示 6 花瓣 + 标签。"""
        in_dir = self.edit_in.text().strip()
        name = self.combo_image.currentText()
        if not name or not os.path.isdir(in_dir):
            return
        image_path = os.path.join(in_dir, name)
        splitter = PetalSplitter()
        petals = splitter.split(image_path)

        self.fig.clear()
        axes = self.fig.subplots(2, 3)
        for ax, entry in zip(axes.flat, petals):
            if entry['has_data']:
                ax.imshow(entry['petal_image'])
                label = entry['label'] or '未知'
                zh = CLASS_NAMES_ZH.get(label, '')
                ax.set_title(f"扇形{entry['position']} -> {label} {zh}",
                             fontsize=10,
                             color='#2E7D32' if label in CLASSES else '#888')
            else:
                ax.set_title(f"扇形{entry['position']} -> 无数据",
                             fontsize=10, color='#BBB')
                ax.imshow(np.zeros((8, 8, 3), dtype=np.uint8))
            ax.axis('off')
        self.fig.suptitle(os.path.basename(image_path), fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    def on_export(self):
        dest = self.export_to_root(
            '03_花瓣拆分', src_dir=self.edit_out.text().strip(),
            description='单花瓣训练图像',
            params={'mode': 'fast' if self.combo_mode.currentIndex() else 'standard',
                    'grayscale': self.chk_gray.isChecked()})
        if dest:
            print(f'已导出到 {dest}')
