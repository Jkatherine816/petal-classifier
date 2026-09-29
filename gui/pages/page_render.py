"""
gui/pages/page_render.py - ② 图像生成页（模式A：训练用六合一图像）
"""

import os
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QSpinBox, QCheckBox, QPushButton, QSplitter,
                               QLabel, QPlainTextEdit, QWidget, QVBoxLayout,
                               QListWidgetItem)

from core.sdp_renderer import SDPRenderer
from core.constants import DEFAULT_IMAGE_SIZE, DEFAULT_DPI
from gui.widgets.image_gallery import ImageGallery
from .base_page import BasePage


class RenderPage(BasePage):
    TITLE = '② 图像生成（六合一 SDP 图）'
    DESC = ('把时序信号渲染成六合一极坐标 SDP 图像（模式A：训练用，固定位置布局）。'
            '训练图像一律单色渲染，杜绝"颜色捷径"；彩色仅供人工预览核对。')

    def build_form(self):
        self.edit_in = self.add_dir_row(
            '数据目录', self.ctx.data_dir or self.ctx.default_path('data'),
            '包含 class_<类别>/ 的 npy 数据集目录')

        self.spin_num = QSpinBox()
        self.spin_num.setRange(2, 100000)
        self.spin_num.setValue(500)
        self.form.addRow('图像数量', self.spin_num)

        self.spin_size = QSpinBox()
        self.spin_size.setRange(96, 1024)
        self.spin_size.setValue(DEFAULT_IMAGE_SIZE)
        self.form.addRow('图像尺寸(px)', self.spin_size)

        self.chk_colorful = QCheckBox('彩色预览（仅人工核对，勿用于训练）')
        self.chk_colorful.setChecked(False)
        self.form.addRow('', self.chk_colorful)

        self.edit_out = self.add_dir_row(
            '输出目录', self.ctx.default_path('images'),
            '六合一图像与元数据输出目录')

        # 结果区：画廊 + 元数据查看
        splitter = QSplitter(Qt.Horizontal)
        self.gallery = ImageGallery()
        splitter.addWidget(self.gallery)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel('元数据（点击左侧图像查看）'))
        self.meta_view = QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        right_layout.addWidget(self.meta_view)
        btn_export = QPushButton('导出本步产物')
        btn_export.clicked.connect(self.on_export)
        right_layout.addWidget(btn_export)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.result_layout.addWidget(splitter)

        self.gallery.list.currentItemChanged.connect(self._show_metadata)

    def on_run(self):
        in_dir = self.edit_in.text().strip()
        out_dir = self.edit_out.text().strip()
        if not in_dir or not out_dir:
            return
        renderer = SDPRenderer(image_size=self.spin_size.value())
        self.run_task(renderer.generate_mode_A, in_dir, out_dir,
                      self.spin_num.value(),
                      self.chk_colorful.isChecked(),
                      on_done=lambda _r: self._show_results(out_dir))

    def _show_results(self, out_dir):
        self.ctx.image_dir = out_dir
        images = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                        if f.endswith('.png'))
        self.gallery.set_images(images)
        print(f'六合一图像生成完成：{len(images)} 张 -> {out_dir}')

    def _show_metadata(self, item, _prev=None):
        if not item:
            return
        row = self.gallery.list.row(item)
        path = self.gallery._paths[row]
        meta_path = path.replace('.png', '_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta_view.setPlainText(
                    json.dumps(json.load(f), indent=2, ensure_ascii=False))
        else:
            self.meta_view.setPlainText('（无元数据文件）')

    def on_export(self):
        dest = self.export_to_root(
            '02_图像生成', src_dir=self.edit_out.text().strip(),
            description='六合一 SDP 训练图像',
            params={'num_images': self.spin_num.value(),
                    'image_size': self.spin_size.value(),
                    'colorful': self.chk_colorful.isChecked()})
        if dest:
            print(f'已导出到 {dest}')
