"""
gui/pages/page_generate.py - ① 数据生成页
"""

import os

import numpy as np
from core import mpl_setup as _mpl_setup  # noqa: F401  配置 matplotlib 中文字体
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QSpinBox, QPushButton, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QSplitter,
                               QWidget, QVBoxLayout)

from core.constants import (CLASSES, CLASS_NAMES_ZH,
                            DEFAULT_SIGNAL_LENGTH, DEFAULT_SAMPLING_RATE)
from core.signal_generator import SignalGenerator
from .base_page import BasePage


class GeneratePage(BasePage):
    TITLE = '① 数据生成'
    DESC = ('生成 7 类时序信号（正常 + 6 种异常），保存为 .npy 并按 '
            'class_<类别>/ 目录组织。设定随机种子可完全复现。')

    def build_form(self):
        self.spin_length = QSpinBox()
        self.spin_length.setRange(1000, 1000000)
        self.spin_length.setValue(DEFAULT_SIGNAL_LENGTH)
        self.spin_length.setSingleStep(10000)
        self.form.addRow('信号长度 L', self.spin_length)

        self.spin_fs = QSpinBox()
        self.spin_fs.setRange(1, 10000)
        self.spin_fs.setValue(DEFAULT_SAMPLING_RATE)
        self.form.addRow('采样频率 Fs', self.spin_fs)

        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 10000)
        self.spin_count.setValue(100)
        self.form.addRow('每类样本数', self.spin_count)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setSpecialValueText('随机')
        self.form.addRow('随机种子', self.spin_seed)

        self.edit_out = self.add_dir_row(
            '输出目录', self.ctx.default_path('data'), '存放 npy 数据集的目录')

        # 结果区：波形预览 + 统计表
        splitter = QSplitter(Qt.Horizontal)
        self.fig = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['类别', '样本数'])
        right_layout.addWidget(QLabel('类别统计'))
        right_layout.addWidget(self.table)
        btn_export = QPushButton('导出本步产物')
        btn_export.clicked.connect(self.on_export)
        right_layout.addWidget(btn_export)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.result_layout.addWidget(splitter)

    def on_run(self):
        out_dir = self.edit_out.text().strip()
        if not out_dir:
            return
        seed = self.spin_seed.value()
        gen = SignalGenerator(length=self.spin_length.value(),
                              sampling_rate=self.spin_fs.value(),
                              seed=None if seed < 0 else seed)
        self.run_task(gen.generate_dataset, out_dir,
                      self.spin_count.value(),
                      on_done=lambda stats: self._show_results(out_dir, stats))

    def _show_results(self, out_dir, stats):
        self.ctx.data_dir = out_dir

        # 统计表
        self.table.setRowCount(len(stats))
        for row, (cls, count) in enumerate(stats.items()):
            self.table.setItem(row, 0, QTableWidgetItem(
                f'{cls}（{CLASS_NAMES_ZH[cls]}）'))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))

        # 每类抽一条画波形
        self.fig.clear()
        axes = self.fig.subplots(4, 2)
        for ax, cls in zip(axes.flat, CLASSES):
            class_dir = os.path.join(out_dir, f'class_{cls}')
            files = sorted(f for f in os.listdir(class_dir)
                           if f.endswith('.npy')) if os.path.isdir(class_dir) else []
            if files:
                sig = np.load(os.path.join(class_dir, files[0]))
                ax.plot(sig[:2000], linewidth=0.6)
            ax.set_title(f'{cls}（{CLASS_NAMES_ZH[cls]}）', fontsize=9)
            ax.tick_params(labelsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_export(self):
        dest = self.export_to_root(
            '01_数据生成', src_dir=self.edit_out.text().strip(),
            description='时序数据集',
            params={'length': self.spin_length.value(),
                    'sampling_rate': self.spin_fs.value(),
                    'count_per_class': self.spin_count.value()})
        if dest:
            print(f'已导出到 {dest}')
