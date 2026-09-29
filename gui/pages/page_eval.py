"""
gui/pages/page_eval.py - ⑤ 模型评估页

混淆矩阵、分类报告表、各类别准确率、错分样本浏览器。
"""

import os

import torch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QPushButton, QSplitter, QLabel, QTableWidget,
                               QTableWidgetItem, QTabWidget, QListWidget,
                               QWidget, QVBoxLayout, QHBoxLayout)

from core.dataset import create_predict_loader
from models.checkpoint import load_checkpoint, build_model_from_checkpoint
from models.registry import build_model
from training.evaluator import evaluate
from .base_page import BasePage


class EvalPage(BasePage):
    TITLE = '⑤ 模型评估'
    DESC = ('在带标签的花瓣数据集上评估模型：混淆矩阵、分类报告、'
            '各类别准确率、置信度分布与错分样本浏览。')

    def build_form(self):
        self.edit_model = self.add_file_row(
            '模型文件', self.ctx.last_checkpoint, 'PyTorch 模型 (*.pth)')
        self.edit_data = self.add_dir_row(
            '评估数据目录', self.ctx.petals_dir or self.ctx.default_path('petals'),
            'class_<类别>/ 结构的花瓣图像目录')

        # 结果区：图片标签页 + 报告表 + 错分浏览
        self.tabs = QTabWidget()
        self.img_labels = {}
        for key, title in [('confusion_matrix', '混淆矩阵'),
                           ('per_class_accuracy', '各类别准确率'),
                           ('confidence_distribution', '置信度分布')]:
            lbl = QLabel('（评估后显示）')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setScaledContents(False)
            self.tabs.addTab(lbl, title)
            self.img_labels[key] = lbl

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ['类别', '准确率', '精确率', '召回率', 'F1', '样本数'])
        self.tabs.addTab(self.table, '分类报告')

        # 错分浏览器
        wrong_widget = QWidget()
        wrong_layout = QHBoxLayout(wrong_widget)
        self.wrong_list = QListWidget()
        self.wrong_list.currentRowChanged.connect(self._show_wrong_image)
        self.wrong_image = QLabel('（选择左侧条目查看错分图像）')
        self.wrong_image.setAlignment(Qt.AlignCenter)
        self.wrong_image.setMinimumWidth(280)
        wrong_layout.addWidget(self.wrong_list, stretch=1)
        wrong_layout.addWidget(self.wrong_image, stretch=1)
        self.tabs.addTab(wrong_widget, '错分样本')

        self.result_layout.addWidget(self.tabs)

        btn_export = QPushButton('导出本步产物')
        btn_export.clicked.connect(self.on_export)
        self.result_layout.addWidget(btn_export)

        self._wrong_paths = []
        self._eval_dir = None

    # ------------------------------------------------------------
    def on_run(self):
        model_path = self.edit_model.text().strip()
        data_dir = self.edit_data.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, '提示', '请选择有效的模型文件')
            return
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, '提示', '请选择有效的评估数据目录')
            return

        self._eval_dir = os.path.join(data_dir, '_evaluation')
        self.run_task(self._evaluate_task, model_path, data_dir,
                      self._eval_dir, on_done=self._show_results)

    @staticmethod
    def _evaluate_task(model_path, data_dir, out_dir, progress_cb=None,
                       stop_event=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        info = load_checkpoint(model_path, map_location=device)
        if info['legacy']:
            raise ValueError('旧版 checkpoint 缺少结构信息，请先转换或重新训练')
        model = build_model_from_checkpoint(info, device)
        preprocess = info.get('preprocess') or {}
        loader, dataset = create_predict_loader(
            data_dir, input_size=int(preprocess.get('input_size', 64)),
            mean=tuple(preprocess.get('mean', (0.5, 0.5, 0.5))),
            std=tuple(preprocess.get('std', (0.5, 0.5, 0.5))),
            num_workers=0)
        if progress_cb:
            progress_cb(30, '模型加载完成，评估中')
        metrics = evaluate(model, loader, device, info['classes'], out_dir)
        # 把错分样本序号映射为文件路径
        for p in metrics['predictions']:
            p['path'] = dataset.get_path(p['index'])
        if progress_cb:
            progress_cb(100, '评估完成')
        return {'metrics': metrics, 'out_dir': out_dir}

    # ------------------------------------------------------------
    def _show_results(self, result):
        out_dir = result['out_dir']
        metrics = result['metrics']

        from PySide6.QtGui import QPixmap
        for key, lbl in self.img_labels.items():
            path = os.path.join(out_dir, f'{key}.png')
            if os.path.exists(path):
                pix = QPixmap(path)
                lbl.setPixmap(pix.scaled(lbl.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation))

        per = metrics['per_class']
        self.table.setRowCount(len(per) + 1)
        for row, (cls, m) in enumerate(per.items()):
            vals = [cls, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}", f"{m['f1']:.4f}", str(m['support'])]
            for col, v in enumerate(vals):
                self.table.setItem(row, col, QTableWidgetItem(v))
        last = len(per)
        for col, v in enumerate(['总体', f"{metrics['accuracy']:.4f}",
                                 f"{metrics['precision']:.4f}",
                                 f"{metrics['recall']:.4f}",
                                 f"{metrics['f1']:.4f}", '']):
            self.table.setItem(last, col, QTableWidgetItem(v))

        wrong = [p for p in metrics['predictions'] if p['true'] != p['pred']]
        self._wrong_paths = [p['path'] for p in wrong]
        self.wrong_list.clear()
        for p in wrong:
            self.wrong_list.addItem(
                f"{os.path.basename(p['path'])}: {p['true']} -> {p['pred']} "
                f"({p['confidence']:.2f})")
        print(f"评估完成：总体准确率 {metrics['accuracy']:.4f}，"
              f"错分 {len(wrong)} 个样本")

    def _show_wrong_image(self, row):
        if 0 <= row < len(self._wrong_paths):
            from PySide6.QtGui import QPixmap
            pix = QPixmap(self._wrong_paths[row])
            self.wrong_image.setPixmap(pix.scaled(
                self.wrong_image.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

    def on_export(self):
        if not self._eval_dir:
            QMessageBox.information(self, '提示', '请先运行评估')
            return
        dest = self.export_to_root('05_模型评估', src_dir=self._eval_dir,
                                   description='评估报告与图表')
        if dest:
            print(f'已导出到 {dest}')
