"""
gui/pages/page_predict.py - ⑥ 预测页

输入：npy 数据文件夹（自动渲染模式B六合一图，>6 个文件自动分组）
     或单张六合一图像。
输出：标注图 + 结果明细表 + json/csv 导出。
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QPushButton, QSplitter, QLabel,
                               QTableWidget, QTableWidgetItem, QWidget,
                               QVBoxLayout, QMessageBox)

from inference.predictor import Predictor
from .base_page import BasePage


class PredictPage(BasePage):
    TITLE = '⑥ 预测'
    DESC = ('加载训练好的模型，对 npy 数据文件夹（模式B）或六合一图像进行预测。'
            '结果以标注图 + 明细表展示，可导出 json/csv。')

    def build_form(self):
        self.edit_model = self.add_file_row(
            '模型文件', self.ctx.last_checkpoint, 'PyTorch 模型 (*.pth)')

        self.combo_input = QComboBox()
        self.combo_input.addItems(['npy 数据文件夹', '六合一图像文件'])
        self.form.addRow('输入类型', self.combo_input)

        self.edit_input_dir = self.add_dir_row('数据文件夹', '',
                                               '包含 .npy 文件的目录')
        self.edit_input_img = self.add_file_row('图像文件', '',
                                                '图像 (*.png *.jpg *.jpeg)')
        self.edit_input_img.parentWidget().setVisible(False)
        self.combo_input.currentIndexChanged.connect(self._on_input_type_change)

        # 结果区：标注图 + 明细表
        splitter = QSplitter(Qt.Horizontal)
        self.lbl_image = QLabel('（预测后显示标注图）')
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(320, 320)
        splitter.addWidget(self.lbl_image)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['扇形', '预测类别', '置信度', '来源文件'])
        right_layout.addWidget(self.table)
        self.lbl_summary = QLabel('')
        right_layout.addWidget(self.lbl_summary)
        btn_export = QPushButton('导出预测结果')
        btn_export.clicked.connect(self.on_export)
        right_layout.addWidget(btn_export)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.result_layout.addWidget(splitter)

        self._payload = None
        self._annotated_paths = []

    # ------------------------------------------------------------
    def _on_input_type_change(self, idx):
        self.edit_input_dir.parentWidget().setVisible(idx == 0)
        self.edit_input_img.parentWidget().setVisible(idx == 1)

    def on_run(self):
        model_path = self.edit_model.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, '提示', '请选择有效的模型文件')
            return

        if self.combo_input.currentIndex() == 0:
            folder = self.edit_input_dir.text().strip()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, '提示', '请选择有效的数据文件夹')
                return
            self.run_task(self._predict_folder_task, model_path, folder,
                          on_done=self._show_results)
        else:
            image = self.edit_input_img.text().strip()
            if not image or not os.path.isfile(image):
                QMessageBox.warning(self, '提示', '请选择有效的图像文件')
                return
            self.run_task(self._predict_image_task, model_path, image,
                          on_done=self._show_results)

    @staticmethod
    def _predict_folder_task(model_path, folder, progress_cb=None,
                             stop_event=None):
        predictor = Predictor(model_path)
        work_dir = os.path.join(folder, '_predict_images')
        payload = predictor.predict_folder(folder, work_dir, progress_cb)
        annotated = []
        for i, img in enumerate(payload['images']):
            out = os.path.join(work_dir, f'annotated_{i:03d}.png')
            predictor.annotate_image(img['image_path'], img['results'], out)
            annotated.append(out)
        return {'payload': payload, 'annotated': annotated}

    @staticmethod
    def _predict_image_task(model_path, image_path, progress_cb=None,
                            stop_event=None):
        predictor = Predictor(model_path)
        results = predictor.predict_image(image_path)
        out = image_path.replace('.png', '_annotated.png')
        predictor.annotate_image(image_path, results, out)
        payload = {'images': [{'image_path': image_path, 'results': results}],
                   'summary': predictor.summarize(
                       [{'image_path': image_path, 'results': results}])}
        return {'payload': payload, 'annotated': [out]}

    # ------------------------------------------------------------
    def _show_results(self, result):
        self._payload = result['payload']
        self._annotated_paths = result['annotated']

        if self._annotated_paths:
            pix = QPixmap(self._annotated_paths[0])
            self.lbl_image.setPixmap(pix.scaled(
                self.lbl_image.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

        rows = []
        for img in self._payload['images']:
            rows.extend(img['results'])
        self.table.setRowCount(len(rows))
        for row, r in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(str(r['position'])))
            self.table.setItem(row, 1, QTableWidgetItem(r['predicted_class']))
            self.table.setItem(row, 2,
                               QTableWidgetItem(f"{r['confidence']:.4f}"))
            self.table.setItem(row, 3,
                               QTableWidgetItem(r.get('source_file') or ''))

        summary = self._payload.get('summary', {})
        self.lbl_summary.setText(
            f"共 {summary.get('total_petals', 0)} 个花瓣 | 类别分布: "
            f"{summary.get('class_distribution', {})}")

    def on_export(self):
        if not self._payload:
            QMessageBox.information(self, '提示', '请先运行预测')
            return
        em = self.ctx.export_manager
        try:
            dest = em.step_dir('06_预测')
        except ValueError as exc:
            QMessageBox.warning(self, '未设置导出目录', str(exc))
            return
        json_path, csv_path = Predictor.export_results(self._payload, dest)
        exported = [json_path, csv_path]
        exported += em.copy_into(self._annotated_paths, dest)
        em.record('06_预测', '预测结果与标注图', exported)
        print(f'已导出到 {dest}')
