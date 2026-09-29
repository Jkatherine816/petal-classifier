"""
gui/pages/base_page.py - 步骤页基类

提供：标题/说明、表单行辅助、目录选择行、"运行"后台任务封装、进度条、
状态栏消息、导出按钮辅助。所有步骤页继承它。
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QLabel, QPushButton, QProgressBar, QLineEdit,
                               QFileDialog, QGroupBox, QMessageBox)

from gui.workers import TaskWorker


class BasePage(QWidget):
    TITLE = ''
    DESC = ''

    def __init__(self, ctx, parent=None):
        """
        ctx: 应用上下文（AppContext），包含 export_manager、共享路径状态等
        """
        super().__init__(parent)
        self.ctx = ctx
        self.worker = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)

        title = QLabel(f'<h2>{self.TITLE}</h2>')
        self.main_layout.addWidget(title)
        if self.DESC:
            desc = QLabel(self.DESC)
            desc.setWordWrap(True)
            desc.setStyleSheet('color:#888;')
            self.main_layout.addWidget(desc)

        # 参数区 / 结果区由子类填充
        self.form_box = QGroupBox('参数')
        self.form = QFormLayout(self.form_box)
        self.main_layout.addWidget(self.form_box)

        self.result_box = QGroupBox('结果')
        self.result_layout = QVBoxLayout(self.result_box)
        self.main_layout.addWidget(self.result_box, stretch=1)

        # 底部：进度条 + 运行按钮
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_run = QPushButton('运行')
        self.btn_run.setMinimumWidth(120)
        self.btn_run.clicked.connect(self.on_run)
        bottom.addWidget(self.progress, stretch=1)
        bottom.addWidget(self.btn_run)
        self.main_layout.addLayout(bottom)

        self.build_form()

    # ------------------------------------------------------------ 子类接口
    def build_form(self):
        """子类实现：往 self.form 里添加参数控件"""
        pass

    def on_run(self):
        """子类实现：点击运行"""
        pass

    # ------------------------------------------------------------ 辅助
    def add_dir_row(self, label, initial='', placeholder='选择文件夹'):
        """添加"目录选择"行，返回 QLineEdit"""
        edit = QLineEdit(initial)
        edit.setPlaceholderText(placeholder)
        btn = QPushButton('浏览…')
        btn.clicked.connect(lambda: self._pick_dir(edit))
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        row.addWidget(btn)
        container = QWidget()
        container.setLayout(row)
        self.form.addRow(label, container)
        return edit

    def add_file_row(self, label, initial='', filter_str='所有文件 (*)'):
        edit = QLineEdit(initial)
        btn = QPushButton('浏览…')
        btn.clicked.connect(lambda: self._pick_file(edit, filter_str))
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        row.addWidget(btn)
        container = QWidget()
        container.setLayout(row)
        self.form.addRow(label, container)
        return edit

    def _pick_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, '选择文件夹',
                                                edit.text() or '')
        if path:
            edit.setText(path)

    def _pick_file(self, edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, '选择文件',
                                              edit.text() or '', filter_str)
        if path:
            edit.setText(path)

    # ------------------------------------------------------------ 任务封装
    def run_task(self, fn, *args, on_done=None, **kwargs):
        """后台执行任务；期间禁用运行按钮。"""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, '提示', '当前有任务正在运行')
            return
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)

        self.worker = TaskWorker(fn, *args, parent=self, **kwargs)
        self.worker.progress.connect(
            lambda p, m: (self.progress.setValue(p),
                          self.progress.setFormat(f'{m} %p%')))
        self.worker.failed.connect(self._on_failed)
        self.worker.succeeded.connect(self._on_succeeded)
        if on_done:
            self.worker.succeeded.connect(on_done)
        self.worker.start()

    def _on_succeeded(self, _result):
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        self.progress.setFormat('完成')

    def _on_failed(self, tb):
        self.btn_run.setEnabled(True)
        self.progress.setFormat('失败')
        print(f'任务失败:\n{tb}')
        QMessageBox.critical(self, '任务失败',
                             tb.strip().splitlines()[-1] if tb.strip() else '未知错误')

    def export_to_root(self, step_name, files=None, src_dir=None,
                       description='', params=None):
        """把本页产物导出到全局输出根目录，返回导出目录。"""
        em = self.ctx.export_manager
        try:
            dest = em.step_dir(step_name)
        except ValueError as exc:
            QMessageBox.warning(self, '未设置导出目录', str(exc))
            return None
        exported = []
        if files:
            exported += em.copy_into(files, dest)
        if src_dir:
            em.copy_tree(src_dir, dest)
            exported.append(dest)
        em.record(step_name, description, exported, params)
        return dest
