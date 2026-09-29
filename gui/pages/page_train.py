"""
gui/pages/page_train.py - ④ 模型训练页
"""

import os

import torch
import torch.nn as nn

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                               QPushButton, QHBoxLayout, QLabel, QWidget,
                               QMessageBox)

from core.dataset import create_dataloaders
from core.constants import NORM_MEAN, NORM_STD, DEFAULT_INPUT_SIZE
from models.registry import build_model, MODEL_REGISTRY
from training.trainer import Trainer, make_optimizer, make_scheduler, \
    load_pretrained_compatible
from training.evaluator import plot_training_curves
from gui.widgets.curve_canvas import CurveCanvas
from gui.workers import TrainWorker
from .base_page import BasePage


class TrainPage(BasePage):
    TITLE = '④ 模型训练'
    DESC = ('选择 Baseline DenseNet 或 CBAM 增强模型进行训练。'
            '勾选"微调模式"可加载同构预训练 checkpoint 并按阶段冻结特征层。'
            '训练过程实时显示曲线，可随时停止。')

    def build_form(self):
        self.edit_data = self.add_dir_row(
            '训练图像目录', self.ctx.petals_dir or self.ctx.default_path('petals'),
            'class_<类别>/ 结构的单花瓣图像目录')

        self.combo_model = QComboBox()
        for key, entry in MODEL_REGISTRY.items():
            self.combo_model.addItem(entry['display_name'], userData=key)
        self.form.addRow('模型', self.combo_model)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(50)
        self.form.addRow('训练轮数', self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(32)
        self.form.addRow('批次大小', self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setValue(1e-4)
        self.spin_lr.setSingleStep(1e-5)
        self.form.addRow('学习率', self.spin_lr)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.0, 0.5)
        self.spin_val.setValue(0.2)
        self.spin_val.setSingleStep(0.05)
        self.form.addRow('验证集比例', self.spin_val)

        self.combo_aug = QComboBox()
        self.combo_aug.addItems(['none', 'light', 'medium', 'heavy'])
        self.combo_aug.setCurrentText('medium')
        self.form.addRow('数据增强', self.combo_aug)

        self.combo_sched = QComboBox()
        self.combo_sched.addItems(['plateau', 'cosine', '无'])
        self.form.addRow('学习率调度', self.combo_sched)

        self.chk_amp = QCheckBox('混合精度训练（AMP，仅 GPU）')
        self.chk_amp.setChecked(True)
        self.form.addRow('', self.chk_amp)

        # 微调选项
        self.chk_finetune = QCheckBox('微调模式（加载同构预训练权重）')
        self.chk_finetune.setChecked(False)
        self.form.addRow('', self.chk_finetune)

        self.edit_pretrained = self.add_file_row(
            '预训练模型', '', 'PyTorch 模型 (*.pth)')

        self.spin_freeze = QSpinBox()
        self.spin_freeze.setRange(0, 4)
        self.spin_freeze.setValue(2)
        self.spin_freeze.setSpecialValueText('不冻结')
        self.form.addRow('冻结特征阶段数', self.spin_freeze)

        self.edit_out = self.add_dir_row(
            '模型输出目录', self.ctx.default_path('models/run'),
            'checkpoint 与训练曲线输出目录')

        # 结果区：实时曲线 + 状态
        self.curves = CurveCanvas()
        self.result_layout.addWidget(self.curves)
        self.lbl_status = QLabel('尚未开始训练')
        self.result_layout.addWidget(self.lbl_status)

        # 底部替换：运行 + 停止 + 导出
        self.btn_stop = QPushButton('停止')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_export = QPushButton('导出本步产物')
        self.btn_export.clicked.connect(self.on_export)
        self.main_layout.itemAt(self.main_layout.count() - 1).layout() \
            .addWidget(self.btn_stop)
        self.main_layout.itemAt(self.main_layout.count() - 1).layout() \
            .addWidget(self.btn_export)

    # ------------------------------------------------------------
    def on_run(self):
        data_dir = self.edit_data.text().strip()
        out_dir = self.edit_out.text().strip()
        if not data_dir or not out_dir:
            return
        try:
            train_loader, val_loader, info = create_dataloaders(
                data_dir, batch_size=self.spin_batch.value(),
                val_split=self.spin_val.value(),
                aug_level=self.combo_aug.currentText(),
                num_workers=0)
        except Exception as exc:
            QMessageBox.critical(self, '数据加载失败', str(exc))
            return

        model_type = self.combo_model.currentData()
        model, model_cfg = build_model(model_type, info['num_classes'])

        if self.chk_finetune.isChecked():
            path = self.edit_pretrained.text().strip()
            if path and os.path.isfile(path):
                ratio = load_pretrained_compatible(model, path, 'cpu')
                print(f'预训练权重兼容度: {ratio * 100:.0f}%')
            n_freeze = self.spin_freeze.value()
            if n_freeze > 0 and hasattr(model, 'freeze_feature_stages'):
                model.freeze_feature_stages(n_freeze)
                print(f'已冻结前 {n_freeze} 个特征阶段')

        if info['class_weights'] is not None:
            criterion = nn.CrossEntropyLoss(
                weight=info['class_weights'].to(torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')))
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = make_optimizer(model, lr=self.spin_lr.value())
        sched_kind = self.combo_sched.currentText()
        scheduler = None if sched_kind == '无' else make_scheduler(
            optimizer, sched_kind, self.spin_epochs.value())

        trainer = Trainer(model, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, amp=self.chk_amp.isChecked(),
                          on_log=print)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        self.train_worker = TrainWorker(
            trainer, train_loader, val_loader, self.spin_epochs.value(),
            save_kwargs={'save_dir': out_dir, 'model_type': model_type,
                         'model_config': model_cfg,
                         'preprocess': info['preprocess'],
                         'classes': info['classes']},
            parent=self)
        self.train_worker.progress.connect(
            lambda p, m: (self.progress.setValue(p),
                          self.progress.setFormat(f'{m} %p%')))
        self.train_worker.epoch_end.connect(self._on_epoch_end)
        self.train_worker.succeeded.connect(
            lambda result: self._on_train_done(out_dir, result))
        self.train_worker.failed.connect(self._on_failed)
        self.train_worker.start()
        print(f'开始训练: {MODEL_REGISTRY[model_type]["display_name"]}, '
              f'{info["train_size"]} 训练 / {info["val_size"]} 验证')

    def _on_epoch_end(self, metrics, history):
        self.curves.update_history(history)
        self.lbl_status.setText(
            f"Epoch {metrics['epoch']} | 训练 acc {metrics['train_acc']:.2f}% | "
            f"验证 acc {metrics['val_acc']:.2f}% | lr {metrics['lr']:.2e}")

    def _on_train_done(self, out_dir, result):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(100)
        self.progress.setFormat('完成' if not result['stopped'] else '已停止')

        history = result['history']
        if history['train_loss']:
            curves_path = os.path.join(out_dir, 'training_curves.png')
            plot_training_curves(history, curves_path)
        best = result['best']
        self.lbl_status.setText(
            f"训练结束 | 最佳验证 acc {best.get('val_acc', 0):.2f}% "
            f"(epoch {best.get('epoch', '-')}) | 模型已保存到 {out_dir}")
        self.ctx.model_dir = out_dir
        self.ctx.last_checkpoint = os.path.join(out_dir, 'best_model.pth')

    def on_stop(self):
        if getattr(self, 'train_worker', None) and self.train_worker.isRunning():
            self.train_worker.stop()
            self.lbl_status.setText('正在停止（当前 epoch 结束后生效）…')

    def on_export(self):
        dest = self.export_to_root(
            '04_模型训练', src_dir=self.edit_out.text().strip(),
            description='训练产物（checkpoint/曲线/配置）',
            params={'model': self.combo_model.currentData(),
                    'epochs': self.spin_epochs.value(),
                    'lr': self.spin_lr.value(),
                    'finetune': self.chk_finetune.isChecked()})
        if dest:
            print(f'已导出到 {dest}')

    def _on_failed(self, tb):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        super()._on_failed(tb)
