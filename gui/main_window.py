"""
gui/main_window.py - 主窗口

左侧步骤导航 + 中央步骤页 + 底部终端框（可折叠停靠）。
会话配置（路径、导出根目录等）退出时自动保存，下次启动恢复。
"""

import os
import sys
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QListWidget, QStackedWidget,
                               QSplitter, QDockWidget, QFileDialog,
                               QMessageBox)

from core.exporter import ExportManager
from gui.widgets.terminal import TerminalWidget
from gui.pages.page_generate import GeneratePage
from gui.pages.page_render import RenderPage
from gui.pages.page_split import SplitPage
from gui.pages.page_train import TrainPage
from gui.pages.page_eval import EvalPage
from gui.pages.page_predict import PredictPage

# 打包成 exe（PyInstaller）后 __file__ 指向临时解压目录，
# 工作区/配置必须锚定到 exe 所在目录，否则退出后数据丢失。
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_PATH = os.path.join(PROJECT_ROOT, 'configs', 'last_session.json')


class AppContext:
    """应用上下文：在步骤页之间共享流水线状态"""

    def __init__(self, workspace=None):
        self.workspace = workspace or os.path.join(PROJECT_ROOT, 'workspace')
        os.makedirs(self.workspace, exist_ok=True)
        self.export_manager = ExportManager()
        # 流水线状态（前一步产物自动流入下一步）
        self.data_dir = ''
        self.image_dir = ''
        self.petals_dir = ''
        self.model_dir = ''
        self.last_checkpoint = ''

    def default_path(self, name):
        return os.path.join(self.workspace, name)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SDP 时序异常分类系统')
        self.resize(1440, 900)

        self.ctx = AppContext()

        # 左侧导航
        self.nav = QListWidget()
        self.nav.addItems(['① 数据生成', '② 图像生成', '③ 花瓣拆分',
                           '④ 模型训练', '⑤ 模型评估', '⑥ 预测'])
        self.nav.setMaximumWidth(160)
        self.nav.setCurrentRow(0)

        # 中央步骤页
        self.pages = [GeneratePage(self.ctx), RenderPage(self.ctx),
                      SplitPage(self.ctx), TrainPage(self.ctx),
                      EvalPage(self.ctx), PredictPage(self.ctx)]
        self.stack = QStackedWidget()
        for p in self.pages:
            self.stack.addWidget(p)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)

        # 底部终端框（Dock，可折叠/拖动）
        self.terminal = TerminalWidget()
        self.terminal.install_global_hooks()
        self.dock = QDockWidget('终端', self)
        self.dock.setWidget(self.terminal)
        self.dock.setFeatures(QDockWidget.DockWidgetMovable |
                              QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)

        container = QSplitter(Qt.Vertical)
        container.addWidget(splitter)
        container.setStretchFactor(0, 1)
        self.setCentralWidget(container)

        self._build_menus()
        self._load_session()

    # ------------------------------------------------------------ 菜单
    def _build_menus(self):
        bar = self.menuBar()
        m_file = bar.addMenu('文件')
        act_root = m_file.addAction('设置导出根目录…')
        act_root.triggered.connect(self._pick_export_root)
        act_ws = m_file.addAction('设置工作区…')
        act_ws.triggered.connect(self._pick_workspace)
        m_file.addSeparator()
        act_exit = m_file.addAction('退出')
        act_exit.triggered.connect(self.close)

        m_view = bar.addMenu('视图')
        act_term = m_view.addAction('显示/隐藏终端')
        act_term.triggered.connect(
            lambda: self.dock.setVisible(not self.dock.isVisible()))

    def _pick_export_root(self):
        path = QFileDialog.getExistingDirectory(self, '选择导出根目录')
        if path:
            self.ctx.export_manager.set_root(path)
            print(f'导出根目录: {path}')

    def _pick_workspace(self):
        path = QFileDialog.getExistingDirectory(self, '选择工作区')
        if path:
            self.ctx.workspace = path
            print(f'工作区: {path}')

    # ------------------------------------------------------------ 会话保存
    def _save_session(self):
        data = {
            'export_root': self.ctx.export_manager.root,
        }
        for key in ('workspace', 'data_dir', 'image_dir', 'petals_dir',
                    'model_dir', 'last_checkpoint'):
            data[key] = getattr(self.ctx, key, '')
        try:
            os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
            with open(SESSION_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _load_session(self):
        if not os.path.exists(SESSION_PATH):
            return
        try:
            with open(SESSION_PATH, encoding='utf-8') as f:
                data = json.load(f)
            for key in ('workspace', 'data_dir', 'image_dir', 'petals_dir',
                        'model_dir', 'last_checkpoint'):
                if data.get(key):
                    setattr(self.ctx, key, data[key])
            if data.get('export_root'):
                self.ctx.export_manager.set_root(data['export_root'])
        except (OSError, json.JSONDecodeError):
            pass

    def closeEvent(self, event):
        self._save_session()
        super().closeEvent(event)
