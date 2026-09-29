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

if getattr(sys, 'frozen', False):
    # PyInstaller 打包后：以 exe 所在目录为根（__file__ 会指向临时解压目录）
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
        self.resize(1280, 860)

        self.ctx = AppContext()

        # 步骤导航
        self.nav = QListWidget()
        self.nav.setMaximumWidth(180)
        self.nav.addItems(['① 数据生成', '② 图像生成', '③ 花瓣拆分',
                           '④ 模型训练', '⑤ 模型评估', '⑥ 预测'])

        # 步骤页
        self.stack = QStackedWidget()
        self.pages = [GeneratePage(self.ctx), RenderPage(self.ctx),
                      SplitPage(self.ctx), TrainPage(self.ctx),
                      EvalPage(self.ctx), PredictPage(self.ctx)]
        for page in self.pages:
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # 终端框（底部停靠）
        self.terminal = TerminalWidget()
        self.terminal_dock = QDockWidget('终端', self)
        self.terminal_dock.setWidget(self.terminal)
        self.terminal_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.terminal_dock)
        self.terminal.install_global_hooks()

        # 菜单
        file_menu = self.menuBar().addMenu('文件')
        act_root = file_menu.addAction('设置导出根目录…')
        act_root.triggered.connect(self._pick_export_root)
        act_ws = file_menu.addAction('设置工作区…')
        act_ws.triggered.connect(self._pick_workspace)
        file_menu.addSeparator()
        act_quit = file_menu.addAction('退出')
        act_quit.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu('视图')
        view_menu.addAction(self.terminal_dock.toggleViewAction())

        self._load_session()
        print('系统就绪。请先在"文件"菜单中设置导出根目录。')

    # ------------------------------------------------------------ 菜单动作
    def _pick_export_root(self):
        path = QFileDialog.getExistingDirectory(self, '选择导出根目录')
        if path:
            self.ctx.export_manager.set_root(path)
            self.statusBar().showMessage(f'导出根目录: {path}')

    def _pick_workspace(self):
        path = QFileDialog.getExistingDirectory(self, '选择工作区')
        if path:
            self.ctx.workspace = path
            os.makedirs(path, exist_ok=True)
            print(f'工作区已切换: {path}')

    # ------------------------------------------------------------ 会话
    def _load_session(self):
        if not os.path.exists(SESSION_PATH):
            return
        try:
            with open(SESSION_PATH, 'r', encoding='utf-8') as f:
                session = json.load(f)
            for key in ('workspace', 'data_dir', 'image_dir', 'petals_dir',
                        'model_dir', 'last_checkpoint'):
                if session.get(key):
                    setattr(self.ctx, key, session[key])
            if session.get('export_root'):
                self.ctx.export_manager.set_root(session['export_root'])
        except Exception as exc:
            print(f'会话配置加载失败: {exc}')

    def closeEvent(self, event):
        session = {
            'workspace': self.ctx.workspace,
            'data_dir': self.ctx.data_dir,
            'image_dir': self.ctx.image_dir,
            'petals_dir': self.ctx.petals_dir,
            'model_dir': self.ctx.model_dir,
            'last_checkpoint': self.ctx.last_checkpoint,
            'export_root': self.ctx.export_manager.root,
        }
        try:
            os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
            with open(SESSION_PATH, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        super().closeEvent(event)
