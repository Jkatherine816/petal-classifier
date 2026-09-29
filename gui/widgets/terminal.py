"""
gui/widgets/terminal.py - 内置终端框

功能：
    - 显示 logging 输出与 print（stdout/stderr 重定向），线程安全（Qt 信号排队）；
    - 级别着色（DEBUG 灰 / INFO 黑 / WARNING 橙 / ERROR 红）；
    - 时间戳、自动滚动、一键清空、一键保存日志文件。
"""

import sys
import logging
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QTextCharFormat, QColor, QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPlainTextEdit, QPushButton, QFileDialog)

LEVEL_COLORS = {
    'DEBUG': '#888888',
    'INFO': '#DDDDDD',
    'WARNING': '#E0A030',
    'ERROR': '#E05050',
    'STDOUT': '#DDDDDD',
    'STDERR': '#E05050',
}


class LogEmitter(QObject):
    """跨线程日志信号载体"""
    message = Signal(str, str)   # (text, level)


class GuiLogHandler(logging.Handler):
    """把 logging 记录转发到终端框（可从任意线程调用）"""

    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        try:
            self.emitter.message.emit(record.getMessage(), record.levelname)
        except Exception:
            pass


class _StreamProxy:
    """stdout/stderr 替身：把 print 内容转发到终端框"""

    def __init__(self, emitter, level):
        self.emitter = emitter
        self.level = level
        self._buffer = ''

    def write(self, text):
        self._buffer += text
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():
                self.emitter.message.emit(line, self.level)

    def flush(self):
        if self._buffer.strip():
            self.emitter.message.emit(self._buffer, self.level)
            self._buffer = ''


class TerminalWidget(QWidget):
    """终端显示框（含工具栏）"""

    def __init__(self, parent=None, max_lines=5000):
        super().__init__(parent)
        self.max_lines = max_lines
        self.emitter = LogEmitter()
        self.emitter.message.connect(self._append)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(max_lines)
        font = QFont('Consolas')
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self.text.setFont(font)
        self.text.setStyleSheet('background:#1E1E1E; color:#DDDDDD;')

        btn_clear = QPushButton('清空')
        btn_clear.clicked.connect(self.text.clear)
        btn_save = QPushButton('保存日志')
        btn_save.clicked.connect(self._save_log)

        bar = QHBoxLayout()
        bar.addStretch(1)
        bar.addWidget(btn_clear)
        bar.addWidget(btn_save)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(bar)
        layout.addWidget(self.text)

    def _append(self, text, level='INFO'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(LEVEL_COLORS.get(level, '#DDDDDD')))
        cursor.insertText(f'[{timestamp}] [{level:7s}] {text}\n', fmt)
        self.text.verticalScrollBar().setValue(
            self.text.verticalScrollBar().maximum())

    def log(self, text, level='INFO'):
        """GUI 主线程直接写日志"""
        self.emitter.message.emit(text, level)

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, '保存日志', 'session.log', '日志文件 (*.log *.txt)')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.text.toPlainText())

    # ------------------------------------------------------------ 全局接管
    def install_global_hooks(self):
        """接管 logging 根记录器 + stdout/stderr。应用启动时调用一次即可。"""
        handler = GuiLogHandler(self.emitter)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        sys.stdout = _StreamProxy(self.emitter, 'STDOUT')
        sys.stderr = _StreamProxy(self.emitter, 'STDERR')
