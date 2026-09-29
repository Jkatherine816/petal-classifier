"""
gui/widgets/curve_canvas.py - 训练实时曲线（嵌入 matplotlib）
"""

from core import mpl_setup as _mpl_setup  # noqa: F401  配置 matplotlib 中文字体
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QWidget, QVBoxLayout


class CurveCanvas(QWidget):
    """loss / accuracy / lr 三联曲线，支持训练过程中实时刷新"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 3.2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self._init_axes()

    def _init_axes(self):
        self.figure.clear()
        self.ax_loss = self.figure.add_subplot(131)
        self.ax_acc = self.figure.add_subplot(132)
        self.ax_lr = self.figure.add_subplot(133)
        for ax, title in ((self.ax_loss, '损失'), (self.ax_acc, '准确率 (%)'),
                          (self.ax_lr, '学习率')):
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=8)
        self.figure.tight_layout()

    def update_history(self, history):
        """根据 history 字典刷新曲线（训练线程经信号调用）"""
        self._init_axes()
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        if not epochs:
            self.canvas.draw_idle()
            return

        self.ax_loss.plot(epochs, history['train_loss'], label='训练')
        if any(v == v for v in history.get('val_loss', [])):  # 非 NaN
            self.ax_loss.plot(epochs, history['val_loss'], label='验证')
        self.ax_loss.legend(fontsize=8)

        self.ax_acc.plot(epochs, history['train_acc'], label='训练')
        if any(v == v for v in history.get('val_acc', [])):
            self.ax_acc.plot(epochs, history['val_acc'], label='验证')
        self.ax_acc.legend(fontsize=8)

        self.ax_lr.plot(epochs, history['learning_rate'])
        self.figure.tight_layout()
        self.canvas.draw_idle()
