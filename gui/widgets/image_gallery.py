"""
gui/widgets/image_gallery.py - 图像画廊（缩略图网格 + 点击查看大图）
"""

import os

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QWidget, QListWidget, QListWidgetItem,
                               QVBoxLayout, QDialog, QLabel, QScrollArea)


class ImageGallery(QWidget):
    """图像缩略图画廊；双击查看大图"""

    def __init__(self, parent=None, thumb_size=140):
        super().__init__(parent)
        self.thumb_size = thumb_size
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(thumb_size, thumb_size))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setSpacing(8)
        self.list.itemDoubleClicked.connect(self._show_full)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)

    def set_images(self, paths, captions=None):
        """设置图像列表。captions 可选，与 paths 等长。"""
        self.list.clear()
        self._paths = list(paths)
        for i, path in enumerate(self._paths):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            item = QListWidgetItem(pixmap.scaled(
                self.thumb_size, self.thumb_size,
                Qt.KeepAspectRatio, Qt.SmoothTransformation), '')
            caption = (captions[i] if captions else os.path.basename(path))
            item.setText(caption)
            item.setTextAlignment(Qt.AlignHCenter)
            item.setSizeHint(QSize(self.thumb_size + 16, self.thumb_size + 34))
            self.list.addItem(item)

    def _show_full(self, item):
        idx = self.list.row(item)
        if not (0 <= idx < len(self._paths)):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(self._paths[idx]))
        label = QLabel()
        pixmap = QPixmap(self._paths[idx])
        label.setPixmap(pixmap)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        lay = QVBoxLayout(dlg)
        lay.addWidget(scroll)
        dlg.resize(min(720, pixmap.width() + 40),
                   min(720, pixmap.height() + 40))
        dlg.exec()
