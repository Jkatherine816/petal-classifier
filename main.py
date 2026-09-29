"""
main.py - GUI 入口

用法：python main.py
"""

import sys
import os

# 保证从任意目录启动都能 import 项目包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # 打包为无控制台 exe 后 sys.stdout/stderr 为 None，先兜底避免 print 崩溃
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName('SDP 时序异常分类系统')

    # 深色主题（可选依赖，未安装则用默认主题）
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet('dark'))
    except ImportError:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()   # PyInstaller 打包后 DataLoader 必需
    main()
