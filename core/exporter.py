"""
core/exporter.py - 统一导出管理

所有步骤的产物集中导出到用户选择的输出根目录：
    <root>/<步骤名>/<时间戳>/...
每次导出更新 manifest.json（时间、参数、文件清单），保证可追溯。
"""

import os
import json
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExportManager:
    """导出管理器：管理输出根目录与各步骤的导出子目录"""

    def __init__(self, root=None):
        self.root = root
        self.manifest = {'created': datetime.now().isoformat(timespec='seconds'),
                         'exports': []}

    def set_root(self, root):
        """设置/更换输出根目录。"""
        self.root = root
        if root:
            os.makedirs(root, exist_ok=True)
        logger.info('输出根目录: %s', root)

    def require_root(self):
        if not self.root:
            raise ValueError('尚未设置输出根目录，请先在界面上选择导出文件夹')

    def step_dir(self, step, timestamp=True):
        """
        获取某步骤的导出目录，例如 <root>/03_花瓣拆分/2026-08-27_14-30-00/
        """
        self.require_root()
        name = step if not timestamp else \
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        path = os.path.join(self.root, step, name) if timestamp else \
            os.path.join(self.root, step)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------ 保存原语
    @staticmethod
    def save_json(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False,
                      default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        return path

    @staticmethod
    def save_csv(rows, path, header=None):
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        return path

    @staticmethod
    def save_figure(fig, path, dpi=150):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return path

    @staticmethod
    def copy_into(paths, dest_dir):
        """把若干已有文件复制到目标目录，返回新路径列表。"""
        os.makedirs(dest_dir, exist_ok=True)
        out = []
        for p in paths:
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))
                out.append(os.path.join(dest_dir, os.path.basename(p)))
        return out

    @staticmethod
    def copy_tree(src_dir, dest_dir):
        """整目录复制（合并），返回目标目录。"""
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        return dest_dir

    # ------------------------------------------------------------ 清单
    def record(self, step, description, files, params=None):
        """登记一次导出到 manifest 并落盘。"""
        if not self.root:
            return
        entry = {
            'time': datetime.now().isoformat(timespec='seconds'),
            'step': step,
            'description': description,
            'params': params or {},
            'files': [str(p) for p in files],
        }
        self.manifest['exports'].append(entry)
        self.save_json(self.manifest, os.path.join(self.root, 'manifest.json'))
        logger.info('导出完成[%s]: %d 个文件', step, len(entry['files']))
