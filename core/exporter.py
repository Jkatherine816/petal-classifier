"""
core/exporter.py - 分步导出管理器

每一步的产物可一键导出到用户指定的根目录：
    <root>/<step>/<timestamp>/...
并在 <root>/manifest.json 追加一条导出记录。
"""

import os
import json
import shutil
import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExportManager:
    def __init__(self, root=None):
        self.root = root

    def set_root(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _require_root(self):
        if not self.root:
            raise RuntimeError('尚未设置导出根目录（菜单：文件 → 设置导出根目录）')

    def step_dir(self, step):
        """创建并返回 <root>/<step>/<timestamp>/"""
        self._require_root()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.root, step, ts)
        os.makedirs(path, exist_ok=True)
        return path

    def save_json(self, obj, dir_path, name):
        path = os.path.join(dir_path, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        return path

    def save_csv(self, rows, dir_path, name, header=None):
        """rows: list[dict] 或 list[list]；utf-8-sig 保证 Excel 打开不乱码"""
        path = os.path.join(dir_path, name)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            if rows and isinstance(rows[0], dict):
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            else:
                writer = csv.writer(f)
                if header:
                    writer.writerow(header)
                writer.writerows(rows)
        return path

    def save_figure(self, fig, dir_path, name, dpi=150):
        path = os.path.join(dir_path, name)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return path

    def copy_into(self, src, dir_path):
        """复制文件/目录进导出目录，返回目标路径"""
        dst = os.path.join(dir_path, os.path.basename(src))
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return dst

    def copy_tree(self, src_dir, dir_path, pattern=None):
        """按扩展名过滤复制目录内容"""
        copied = []
        for fn in os.listdir(src_dir):
            if pattern and not fn.endswith(pattern):
                continue
            src = os.path.join(src_dir, fn)
            if os.path.isfile(src):
                copied.append(self.copy_into(src, dir_path))
        return copied

    def record(self, step, files, description='', params=None):
        """在 <root>/manifest.json 追加导出记录"""
        self._require_root()
        manifest_path = os.path.join(self.root, 'manifest.json')
        manifest = []
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
        manifest.append({
            'step': step,
            'time': datetime.now().isoformat(timespec='seconds'),
            'description': description,
            'params': params or {},
            'files': [str(x) for x in files],
        })
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info('导出记录已写入 %s', manifest_path)
