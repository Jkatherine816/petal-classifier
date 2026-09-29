"""
core/mpl_setup.py - matplotlib 中文显示配置

导入本模块时自动执行：
1. 从系统已安装字体中挑选第一个可用的中文字体
   （Windows: Microsoft YaHei / SimHei；macOS: PingFang SC；Linux: Noto Sans CJK 等）
2. 让负号 '-' 正常显示（axes.unicode_minus=False）

任何使用 matplotlib 的模块，在 import pyplot / Figure 之前先 import 本模块即可，
否则图表中的中文会显示为方框并刷出大量 "Glyph ... missing from font" 警告。
"""

import matplotlib
from matplotlib import font_manager

# 候选中文字体（按平台常见程度排序）
_CJK_CANDIDATES = [
    'Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun',      # Windows
    'PingFang SC', 'Hiragino Sans GB', 'STHeiti',           # macOS
    'Noto Sans CJK SC', 'Noto Serif CJK SC',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',             # Linux
]


def _pick_cjk_font():
    """返回系统中第一个可用的中文字体名，找不到返回 None"""
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return None
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def setup():
    """配置 matplotlib 中文字体，返回实际选用的字体名（可能为 None）"""
    font = _pick_cjk_font()
    if font:
        matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return font


# 导入即生效
CHINESE_FONT = setup()
