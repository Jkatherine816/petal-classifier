"""
core/constants.py - 全局常量（单一事实源）

所有模块（信号生成、SDP 渲染、花瓣拆分、训练、预测、GUI 可视化）
都必须从这里读取类别与扇形定义，禁止在别处重复定义，避免再次漂移。

角度约定（务必统一）：
    六合一图像由 matplotlib polar 绘制，theta=0 在正东方向，逆时针增长。
    保存为 PNG 后图像坐标 y 轴向下。
    从像素坐标还原数学角度时必须使用：angle = degrees(arctan2(-dy, dx)) % 360
    其中 dx = x - cx, dy = y - cy（cx, cy 为圆心）。
"""

# ---------------------------------------------------------------- 类别体系
CLASSES = ['normal', 'outlier', 'constant', 'trend',
           'missing', 'bias', 'staircase']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# 类别中文名（GUI 展示用）
CLASS_NAMES_ZH = {
    'normal': '正常', 'outlier': '离群点', 'constant': '常数',
    'trend': '趋势', 'missing': '缺失', 'bias': '偏置', 'staircase': '阶梯',
}

# ---------------------------------------------------------------- 扇形定义
# 6 个 60° 扇形（数学角度，逆时针，0° 在正东）
SECTOR_ANGLES = [
    (0, 60),     # 扇形 0
    (60, 120),   # 扇形 1
    (120, 180),  # 扇形 2
    (180, 240),  # 扇形 3
    (240, 300),  # 扇形 4
    (300, 360),  # 扇形 5
]
SECTOR_CENTERS = [30, 90, 150, 210, 270, 330]

# 花瓣位置 -> 异常类型（模式A固定异常图：位置 i 放置第 i 种异常）
POSITION_TO_TYPE = {
    0: 'outlier', 1: 'constant', 2: 'trend',
    3: 'missing', 4: 'bias', 5: 'staircase',
}

# ---------------------------------------------------------------- 渲染参数
# 训练/预测统一单色（修复旧版"颜色捷径"：旧版训练彩色、预测灰色，
# 模型学到的是颜色而非形状）；彩色仅用于人工预览。
MONO_COLOR = '#4A4A4A'
CLASS_COLORS = {           # 仅预览/测试用，禁止用于训练与预测数据
    'normal': '#2ca02c', 'outlier': '#d62728', 'constant': '#1f77b4',
    'trend': '#ff7f0e', 'missing': '#9467bd', 'bias': '#8c564b',
    'staircase': '#e377c2',
}

DEFAULT_IMAGE_SIZE = 224        # 六合一图像边长
DEFAULT_INPUT_SIZE = 64         # 模型输入边长
DEFAULT_SEGMENT_LENGTH = 6000   # 每瓣取信号前 N 点
DEFAULT_TAU = 3                 # SDP 时间延迟
DEFAULT_DPI = 100

# 归一化参数（与旧版一致，写进 checkpoint 的 preprocess）
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# ---------------------------------------------------------------- 目录约定
CLASS_DIR_PREFIX = 'class_'     # class_normal / class_outlier / ...
METADATA_SUFFIX = '_metadata.json'
