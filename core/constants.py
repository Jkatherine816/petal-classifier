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
NUM_SECTORS = 6

# 模式A固定位置：扇形位置 -> 异常类型
POSITION_TO_TYPE = {
    0: 'outlier',
    1: 'constant',
    2: 'trend',
    3: 'missing',
    4: 'bias',
    5: 'staircase',
}
TYPE_TO_POSITION = {v: k for k, v in POSITION_TO_TYPE.items()}

# ---------------------------------------------------------------- 颜色（仅人工预览用）
# 注意：训练管线的图像一律单色渲染（MONO_COLOR），彩色只允许用于人工预览，
# 绝不允许进入训练/预测数据流，否则模型会学到"颜色捷径"。
CLASS_COLORS = {
    'outlier': '#FF6B6B',
    'constant': '#4ECDC4',
    'trend': '#45B7D1',
    'missing': '#96CEB4',
    'bias': '#FFEAA7',
    'staircase': '#DDA0DD',
    'normal': '#CCCCCC',
}
MONO_COLOR = '#4A4A4A'  # 训练/预测统一单色（深灰）

# ---------------------------------------------------------------- 默认参数
DEFAULT_SIGNAL_LENGTH = 50000       # 信号长度 L
DEFAULT_SAMPLING_RATE = 20          # 采样频率 Fs
DEFAULT_SEGMENT_LENGTH = 6000       # 渲染截取的数据段长度
DEFAULT_TAU = 3                     # SDP 时间延迟
DEFAULT_IMAGE_SIZE = 224            # 六合一图像尺寸
DEFAULT_DPI = 100
DEFAULT_PETAL_SIZE = 64             # 拆分花瓣输出尺寸（模型输入尺寸）
DEFAULT_INPUT_SIZE = 64

# 归一化参数（统一约定，写入 checkpoint 随模型走）
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# ---------------------------------------------------------------- 目录约定
CLASS_DIR_PREFIX = 'class_'         # 类别目录前缀：class_normal / ...
METADATA_SUFFIX = '_metadata.json'  # 六合一图像元数据后缀
