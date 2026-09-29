"""
models/registry.py - 模型注册表

通过统一的 build_model(model_type, n_classes, **config) 构建模型；
checkpoint 中保存 model_type + model_config，预测时自动重建同构模型。
"""

from .densenet import DenseNet, DEFAULT_CONFIG as DENSENET_CONFIG
from .enhanced_densenet import (EnhancedDenseNet,
                                DEFAULT_CONFIG as ENHANCED_CONFIG)

MODEL_REGISTRY = {
    'densenet': {
        'cls': DenseNet,
        'default_config': DENSENET_CONFIG,
        'display_name': 'Baseline DenseNet (20层)',
    },
    'enhanced': {
        'cls': EnhancedDenseNet,
        'default_config': ENHANCED_CONFIG,
        'display_name': 'Enhanced DenseNet + CBAM (80层)',
    },
}


def build_model(model_type, n_classes, config=None):
    """按注册表构建模型。config 缺省项用默认配置补齐。"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f'未知模型类型: {model_type}，可选: {list(MODEL_REGISTRY)}')
    entry = MODEL_REGISTRY[model_type]
    cfg = dict(entry['default_config'])
    if config:
        cfg.update(config)
    if model_type == 'densenet':
        cfg['nClasses'] = n_classes
    else:
        cfg['n_classes'] = n_classes
    return entry['cls'](**cfg), cfg


def model_config_of(model_type):
    return dict(MODEL_REGISTRY[model_type]['default_config'])
