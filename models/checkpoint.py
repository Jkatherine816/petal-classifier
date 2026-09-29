"""
models/checkpoint.py - 自描述 checkpoint 的保存与加载

新格式 checkpoint 包含：
    model_state_dict / model_type / model_config / classes /
    preprocess {input_size, mean, std} / metrics / history / saved_at
预测器加载后自动按 preprocess 复现预处理，彻底解决
"baseline 用 0.5 归一化、微调用 ImageNet 归一化，预测时无从分辨"的旧问题。

同时兼容旧版 checkpoint（仅有 model_state_dict 的旧文件）：
    缺少的元信息按默认值补齐，并给出警告。
"""

import logging
from datetime import datetime

import torch

from core.constants import CLASSES, DEFAULT_INPUT_SIZE, NORM_MEAN, NORM_STD
from .registry import build_model

logger = logging.getLogger(__name__)


def save_checkpoint(path, model, model_type, model_config,
                    classes=None, preprocess=None, metrics=None,
                    history=None, optimizer=None, epoch=None):
    """保存自描述 checkpoint。"""
    payload = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'model_config': model_config,
        'classes': list(classes) if classes else list(CLASSES),
        'preprocess': preprocess or {
            'input_size': DEFAULT_INPUT_SIZE,
            'mean': list(NORM_MEAN), 'std': list(NORM_STD)},
        'metrics': metrics or {},
        'history': history or {},
        'epoch': epoch,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'format_version': 2,
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, path)
    logger.info('checkpoint 已保存: %s', path)
    return path


def load_checkpoint(path, map_location='cpu'):
    """
    加载 checkpoint，返回统一字典：
        {state_dict, model_type, model_config, classes, preprocess,
         metrics, history, epoch, legacy}
    """
    raw = torch.load(path, map_location=map_location, weights_only=False)

    # 旧格式兼容：直接是 state_dict 或包了一层
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
    elif isinstance(raw, dict) and 'state_dict' in raw:
        state_dict = raw['state_dict']
    else:
        state_dict = raw

    # 多 GPU 前缀清理
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v
                      for k, v in state_dict.items()}

    legacy = not (isinstance(raw, dict) and 'model_type' in raw)
    info = {
        'state_dict': state_dict,
        'model_type': raw.get('model_type') if isinstance(raw, dict) else None,
        'model_config': raw.get('model_config') if isinstance(raw, dict) else None,
        'classes': raw.get('classes', raw.get('class_names')) if isinstance(raw, dict) else None,
        'preprocess': raw.get('preprocess') if isinstance(raw, dict) else None,
        'metrics': raw.get('metrics', {}) if isinstance(raw, dict) else {},
        'history': raw.get('history', {}) if isinstance(raw, dict) else {},
        'epoch': raw.get('epoch') if isinstance(raw, dict) else None,
        'legacy': legacy,
    }
    if legacy:
        logger.warning('旧版 checkpoint（无自描述信息），按默认配置处理: %s', path)
    return info


def build_model_from_checkpoint(info, device='cpu'):
    """根据 checkpoint 信息重建模型并加载权重。旧版 checkpoint 需调用方指定 model_type。"""
    model_type = info.get('model_type')
    if model_type is None:
        raise ValueError('该 checkpoint 不含模型类型信息（旧版），'
                         '请手动指定模型结构')
    classes = info.get('classes') or list(CLASSES)
    model, _ = build_model(model_type, n_classes=len(classes),
                           config=info.get('model_config'))
    model.load_state_dict(info['state_dict'])
    model.to(device).eval()
    return model
