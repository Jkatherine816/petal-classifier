"""
tests/test_checkpoint.py - checkpoint 与预测器回归测试
"""

import numpy as np
import torch
import pytest

from models.registry import build_model
from models.checkpoint import (save_checkpoint, load_checkpoint,
                               build_model_from_checkpoint)
from inference.predictor import Predictor


@pytest.mark.parametrize('model_type', ['densenet', 'enhanced'])
def test_checkpoint_roundtrip(model_type, tmp_path):
    model, cfg = build_model(model_type, n_classes=7)
    model.eval()  # 与加载后状态一致（BatchNorm 用 running stats）
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref_out, _ = model(x)

    path = str(tmp_path / 'm.pth')
    save_checkpoint(path, model, model_type, cfg,
                    preprocess={'input_size': 64, 'mean': [0.5] * 3,
                                'std': [0.5] * 3},
                    metrics={'best_val_acc': 99.0})

    info = load_checkpoint(path)
    assert not info['legacy']
    assert info['model_type'] == model_type
    assert info['preprocess']['input_size'] == 64

    model2 = build_model_from_checkpoint(info)
    with torch.no_grad():
        out2, _ = model2(x)
    assert torch.allclose(ref_out, out2, atol=1e-5)


def test_predictor_uses_checkpoint_preprocess(tmp_path):
    """预处理参数写入非默认值时，预测器应从 checkpoint 恢复而非用默认。"""
    model, cfg = build_model('densenet', n_classes=7)
    path = str(tmp_path / 'm.pth')
    save_checkpoint(path, model, 'densenet', cfg,
                    preprocess={'input_size': 48,
                                'mean': [0.4, 0.4, 0.4],
                                'std': [0.2, 0.2, 0.2]})
    predictor = Predictor(path, device='cpu')
    assert predictor.input_size == 48

    # 随机花瓣图像可正常预测
    arr = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    result = predictor.predict_petal(arr)
    assert result['predicted_class'] in predictor.classes
    assert 0 <= result['confidence'] <= 1
    assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-4
