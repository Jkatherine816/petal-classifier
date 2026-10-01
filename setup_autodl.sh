#!/bin/bash
# ============================================================
# AutoDL 环境一键配置脚本
# 适用镜像：PyTorch 2.8.0 / Python 3.12 / CUDA 12.8
# 用法：bash setup_autodl.sh
# 说明：阿里云等国内镜像缺 opencv-python-headless，
#       本脚本逐包安装并对 opencv 自动回退到官方源。
# ============================================================
set -e
cd "$(dirname "$0")"

echo "==> 1/3 安装依赖（torch 已由镜像提供，跳过）"
pip install numpy matplotlib Pillow scikit-learn pytest

# opencv：镜像已带则跳过；否则优先官方源装 headless 版（避免 libGL 依赖）
if python -c "import cv2" 2>/dev/null; then
    echo "cv2 已存在，跳过 opencv 安装"
else
    echo "安装 opencv-python-headless（官方源）..."
    pip install opencv-python-headless -i https://pypi.org/simple || \
    pip install opencv-python -i https://pypi.org/simple
fi

echo "==> 2/3 检查 GPU 环境"
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

echo "==> 3/3 运行回归测试（11 项，应全部通过）"
python -m pytest tests/ -q

echo ""
echo "环境就绪，可以开始训练：bash train_autodl.sh"
