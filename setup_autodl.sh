#!/bin/bash
# ============================================================
# AutoDL 环境一键配置脚本
# 适用镜像：PyTorch 2.8.0 / Python 3.12 / CUDA 12.8
# 用法：bash setup_autodl.sh
# 说明：AutoDL 直连 pypi.org 极慢（~20KB/s），阿里云镜像
#       部分包缺失，统一使用清华 tuna 镜像（国内满速）。
# ============================================================
set -e
cd "$(dirname "$0")"

MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> 1/3 安装依赖（清华镜像；torch 已由镜像提供，跳过）"
pip install numpy matplotlib Pillow scikit-learn pytest \
            opencv-python-headless -i "${MIRROR}"

echo "==> 2/3 检查 GPU 环境"
python -c "import cv2, torch; print('PyTorch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无', '| cv2', cv2.__version__)"

echo "==> 3/3 运行回归测试（11 项，应全部通过）"
python -m pytest tests/ -q

echo ""
echo "环境就绪，可以开始训练：bash train_autodl.sh"
