#!/bin/bash
# ============================================================
# AutoDL 训练一条龙：数据生成 -> 域随机化渲染 -> 训练
# 用法：bash train_autodl.sh [每类样本数] [训练轮数] [批次大小]
#   默认：bash train_autodl.sh 2000 100 128
# ============================================================
set -e
cd "$(dirname "$0")"

COUNT=${1:-2000}    # 每类样本数（7 类，2000 -> 共 14000 条信号/花瓣）
EPOCHS=${2:-100}
BATCH=${3:-128}

if [ -d data/raw ] && [ "$(find data/raw -name '*.npy' 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "==> 1/3 检测到 data/raw 已有数据，跳过生成（删除该目录可重新生成）"
else
    echo "==> 1/3 生成时序信号（每类 ${COUNT} 条）"
    python cli.py generate --output data/raw --count "${COUNT}" --seed 42
fi

echo "==> 2/3 快速路径渲染单花瓣（域随机化默认开启，7 进程并行）"
python cli.py split --fast --input data/raw --output data/petals --render_workers 7

echo "==> 3/3 训练（Enhanced DenseNet + CBAM，输入 96，标签平滑 0.1）"
python cli.py train --data data/petals --output models/run --model enhanced \
    --epochs "${EPOCHS}" --batch_size "${BATCH}" --learning_rate 1e-4 \
    --input_size 96 --label_smoothing 0.1 --augmentation medium \
    --scheduler plateau --amp --workers 12

echo ""
echo "训练完成！最佳模型：models/run/best_model.pth"
echo "下载该文件后，在 exe 的 ⑤评估 / ⑥预测 页直接加载即可（渲染参数已随 checkpoint 自描述）。"
