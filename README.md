# 时序异常花瓣分类系统（重构版）

基于 SDP（对称点模式）极坐标"六合一"花瓣图 + DenseNet 的时序异常分类系统。
支持 7 个类别：normal（正常）、outlier（离群）、constant（恒定）、trend（趋势）、
missing（缺失）、bias（偏置）、staircase（阶梯）。

本版在原"整合包1001"基础上完成了 **bug 修复 + 全面重构 + PySide6 GUI 化**。

---

## 一、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 GUI

```bash
python main.py
```

界面按流水线分为 6 个步骤页（左侧导航切换）：

| 步骤 | 页面 | 功能 | 可视化结果 |
|------|------|------|-----------|
| ① | 数据生成 | 生成 7 类模拟时序信号（.npy） | 7 类波形网格图 + 统计表 |
| ② | 图像生成 | 渲染六合一 SDP 花瓣图 + 元数据 | 缩略图库（双击看大图）+ 元数据查看 |
| ③ | 花瓣拆分 | 拆出 6 个扇形并对齐 / 快速路径直渲 | 选中图的 6 瓣预览（含标签）+ 类别分布图 |
| ④ | 模型训练 | DenseNet / CBAM 增强模型训练、微调、冻结 | 实时 loss/acc/lr 曲线 |
| ⑤ | 模型评估 | 在拆分花瓣目录上评估 checkpoint | 混淆矩阵 / 分类报告 / 错分样本浏览器 |
| ⑥ | 预测 | npy 文件夹或六合一图片预测 | 标注图 + 逐瓣预测表 + 汇总 |

底部内置**终端框**（显示日志与 print 输出，可清空/保存）。
每一步都有"导出本步产物"按钮，通过菜单 **文件 → 设置导出根目录** 选择任意文件夹后，
产物按 `步骤/时间戳/` 归档并记录 `manifest.json`。

### 3. 命令行（无界面批量跑）

```bash
python cli.py generate --output data/raw --count 100 --length 6000 --seed 42
python cli.py render   --input data/raw --output data/images --num 200
python cli.py split    --input data/images --output data/petals --grayscale
python cli.py split    --input data/raw --output data/petals --fast   # 快速路径（跳过六合一）
python cli.py train    --data data/petals --output models/run --model enhanced --epochs 50 --batch_size 32 --amp
python cli.py evaluate --model models/run/best_model.pth --data data/petals
python cli.py predict  --model models/run/best_model.pth --input 待测npy文件夹 --output predictions
```

---

## 二、目录结构

```
petal_classifier/
├── main.py                  # GUI 入口
├── cli.py                   # 命令行入口
├── core/                    # 核心算法层（与界面无关）
│   ├── constants.py         #   单一事实源：类别/扇形角/颜色/默认参数
│   ├── signal_generator.py  #   7 类信号生成（可复现，numpy Generator）
│   ├── sdp_renderer.py      #   SDP 极坐标渲染（模式A训练图/模式B预测图/单瓣直渲）
│   ├── petal_splitter.py    #   六合一图拆分、扇形对齐、标签恢复
│   ├── image_utils.py       #   裁剪/灰度/几何工具
│   ├── dataset.py           #   数据集、分层划分、类权重、DataLoader
│   └── exporter.py          #   分步导出管理器
├── models/                  # DenseNet / CBAM 增强 DenseNet / 注册表 / 自描述 checkpoint
├── training/                # Trainer（AMP/调度器/早停回调）+ 评估器（指标与图表）
├── inference/               # Predictor（自动匹配 checkpoint 预处理、标注、导出）
├── gui/                     # PySide6 界面：widgets / pages / workers / main_window
├── tests/                   # pytest 回归（9 项，含镜像修复回归测试）
└── configs/                 # GUI 会话自动保存
```

---

## 三、相对旧版的关键修复

1. **扇形镜像 bug（核心）**：旧版两个拆分器取扇形时位置 i 实际取到的是 5−i 的内容。
   像素角度恢复必须 `degrees(arctan2(-dy, dx)) % 360`（图像 y 轴向下），
   扇形表保持 [(0,60)…(300,360)] 不反转。本版已修复并加入像素级回归测试。
2. **颜色捷径**：旧版训练图按类别上色而预测图全灰，模型会学"颜色"而非形状。
   现训练/预测统一单色渲染（`MONO_COLOR`），彩色仅供人类预览。
3. **渲染域差**：去除了依赖图案特征的渲染增强，训练/预测域一致。
4. **快速路径锚点**：`render_single_petal` 增加圆心锚点 + 居中裁剪，
   与拆分路径图案一致（相关系数 0.7，差异仅来自抖动）。
5. **微调流程**：旧版把 Baseline DenseNet 权重灌进 EnhancedDenseNet（兼容度≈0），
   70% 随机权重还被冻结。现只允许同构加载并提示兼容度。
6. **类权重失效**：旧版 `Subset.labels` 从未生效；现数据集提供真正的 `labels` 属性。
7. **checkpoint 自描述**：保存 model_type / model_config / classes / preprocess，
   预测时自动还原输入尺寸与归一化，不再靠猜。
8. **信号生成慢**：staircase 量化由 Python 循环改为向量化。

## 四、已知边界

- 分层训练/验证划分要求样本充足（验证集样本数 ≥ 7 且每类 ≥ 2），
  小数据集请把"验证集比例"设为 0。
- AMP 仅在 CUDA 上生效，CPU 自动忽略。
- GUI 终端框会接管 stdout/stderr（界面内显示），命令行调试请看真实控制台。

## 五、测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q    # 9 项回归
```

## 六、获取 Windows EXE（云端自动打包，免部署分发）

本仓库已配置 GitHub Actions 工作流（`build-exe.yml`），每次推送到 main 分支
都会在云端 Windows 机器上自动完成：回归测试 → PyInstaller 打包 → 上传产物。

获取 exe 的步骤：

1. 打开仓库的 **Actions** 标签页，等待最新的 `Build Windows EXE` 运行完成（约 10-15 分钟）；
2. 点进该次运行，在页面底部 **Artifacts** 区下载 `SDP-PetalClassifier-windows`（zip）；
3. 解压后双击其中的 `SDP_PetalClassifier.exe` 即可运行，
   无需安装 Python 和任何依赖（绿色文件夹，整个目录可随意拷贝）。

注意事项：
- exe 版使用 CPU 推理/训练，训练大模型建议仍用 GPU 服务器 + cli.py；
- 模型文件不打进 exe，训练好的 best_model.pth 拷到目标机器后在 ⑤⑥ 页加载即可；
- 工作区（workspace/）和会话配置（configs/）会自动建在 exe 旁边；
- 产物保留 7 天，下载后本地永久可用；
- 如需本地自行打包，保留旧脚本 `打包成EXE.bat`（在 Windows cmd 中运行）。
