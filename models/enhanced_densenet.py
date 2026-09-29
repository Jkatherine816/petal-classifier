"""
models/enhanced_densenet.py - CBAM 增强 DenseNet（微调模型）

相对旧版 fine_tuning_model.py 的修复：
    - 深度计算整理（4 个 Dense Block）；
    - 提供 DEFAULT_CONFIG，配置随 checkpoint 保存，预测时自动重建同构模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        return self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, use_attention=True,
                 dropout=0.2):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3,
                               padding=1, bias=False)
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(growth_rate)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.use_attention:
            out = self.attention(out)
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class EnhancedTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class EnhancedDenseNet(nn.Module):
    """带 CBAM 注意力的 DenseNet（4 个 Dense Block）"""

    def __init__(self, growth_rate=24, depth=80, reduction=0.5,
                 n_classes=7, bottleneck=True, use_attention=True):
        super().__init__()
        n_blocks = (depth - 4) // 6
        if bottleneck:
            n_blocks //= 2

        n_channels = 2 * growth_rate
        self.conv0 = nn.Conv2d(3, n_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(n_channels)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks = []
        for i in range(4):
            blocks.append(self._make_dense_block(
                n_blocks, n_channels, growth_rate, bottleneck, use_attention))
            n_channels += growth_rate * n_blocks
            if i < 3:
                out_ch = max(1, int(n_channels * reduction))
                blocks.append(EnhancedTransition(n_channels, out_ch))
                n_channels = out_ch
        self.features = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm2d(n_channels)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n_classes))

        self._initialize_weights()

    @staticmethod
    def _make_dense_block(n_layers, in_channels, growth_rate, bottleneck,
                          use_attention):
        layers = []
        for i in range(n_layers):
            layers.append(BottleneckBlock(in_channels + i * growth_rate,
                                          growth_rate, use_attention))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.features(x)
        x = F.relu(self.final_bn(x))
        features = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(features), features

    # 微调辅助：按模块冻结（替代旧版按"参数张量数量比例"的粗糙冻结）
    def freeze_feature_stages(self, n_stages):
        """
        冻结前 n_stages 个阶段（0=不冻结，4=冻结全部特征层）。
        阶段顺序：conv0/bn0 -> features[0] -> features[1] -> features[2] -> features[3]
        """
        if n_stages <= 0:
            for p in self.parameters():
                p.requires_grad = True
            return
        for p in self.conv0.parameters():
            p.requires_grad = False
        for p in self.bn0.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.features):
            requires = i >= n_stages
            for p in block.parameters():
                p.requires_grad = requires


DEFAULT_CONFIG = {'growth_rate': 24, 'depth': 80, 'reduction': 0.5,
                  'bottleneck': True, 'use_attention': True}
