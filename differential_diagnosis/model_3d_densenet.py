"""
DenseNet-121 3D模型实现
用于医学影像分类任务
密集连接网络，参数效率高，通过调整growth_rate控制参数量
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer3D(nn.Module):
    """
    DenseNet 3D DenseLayer
    
    包含BN-ReLU-Conv的密集连接层
    使用bottleneck结构（1x1x1 conv + 3x3x3 conv）以提高效率
    """
    
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return torch.cat([x, new_features], 1)


class _DenseBlock3D(nn.Module):
    """
    DenseNet 3D DenseBlock
    
    包含多个DenseLayer的密集连接块
    """
    
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, module in self.named_children():
            x = module(x)
        return x


class _Transition3D(nn.Module):
    """
    DenseNet 3D Transition Layer
    
    用于压缩特征图尺寸和通道数
    """
    
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet121_3D(nn.Module):
    """
    DenseNet-121 3D模型
    
    通过调整growth_rate使参数量接近33M（ResNet-18的水平）
    原始DenseNet-121使用growth_rate=32，参数量约8M
    调整为growth_rate=48-64可使参数量接近30-35M
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 2,
        growth_rate: int = 48,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        drop_rate: float = 0.0,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        
        # DenseBlocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition3D(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
        
        # 最终BN和ReLU
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] 输入张量
        
        Returns:
            logits: [B, num_classes] 分类logits
        """
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


def densenet121_3d(
    num_classes: int = 3,
    in_channels: int = 2,
    growth_rate: int = 56,
    dropout: float = 0.5,
) -> DenseNet121_3D:
    """
    创建DenseNet-121 3D模型
    
    Args:
        num_classes: 分类类别数
        in_channels: 输入通道数（默认2，MRA+Vessel Mask双通道）
        growth_rate: 增长率（默认56，参数量约30-31M）
        dropout: Dropout概率
    
    Returns:
        DenseNet-121 3D模型实例
    """
    return DenseNet121_3D(
        num_classes=num_classes,
        in_channels=in_channels,
        growth_rate=growth_rate,
        dropout=dropout,
    )
