from __future__ import annotations

import os
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes: int, out_planes: int, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    """
    Shortcut type A: 使用平均池化 + 零填充
    用于匹配 MedicalNet 的 ResNet-18 和 ResNet-34 预训练权重
    """
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
        dtype=out.dtype, device=out.device
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes: int = 3, in_channels: int = 2, 
                 dropout: float = 0.5, shortcut_type: str = 'B'):
        """
        Args:
            block: BasicBlock 或 BottleneckBlock
            layers: 每层的block数量列表
            num_classes: 分类类别数
            in_channels: 输入通道数（默认2，MRA+mask）
            dropout: Dropout比率
            shortcut_type: 'A' 或 'B'，默认 'B' 保持向后兼容
                - 'A': 使用平均池化+零填充（匹配 MedicalNet ResNet-18/34）
                - 'B': 使用1x1卷积+BatchNorm（默认，保持现有ResNet-18行为）
        """
        super().__init__()
        self.inplanes = 64
        self.shortcut_type = shortcut_type

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcut_type == 'A':
                # Shortcut type A: 使用平均池化+零填充（匹配 MedicalNet）
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=False
                )
            else:
                # Shortcut type B: 使用1x1卷积+BatchNorm（默认，保持兼容）
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def resnet3d10(num_classes: int = 3, in_channels: int = 2, dropout: float = 0.5) -> ResNet3D:
    # [1,1,1,1] 类似 ResNet10
    return ResNet3D(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, in_channels=in_channels, dropout=dropout)


def resnet3d18(num_classes: int = 3, in_channels: int = 2, dropout: float = 0.5) -> ResNet3D:
    # [2,2,2,2] ResNet-18
    return ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels, dropout=dropout)


def resnet3d34(num_classes: int = 3, in_channels: int = 2, dropout: float = 0.5) -> ResNet3D:
    # [3,4,6,3] ResNet-34, shortcut_type='A' 匹配 MedicalNet 预训练权重
    return ResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, 
                    in_channels=in_channels, dropout=dropout, shortcut_type='A')


def resnet3d50(num_classes: int = 3, in_channels: int = 2, dropout: float = 0.5) -> ResNet3D:
    # [3,4,6,3] ResNet-50
    return ResNet3D(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, dropout=dropout)


def load_pretrained_weights(
    model: ResNet3D,
    pretrained_path: str,
    in_channels: int = 2,
    strict: bool = False,
) -> ResNet3D:
    """
    加载MedicalNet预训练权重
    
    Args:
        model: 要加载权重的模型
        pretrained_path: 预训练权重文件路径
        in_channels: 模型输入通道数（默认2，MRA+mask）
        strict: 是否严格匹配所有层
    
    Returns:
        加载权重后的模型
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")

    print(f"[INFO] 加载预训练权重: {pretrained_path}")
    
    # 加载权重文件
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    # 处理不同的权重格式
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint
    
    # 获取模型当前状态字典
    model_dict = model.state_dict()
    
    # 处理权重键名：移除 'module.' 前缀（如果存在，来自DataParallel）
    pretrained_dict_clean = {}
    for k, v in pretrained_dict.items():
        # 移除 'module.' 前缀
        k_clean = k.replace('module.', '') if k.startswith('module.') else k
        
        # 跳过不匹配的层（如分类头、分割头等）
        if k_clean not in model_dict:
            continue
        
        # 检查形状是否匹配
        if k_clean == 'conv1.weight':
            # 处理输入通道不匹配：从1通道扩展到2通道
            if v.shape[1] == 1 and in_channels == 2:
                # 策略：将1通道权重复制到2通道
                v_expanded = v.repeat(1, in_channels, 1, 1, 1) / in_channels
                pretrained_dict_clean[k_clean] = v_expanded
                print(f"[INFO] 扩展 conv1.weight: {v.shape} -> {v_expanded.shape}")
            elif v.shape[1] == in_channels:
                pretrained_dict_clean[k_clean] = v
            else:
                print(f"[WARN] 跳过 conv1.weight: 形状不匹配 {v.shape} vs 期望 {in_channels} 通道")
                continue
        else:
            if v.shape == model_dict[k_clean].shape:
                pretrained_dict_clean[k_clean] = v
            else:
                if strict:
                    raise RuntimeError(f"权重形状不匹配: {k_clean}, {v.shape} vs {model_dict[k_clean].shape}")
                else:
                    print(f"[WARN] 跳过 {k_clean}: 形状不匹配 {v.shape} vs {model_dict[k_clean].shape}")
    
    # 更新模型权重
    model_dict.update(pretrained_dict_clean)
    model.load_state_dict(model_dict, strict=False)
    
    loaded_count = len(pretrained_dict_clean)
    total_count = len(model_dict)
    print(f"[INFO] 成功加载 {loaded_count}/{total_count} 层权重")
    
    return model
