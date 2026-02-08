"""
Grad-CAM工具模块 - 全脑版本（鉴别诊断）

适用于全脑图像（224x224x224），支持热力图生成和逆变换到原图空间。

完整预处理流程（与训练一致）:
1. 加载原始MRA和Mask
2. BBox裁剪（基于Mask，margin=10）
3. 0.75mm等向性重采样
4. 中心Pad/Crop到224x224x224
5. Z-score归一化（非零像素）

数据来源:
- MRA: {sample_id}_0000_brain_0000.nii.gz
- Mask: {sample_id}_0000_brain.nii.gz
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional, List

# 设置matplotlib后端为Agg（非交互式，支持多线程）
import matplotlib
matplotlib.use('Agg')

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 预处理常量（与训练一致）
# =============================================================================
TARGET_SPACING = 0.75  # mm，等向性spacing
TARGET_SHAPE = (224, 224, 224)  # (Z, Y, X)
BBOX_MARGIN = 10  # bbox裁剪的margin（体素数）

# 类别名称映射
LABEL_NAMES = {0: "MMD", 1: "ICAS", 2: "NC"}


# =============================================================================
# 辅助函数
# =============================================================================

def _to_python_types(obj):
    """将numpy类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python_types(v) for v in obj]
    else:
        return obj


def get_bbox_3d(mask_array: np.ndarray, margin_voxels: int = 10) -> Optional[Tuple[int, ...]]:
    """
    计算mask的3D边界框，并添加margin
    
    Args:
        mask_array: 3D mask数组
        margin_voxels: margin大小（体素数）
    
    Returns:
        (zmin, zmax, ymin, ymax, xmin, xmax) 或 None（如果mask为空）
    """
    coords = np.argwhere(mask_array > 0)
    if coords.shape[0] == 0:
        return None
    
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    
    # 添加margin
    zmin = max(0, zmin - margin_voxels)
    ymin = max(0, ymin - margin_voxels)
    xmin = max(0, xmin - margin_voxels)
    zmax = min(mask_array.shape[0] - 1, zmax + margin_voxels)
    ymax = min(mask_array.shape[1] - 1, ymax + margin_voxels)
    xmax = min(mask_array.shape[2] - 1, xmax + margin_voxels)
    
    return (int(zmin), int(zmax + 1), int(ymin), int(ymax + 1), int(xmin), int(xmax + 1))


def crop_by_bbox(image_array: np.ndarray, bbox: Tuple[int, ...]) -> np.ndarray:
    """根据bbox裁剪图像"""
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    return image_array[zmin:zmax, ymin:ymax, xmin:xmax].copy()


def resample_to_isotropic(
    image_array: np.ndarray, 
    original_spacing: Tuple[float, float, float], 
    target_spacing: float, 
    order: int = 3
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    将图像重采样到等向性spacing
    """
    zoom_factors = np.array(original_spacing) / target_spacing
    resampled = zoom(image_array, zoom_factors, order=order, mode='constant', cval=0.0)
    return resampled, (target_spacing, target_spacing, target_spacing)


def pad_or_crop_to_shape(
    image_array: np.ndarray, 
    target_shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    通过中心pad或crop将图像调整到目标尺寸，并记录变换参数
    """
    current_shape = np.array(image_array.shape)
    target_shape_arr = np.array(target_shape)
    
    diff = target_shape_arr - current_shape
    
    # 分别计算每个维度的pad和crop
    pad_before = np.zeros(3, dtype=np.int64)
    pad_after = np.zeros(3, dtype=np.int64)
    crop_before = np.zeros(3, dtype=np.int64)
    crop_after = np.zeros(3, dtype=np.int64)
    
    for i in range(3):
        if diff[i] > 0:
            pad_before[i] = diff[i] // 2
            pad_after[i] = diff[i] - pad_before[i]
        elif diff[i] < 0:
            crop_total = -diff[i]
            crop_before[i] = crop_total // 2
            crop_after[i] = crop_total - crop_before[i]
    
    transform_info = {
        "shape_before_padcrop": [int(x) for x in current_shape],
        "pad_before": [int(x) for x in pad_before],
        "pad_after": [int(x) for x in pad_after],
        "crop_before": [int(x) for x in crop_before],
        "crop_after": [int(x) for x in crop_after],
    }
    
    # 先crop
    if np.any(crop_before > 0) or np.any(crop_after > 0):
        slices = tuple(
            slice(int(cb), int(cs - ca) if ca > 0 else None) 
            for cb, ca, cs in zip(crop_before, crop_after, current_shape)
        )
        image_array = image_array[slices]
    
    # 再pad
    if np.any(pad_before > 0) or np.any(pad_after > 0):
        pad_width = tuple((int(pb), int(pa)) for pb, pa in zip(pad_before, pad_after))
        image_array = np.pad(image_array, pad_width, mode='constant', constant_values=0)
    
    return image_array, transform_info


def normalize_mra_nonzero(mra: np.ndarray) -> np.ndarray:
    """非零像素Z-score归一化（与训练一致）"""
    non_zero = mra[mra > 0]
    if non_zero.size >= 32:
        mean = float(non_zero.mean())
        std = float(non_zero.std() + 1e-6)
        mra = (mra - mean) / std
    else:
        mean = float(mra.mean())
        std = float(mra.std() + 1e-6)
        mra = (mra - mean) / std
    return mra.astype(np.float32)


# =============================================================================
# 带变换参数记录的预处理（全脑版本）- 完整流程
# =============================================================================

def preprocess_fullbrain_with_transform_record(
    mra_path: str,
    mask_path: str,
    target_spacing: float = TARGET_SPACING,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    bbox_margin: int = BBOX_MARGIN,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    执行全脑预处理并记录所有变换参数，用于后续逆变换
    
    完整预处理流程（与训练一致）:
    1. 加载原始NIfTI
    2. BBox裁剪（基于Mask，margin=10）
    3. 0.75mm等向性重采样
    4. 中心Pad/Crop到224x224x224
    5. Z-score归一化
    
    Args:
        mra_path: 原始MRA文件路径 (*_0000_brain_0000.nii.gz)
        mask_path: 分割Mask文件路径 (*_0000_brain.nii.gz)
        target_spacing: 目标spacing (mm)
        target_shape: 目标形状
        bbox_margin: BBox裁剪margin（体素数）
    
    Returns:
        preprocessed_input: [2, D, H, W] 模型输入
        transform_params: dict 包含所有逆变换所需参数
    """
    transform_params = {
        "mra_path": mra_path,
        "mask_path": mask_path,
        "target_spacing": float(target_spacing),
        "target_shape": list(target_shape),
        "bbox_margin": int(bbox_margin),
    }
    
    # 1. 加载原始数据
    mra_nii = nib.load(mra_path)
    mask_nii = nib.load(mask_path)
    
    mra_data = mra_nii.get_fdata(dtype=np.float32)
    mask_data = mask_nii.get_fdata(dtype=np.float32)
    mask_data = (mask_data > 0.5).astype(np.uint8)
    
    # 获取原始信息 (nibabel返回的是XYZ顺序)
    original_spacing_xyz = np.array(mra_nii.header.get_zooms()[:3])
    original_affine = mra_nii.affine
    
    transform_params["original_shape_xyz"] = [int(x) for x in mra_data.shape]
    transform_params["original_affine"] = [[float(x) for x in row] for row in original_affine]
    transform_params["original_spacing_xyz"] = [float(x) for x in original_spacing_xyz]
    
    # 2. 计算BBox并裁剪（在原始XYZ空间进行）
    bbox = get_bbox_3d(mask_data, bbox_margin)
    if bbox is None:
        raise ValueError(f"Mask为空，无法计算BBox: {mask_path}")
    
    transform_params["bbox_xyz"] = list(bbox)
    
    mra_cropped = crop_by_bbox(mra_data, bbox)
    mask_cropped = crop_by_bbox(mask_data, bbox)
    
    transform_params["cropped_shape_xyz"] = [int(x) for x in mra_cropped.shape]
    
    # 3. 转换为ZYX顺序（与训练脚本一致）
    # 注意：spacing也要转换顺序
    original_spacing_zyx = np.array([original_spacing_xyz[2], original_spacing_xyz[1], original_spacing_xyz[0]])
    mra_zyx = np.transpose(mra_cropped, (2, 1, 0))  # (X, Y, Z) -> (Z, Y, X)
    mask_zyx = np.transpose(mask_cropped, (2, 1, 0))
    
    transform_params["cropped_shape_zyx"] = [int(x) for x in mra_zyx.shape]
    transform_params["cropped_spacing_zyx"] = [float(x) for x in original_spacing_zyx]
    
    # 4. 等向性重采样
    mra_resampled, new_spacing = resample_to_isotropic(
        mra_zyx, original_spacing_zyx, target_spacing, order=3
    )
    mask_resampled, _ = resample_to_isotropic(
        mask_zyx, original_spacing_zyx, target_spacing, order=0
    )
    mask_resampled = (mask_resampled > 0.5).astype(np.uint8)
    
    transform_params["resampled_shape_zyx"] = [int(x) for x in mra_resampled.shape]
    transform_params["resampled_spacing_zyx"] = [float(x) for x in new_spacing]
    
    # 5. Pad/Crop到目标尺寸
    mra_final, padcrop_info = pad_or_crop_to_shape(mra_resampled, target_shape)
    mask_final, _ = pad_or_crop_to_shape(mask_resampled, target_shape)
    mask_final = (mask_final > 0.5).astype(np.uint8)
    
    transform_params.update(padcrop_info)
    
    # 6. 归一化
    mra_final = normalize_mra_nonzero(mra_final)
    mask_final = mask_final.astype(np.float32)
    
    # 7. 堆叠为双通道
    preprocessed_input = np.stack([mra_final, mask_final], axis=0).astype(np.float32)
    
    # 转换所有numpy类型为Python类型
    transform_params = _to_python_types(transform_params)
    
    return preprocessed_input, transform_params


# =============================================================================
# Grad-CAM 3D 实现
# =============================================================================

class GradCAM3D:
    """3D Grad-CAM实现"""
    
    def __init__(self, model: nn.Module, target_layer_name: str = 'features.denseblock4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _get_target_layer(self) -> nn.Module:
        parts = self.target_layer_name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _register_hooks(self):
        target_layer = self._get_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        upsample_to_input: bool = True,
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        if upsample_to_input:
            input_shape = input_tensor.shape[2:]
            zoom_factors = [t / c for t, c in zip(input_shape, cam.shape)]
            cam = zoom(cam, zoom_factors, order=3)  # 三次样条插值，减少横线伪影
        
        return cam


def generate_ensemble_gradcam(
    models: List[nn.Module],
    input_tensor: torch.Tensor,
    target_layer_name: str = 'features.denseblock4',
    target_class: Optional[int] = None,
    device: torch.device = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """使用多个模型生成集成Grad-CAM热力图"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cams = []
    probs_sum = None
    
    for model in models:
        model = model.to(device)
        model.eval()
        
        input_tensor_dev = input_tensor.to(device)
        
        with torch.no_grad():
            logits = model(input_tensor_dev)
            probs = F.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = probs.cpu().numpy()
            else:
                probs_sum += probs.cpu().numpy()
        
        gradcam = GradCAM3D(model, target_layer_name)
        cam = gradcam.generate(input_tensor_dev, target_class=target_class)
        cams.append(cam)
    
    avg_probs = probs_sum / len(models)
    predicted_class = int(avg_probs.argmax())
    
    ensemble_cam = np.mean(cams, axis=0)
    
    if ensemble_cam.max() > ensemble_cam.min():
        ensemble_cam = (ensemble_cam - ensemble_cam.min()) / (ensemble_cam.max() - ensemble_cam.min())
    
    return ensemble_cam, predicted_class, avg_probs.squeeze()


# =============================================================================
# 逆变换函数 - 完整流程
# =============================================================================

def inverse_transform_heatmap_to_original(
    heatmap: np.ndarray,
    transform_params: Dict[str, Any],
) -> np.ndarray:
    """
    将热力图从预处理空间逆变换到原图空间
    
    逆变换流程:
    1. 逆Pad/Crop (224x224x224 -> resampled_shape_zyx)
    2. 逆重采样 (resampled_shape_zyx -> cropped_shape_zyx)
    3. 逆BBox裁剪 (cropped_shape_zyx -> original_shape_zyx)
    
    Args:
        heatmap: [D, H, W] 预处理空间中的热力图（224x224x224，ZYX顺序）
        transform_params: 预处理时记录的变换参数
    
    Returns:
        heatmap_original: 原图空间中的热力图（ZYX顺序）
    """
    # ===== 步骤1: 逆Pad/Crop =====
    # 逆Crop（添加被裁掉的部分）
    crop_before = transform_params.get("crop_before", [0, 0, 0])
    crop_after = transform_params.get("crop_after", [0, 0, 0])
    
    if any(c > 0 for c in crop_before) or any(c > 0 for c in crop_after):
        pad_width = tuple((cb, ca) for cb, ca in zip(crop_before, crop_after))
        heatmap = np.pad(heatmap, pad_width, mode='constant', constant_values=0)
    
    # 逆Pad（去除被填充的部分）
    pad_before = transform_params.get("pad_before", [0, 0, 0])
    pad_after = transform_params.get("pad_after", [0, 0, 0])
    
    if any(p > 0 for p in pad_before) or any(p > 0 for p in pad_after):
        z, y, x = heatmap.shape
        heatmap = heatmap[
            pad_before[0]:(z - pad_after[0]) if pad_after[0] > 0 else z,
            pad_before[1]:(y - pad_after[1]) if pad_after[1] > 0 else y,
            pad_before[2]:(x - pad_after[2]) if pad_after[2] > 0 else x,
        ]
    
    # 此时应该回到 resampled_shape_zyx
    
    # ===== 步骤2: 逆重采样 =====
    cropped_shape_zyx = transform_params.get("cropped_shape_zyx")
    if cropped_shape_zyx is not None:
        zoom_factors = [o / h for o, h in zip(cropped_shape_zyx, heatmap.shape)]
        heatmap = zoom(heatmap, zoom_factors, order=3)  # 三次样条插值，减少横线伪影
        
        # 确保形状完全匹配
        if list(heatmap.shape) != list(cropped_shape_zyx):
            result = np.zeros(cropped_shape_zyx, dtype=np.float32)
            min_shape = [min(a, b) for a, b in zip(heatmap.shape, cropped_shape_zyx)]
            result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                heatmap[:min_shape[0], :min_shape[1], :min_shape[2]]
            heatmap = result
    
    # 此时应该回到 cropped_shape_zyx（裁剪后的形状）
    
    # ===== 步骤3: 逆BBox裁剪 =====
    # 将热力图放回原始图像的正确位置
    original_shape_xyz = transform_params.get("original_shape_xyz")
    bbox_xyz = transform_params.get("bbox_xyz")
    
    if original_shape_xyz is not None and bbox_xyz is not None:
        # 原始形状（XYZ顺序），转换为ZYX
        original_shape_zyx = [original_shape_xyz[2], original_shape_xyz[1], original_shape_xyz[0]]
        
        # BBox（XYZ顺序）: [xmin, xmax, ymin, ymax, zmin, zmax]
        # 需要转换为ZYX顺序
        xmin, xmax, ymin, ymax, zmin, zmax = bbox_xyz
        bbox_zyx = (zmin, zmax, ymin, ymax, xmin, xmax)
        
        # 创建原始大小的数组
        heatmap_original = np.zeros(original_shape_zyx, dtype=np.float32)
        
        # 当前热力图是ZYX顺序，bbox_zyx也是ZYX顺序
        z_start, z_end = bbox_zyx[0], bbox_zyx[1]
        y_start, y_end = bbox_zyx[2], bbox_zyx[3]
        x_start, x_end = bbox_zyx[4], bbox_zyx[5]
        
        # 计算实际可放入的范围
        h_z, h_y, h_x = heatmap.shape
        actual_z = min(h_z, z_end - z_start)
        actual_y = min(h_y, y_end - y_start)
        actual_x = min(h_x, x_end - x_start)
        
        heatmap_original[z_start:z_start+actual_z, 
                        y_start:y_start+actual_y, 
                        x_start:x_start+actual_x] = heatmap[:actual_z, :actual_y, :actual_x]
    else:
        heatmap_original = heatmap
    
    return heatmap_original


def get_original_mra_data(
    mra_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取原图MRA数据（ZYX顺序）
    
    Args:
        mra_path: 原始MRA文件路径
    
    Returns:
        mra_data: 原图MRA数据（ZYX顺序）
        affine: 原图affine矩阵
    """
    mra_nii = nib.load(mra_path)
    mra_data = mra_nii.get_fdata(dtype=np.float32)
    
    # 转换为ZYX顺序
    mra_zyx = np.transpose(mra_data, (2, 1, 0))
    
    return mra_zyx, mra_nii.affine


# =============================================================================
# 保存函数
# =============================================================================

def save_heatmap_as_nifti(
    heatmap: np.ndarray,
    affine: np.ndarray,
    output_path: str,
):
    """保存热力图为NIfTI文件（ZYX输入，转换为XYZ保存）"""
    heatmap_xyz = np.transpose(heatmap, (2, 1, 0))
    nii = nib.Nifti1Image(heatmap_xyz.astype(np.float32), affine=affine)
    nib.save(nii, output_path)
    print(f"[INFO] 热力图已保存: {output_path}")


def save_heatmap_as_nifti_uint8(
    heatmap: np.ndarray,
    affine: np.ndarray,
    output_path: str,
):
    """保存热力图为uint8格式NIfTI文件（0-255）"""
    heatmap_clipped = np.clip(heatmap, 0, 1)
    heatmap_uint8 = (heatmap_clipped * 255).astype(np.uint8)
    heatmap_xyz = np.transpose(heatmap_uint8, (2, 1, 0))
    nii = nib.Nifti1Image(heatmap_xyz, affine=affine)
    nib.save(nii, output_path)
    print(f"[INFO] 热力图(uint8)已保存: {output_path}")


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_heatmap_slices(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_path: str,
    spacing_zyx: Tuple[float, float, float] = None,
    alpha: float = 0.4,
    cmap: str = 'jet',
    title: str = None,
    percentile_clip: Tuple[float, float] = (1, 99),
):
    """
    生成三视图切片可视化
    
    Args:
        mra_data: MRA数据，ZYX顺序
        heatmap_data: 热力图数据，ZYX顺序
        output_path: 输出路径
        spacing_zyx: 体素间距 (spacing_z, spacing_y, spacing_x)，用于计算正确的显示比例
        alpha: 热力图叠加透明度
        cmap: 热力图颜色映射
        title: 标题
        percentile_clip: 显示裁剪百分位数
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    max_pos = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    z_max, y_max, x_max = max_pos
    
    mra_display = mra_data.copy()
    if (mra_display > 0).any():
        lo = np.percentile(mra_display[mra_display > 0], percentile_clip[0])
        hi = np.percentile(mra_display[mra_display > 0], percentile_clip[1])
    else:
        lo, hi = 0, 1
    mra_display = np.clip(mra_display, lo, hi)
    mra_display = (mra_display - lo) / (hi - lo + 1e-8)
    
    # 默认spacing（各向同性）
    if spacing_zyx is None:
        spacing_zyx = (1.0, 1.0, 1.0)
    
    spacing_z, spacing_y, spacing_x = spacing_zyx
    
    # aspect = 行方向间距 / 列方向间距（使物理尺寸正确显示）
    aspect_axial = spacing_y / spacing_x      # Axial (Y行, X列): Y间距/X间距
    aspect_coronal = spacing_z / spacing_x    # Coronal (Z行, X列): Z间距/X间距
    aspect_sagittal = spacing_z / spacing_y   # Sagittal (Z行, Y列): Z间距/Y间距
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 标准医学图像方向：
    # - Axial: R(右)在左, L(左)在右 -> 翻转X轴
    # - Coronal: R(右)在左, L(左)在右 -> 翻转X轴
    # - Sagittal: A(前)在左, P(后)在右 -> 翻转Y轴
    # 使用 [:, ::-1] 进行水平翻转
    axes[0, 0].imshow(mra_display[z_max, :, :][:, ::-1], cmap='gray', origin='lower', aspect=aspect_axial)
    axes[0, 0].set_title(f'Axial (z={z_max})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mra_display[:, y_max, :][:, ::-1], cmap='gray', origin='lower', aspect=aspect_coronal)
    axes[0, 1].set_title(f'Coronal (y={y_max})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mra_display[:, :, x_max][:, ::-1], cmap='gray', origin='lower', aspect=aspect_sagittal)
    axes[0, 2].set_title(f'Sagittal (x={x_max})')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(mra_display[z_max, :, :][:, ::-1], cmap='gray', origin='lower', aspect=aspect_axial)
    im1 = axes[1, 0].imshow(heatmap_data[z_max, :, :][:, ::-1], cmap=cmap, alpha=alpha, origin='lower',
                            aspect=aspect_axial, norm=Normalize(vmin=0, vmax=1))
    axes[1, 0].set_title(f'Axial + Heatmap')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mra_display[:, y_max, :][:, ::-1], cmap='gray', origin='lower', aspect=aspect_coronal)
    axes[1, 1].imshow(heatmap_data[:, y_max, :][:, ::-1], cmap=cmap, alpha=alpha, origin='lower',
                      aspect=aspect_coronal, norm=Normalize(vmin=0, vmax=1))
    axes[1, 1].set_title(f'Coronal + Heatmap')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mra_display[:, :, x_max][:, ::-1], cmap='gray', origin='lower', aspect=aspect_sagittal)
    axes[1, 2].imshow(heatmap_data[:, :, x_max][:, ::-1], cmap=cmap, alpha=alpha, origin='lower',
                      aspect=aspect_sagittal, norm=Normalize(vmin=0, vmax=1))
    axes[1, 2].set_title(f'Sagittal + Heatmap')
    axes[1, 2].axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(im1, cax=cbar_ax, label='Activation')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 可视化已保存: {output_path}")


def export_axial_slices(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_dir: str,
    spacing_zyx: Tuple[float, float, float] = None,
    alpha: float = 0.4,
    cmap: str = 'jet',
    prefix: str = 'slice',
    percentile_clip: Tuple[float, float] = (1, 99),
):
    """
    导出所有三个轴位的切片为单独的PNG文件
    
    Args:
        mra_data: MRA数据，ZYX顺序
        heatmap_data: 热力图数据，ZYX顺序
        output_dir: 输出目录
        spacing_zyx: 体素间距 (spacing_z, spacing_y, spacing_x)，用于计算正确的显示比例
        alpha: 热力图叠加透明度
        cmap: 热力图颜色映射
        prefix: 文件名前缀
        percentile_clip: 显示裁剪百分位数
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    mra_display = mra_data.copy()
    if (mra_display > 0).any():
        lo = np.percentile(mra_display[mra_display > 0], percentile_clip[0])
        hi = np.percentile(mra_display[mra_display > 0], percentile_clip[1])
    else:
        lo, hi = 0, 1
    mra_display = np.clip(mra_display, lo, hi)
    mra_display = (mra_display - lo) / (hi - lo + 1e-8)
    
    # 默认spacing（各向同性）
    if spacing_zyx is None:
        spacing_zyx = (1.0, 1.0, 1.0)
    
    spacing_z, spacing_y, spacing_x = spacing_zyx
    
    # 数据是ZYX顺序，标准医学图像方向：
    # - Axial: 沿Z切，显示YX平面，R(右)在左L(左)在右 -> 翻转X轴
    # - Coronal: 沿Y切，显示ZX平面，R(右)在左L(左)在右 -> 翻转X轴
    # - Sagittal: 沿X切，显示ZY平面，A(前)在左P(后)在右 -> 翻转Y轴
    # aspect = 行方向间距 / 列方向间距，使物理尺寸正确显示
    # 使用 [:, ::-1] 进行水平翻转
    orientations = {
        'axial': {
            'num_slices': mra_data.shape[0],  # Z方向
            'get_slice': lambda data, i: data[i, :, :][:, ::-1],  # (Y, X) -> 翻转X轴
            'aspect': spacing_y / spacing_x,  # 行是Y，列是X -> Y间距/X间距
        },
        'coronal': {
            'num_slices': mra_data.shape[1],  # Y方向
            'get_slice': lambda data, i: data[:, i, :][:, ::-1],  # (Z, X) -> 翻转X轴
            'aspect': spacing_z / spacing_x,  # 行是Z，列是X -> Z间距/X间距
        },
        'sagittal': {
            'num_slices': mra_data.shape[2],  # X方向
            'get_slice': lambda data, i: data[:, :, i][:, ::-1],  # (Z, Y) -> 翻转Y轴
            'aspect': spacing_z / spacing_y,  # 行是Z，列是Y -> Z间距/Y间距
        },
    }
    
    # 固定figsize，通过aspect控制比例
    figsize = (8, 8)
    
    for orient_name, orient_info in orientations.items():
        orient_dir = os.path.join(output_dir, orient_name)
        
        mra_dir = os.path.join(orient_dir, "mra")
        heatmap_dir = os.path.join(orient_dir, "heatmap")
        overlay_dir = os.path.join(orient_dir, "overlay")
        os.makedirs(mra_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        
        num_slices = orient_info['num_slices']
        get_slice = orient_info['get_slice']
        aspect = orient_info['aspect']
        
        print(f"[INFO] 正在导出 {num_slices} 个 {orient_name} 切片 (aspect={aspect:.3f})...")
        
        for i in range(num_slices):
            mra_slice = get_slice(mra_display, i)
            heatmap_slice = get_slice(heatmap_data, i)
            
            # MRA
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(mra_slice, cmap='gray', origin='lower', aspect=aspect)
            ax.axis('off')
            plt.savefig(os.path.join(mra_dir, f"{prefix}_{i:03d}.png"), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Heatmap
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(heatmap_slice, cmap=cmap, origin='lower', aspect=aspect,
                      norm=Normalize(vmin=0, vmax=1))
            ax.axis('off')
            plt.savefig(os.path.join(heatmap_dir, f"{prefix}_{i:03d}.png"), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Overlay
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(mra_slice, cmap='gray', origin='lower', aspect=aspect)
            ax.imshow(heatmap_slice, cmap=cmap, alpha=alpha, origin='lower', aspect=aspect,
                      norm=Normalize(vmin=0, vmax=1))
            ax.axis('off')
            plt.savefig(os.path.join(overlay_dir, f"{prefix}_{i:03d}.png"), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            if (i + 1) % 50 == 0 or i == num_slices - 1:
                print(f"[INFO] {orient_name}: 已导出 {i + 1}/{num_slices} 切片")
    
    print(f"[INFO] 所有切片已导出到: {output_dir}")
