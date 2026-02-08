"""
Grad-CAM工具模块

包含:
1. GradCAM3D类 - 用于3D CNN的Grad-CAM实现
2. 预处理函数 - 带变换参数记录的预处理
3. 逆变换函数 - 将热力图从预处理空间映射回原始空间
4. 可视化函数 - 多切片可视化
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 预处理常量（与训练一致）
# =============================================================================
SPACING_MM = (1.0, 1.0, 1.0)
TARGET_SHAPE = (160, 160, 160)
CROP_MARGIN_VOX = 10


# =============================================================================
# 辅助函数
# =============================================================================

def load_canonical(path: str) -> nib.Nifti1Image:
    """加载NIfTI并转换为标准方向"""
    nii = nib.load(path)
    return nib.as_closest_canonical(nii)


def resample_iso(nii: nib.Nifti1Image, spacing_mm: Tuple[float, float, float], order: int) -> nib.Nifti1Image:
    """等向性重采样"""
    return resample_to_output(nii, voxel_sizes=spacing_mm, order=order)


def to_mask_u8(x: np.ndarray) -> np.ndarray:
    """转换为uint8 mask"""
    if x.dtype != np.uint8:
        x = (x > 0.5).astype(np.uint8)
    return x


def centroid_split_idx(mask_u8: np.ndarray) -> int:
    """基于mask质心计算分割索引"""
    coords = np.argwhere(mask_u8 > 0)
    if coords.shape[0] == 0:
        return mask_u8.shape[0] // 2
    cx = float(coords[:, 0].mean())
    idx = int(round(cx))
    idx = max(1, min(mask_u8.shape[0] - 1, idx))
    return idx


def bbox_3d(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """计算3D bounding box"""
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def normalize_mra_nonzero(mra: np.ndarray) -> np.ndarray:
    """
    非零像素归一化（与训练一致）
    """
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
# 带变换参数记录的预处理
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


def preprocess_with_transform_record(
    mra_path: str,
    mask_path: str,
    hemi: str = 'L',
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    crop_margin_vox: int = CROP_MARGIN_VOX,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    执行预处理并记录所有变换参数，用于后续逆变换
    
    Args:
        mra_path: 原始MRA文件路径
        mask_path: 原始Mask文件路径
        hemi: 'L' 或 'R'，指定要处理的半脑
        target_shape: 目标形状
        crop_margin_vox: bbox裁剪margin
    
    Returns:
        preprocessed_input: [2, D, H, W] 模型输入
        transform_params: dict 包含所有逆变换所需参数
    """
    transform_params = {
        "mra_path": mra_path,
        "mask_path": mask_path,
        "hemi": hemi,
        "target_shape": list(target_shape),
        "crop_margin_vox": int(crop_margin_vox),
    }
    
    # 1. 加载原始数据
    mra_nii_orig = nib.load(mra_path)
    mask_nii_orig = nib.load(mask_path)
    
    transform_params["original_shape"] = [int(x) for x in mra_nii_orig.shape]
    transform_params["original_affine"] = [[float(x) for x in row] for row in mra_nii_orig.affine]
    transform_params["original_spacing"] = [float(x) for x in mra_nii_orig.header.get_zooms()[:3]]
    
    # 2. 标准方向重定向
    mra_nii_canonical = nib.as_closest_canonical(mra_nii_orig)
    mask_nii_canonical = nib.as_closest_canonical(mask_nii_orig)
    
    transform_params["canonical_shape"] = [int(x) for x in mra_nii_canonical.shape]
    transform_params["canonical_affine"] = [[float(x) for x in row] for row in mra_nii_canonical.affine]
    
    # 3. 等向性重采样
    mra_nii_resampled = resample_iso(mra_nii_canonical, SPACING_MM, order=1)
    mask_nii_resampled = resample_iso(mask_nii_canonical, SPACING_MM, order=0)
    
    mra_np = mra_nii_resampled.get_fdata(dtype=np.float32)
    mask_np = to_mask_u8(mask_nii_resampled.get_fdata(dtype=np.float32))
    
    transform_params["resampled_shape"] = [int(x) for x in mra_np.shape]
    transform_params["resampled_affine"] = [[float(x) for x in row] for row in mra_nii_resampled.affine]
    
    # 4. 半脑分割
    split_idx = centroid_split_idx(mask_np)
    transform_params["split_idx"] = int(split_idx)
    transform_params["is_right_hemi"] = bool(hemi == 'R')
    
    # 计算原图空间的分割索引（使用相同比例）
    canonical_shape = transform_params["canonical_shape"]
    resampled_shape = transform_params["resampled_shape"]
    split_idx_original = int(split_idx * canonical_shape[0] / resampled_shape[0])
    transform_params["split_idx_original"] = split_idx_original
    
    # 计算原图半脑形状
    if hemi == 'L':
        original_hemi_shape = [split_idx_original, canonical_shape[1], canonical_shape[2]]
    else:
        original_hemi_shape = [canonical_shape[0] - split_idx_original, canonical_shape[1], canonical_shape[2]]
    transform_params["original_hemi_shape"] = [int(x) for x in original_hemi_shape]
    
    if hemi == 'L':
        hemi_mra = mra_np[:split_idx, :, :]
        hemi_mask = mask_np[:split_idx, :, :]
    else:  # R
        hemi_mra = mra_np[split_idx:, :, :]
        hemi_mask = mask_np[split_idx:, :, :]
        # 右半球翻转对齐
        hemi_mra = np.flip(hemi_mra, axis=0).copy()
        hemi_mask = np.flip(hemi_mask, axis=0).copy()
    
    transform_params["hemi_shape_before_bbox"] = [int(x) for x in hemi_mra.shape]
    
    # 5. BBox裁剪
    bb = bbox_3d(hemi_mask)
    if bb is not None:
        zmin, zmax, ymin, ymax, xmin, xmax = bb
        # 添加margin
        zmin_m = max(0, zmin - crop_margin_vox)
        ymin_m = max(0, ymin - crop_margin_vox)
        xmin_m = max(0, xmin - crop_margin_vox)
        zmax_m = min(hemi_mra.shape[0] - 1, zmax + crop_margin_vox)
        ymax_m = min(hemi_mra.shape[1] - 1, ymax + crop_margin_vox)
        xmax_m = min(hemi_mra.shape[2] - 1, xmax + crop_margin_vox)
        
        transform_params["bbox"] = [int(zmin_m), int(zmax_m), int(ymin_m), int(ymax_m), int(xmin_m), int(xmax_m)]
        
        hemi_mra = hemi_mra[zmin_m:zmax_m+1, ymin_m:ymax_m+1, xmin_m:xmax_m+1]
        hemi_mask = hemi_mask[zmin_m:zmax_m+1, ymin_m:ymax_m+1, xmin_m:xmax_m+1]
    else:
        transform_params["bbox"] = None
    
    transform_params["shape_after_bbox"] = [int(x) for x in hemi_mra.shape]
    
    # 6. Pad/Crop到目标形状
    tz, ty, tx = target_shape
    z, y, x = hemi_mra.shape
    
    # 计算padding
    pad_z0 = max(0, (tz - z) // 2)
    pad_z1 = max(0, tz - z - pad_z0)
    pad_y0 = max(0, (ty - y) // 2)
    pad_y1 = max(0, ty - y - pad_y0)
    pad_x0 = max(0, (tx - x) // 2)
    pad_x1 = max(0, tx - x - pad_x0)
    
    transform_params["pad_before"] = [int(pad_z0), int(pad_y0), int(pad_x0)]
    transform_params["pad_after"] = [int(pad_z1), int(pad_y1), int(pad_x1)]
    
    if pad_z0 or pad_z1 or pad_y0 or pad_y1 or pad_x0 or pad_x1:
        hemi_mra = np.pad(hemi_mra, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), 
                         mode="constant", constant_values=0)
        hemi_mask = np.pad(hemi_mask, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), 
                          mode="constant", constant_values=0)
    
    # 计算crop
    z, y, x = hemi_mra.shape
    crop_z0 = max(0, (z - tz) // 2)
    crop_y0 = max(0, (y - ty) // 2)
    crop_x0 = max(0, (x - tx) // 2)
    
    transform_params["crop_start"] = [int(crop_z0), int(crop_y0), int(crop_x0)]
    transform_params["shape_before_crop"] = [int(z), int(y), int(x)]
    
    hemi_mra = hemi_mra[crop_z0:crop_z0+tz, crop_y0:crop_y0+ty, crop_x0:crop_x0+tx]
    hemi_mask = hemi_mask[crop_z0:crop_z0+tz, crop_y0:crop_y0+ty, crop_x0:crop_x0+tx]
    
    # 7. 归一化
    hemi_mra = normalize_mra_nonzero(hemi_mra)
    hemi_mask = hemi_mask.astype(np.float32)
    
    # 8. 堆叠为双通道
    preprocessed_input = np.stack([hemi_mra, hemi_mask], axis=0).astype(np.float32)
    
    return preprocessed_input, transform_params


# =============================================================================
# Grad-CAM 3D 实现
# =============================================================================

class GradCAM3D:
    """
    3D Grad-CAM实现
    
    用于DenseNet121等3D CNN的可解释性分析
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str = 'features.denseblock4'):
        """
        Args:
            model: 3D CNN模型
            target_layer_name: 目标层名称，用于生成CAM
        """
        self.model = model
        self.target_layer_name = target_layer_name
        
        self.activations = None
        self.gradients = None
        
        # 注册hooks
        self._register_hooks()
    
    def _get_target_layer(self) -> nn.Module:
        """获取目标层"""
        parts = self.target_layer_name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _register_hooks(self):
        """注册前向和反向hooks"""
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
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: [B, C, D, H, W] 输入张量
            target_class: 目标类别索引，None则使用预测类别
            upsample_to_input: 是否上采样到输入大小
        
        Returns:
            heatmap: [D, H, W] 热力图（如果upsample_to_input=True，则与输入同大小）
        """
        self.model.eval()
        
        # 确保input_tensor需要梯度
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重（全局平均池化梯度）
        # gradients shape: [B, C, D, H, W]
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, D, H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # 转换为numpy
        cam = cam.squeeze().cpu().numpy()  # [D, H, W]
        
        # 归一化到[0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        # 上采样到输入大小
        if upsample_to_input:
            input_shape = input_tensor.shape[2:]  # [D, H, W]
            cam = self._upsample_3d(cam, input_shape)
        
        return cam
    
    def _upsample_3d(self, cam: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """使用三次样条插值上采样，减少横线伪影"""
        zoom_factors = [t / c for t, c in zip(target_shape, cam.shape)]
        cam_upsampled = zoom(cam, zoom_factors, order=3)  # order=3 三次样条插值，减少横线伪影
        return cam_upsampled


def generate_ensemble_gradcam(
    models: List[nn.Module],
    input_tensor: torch.Tensor,
    target_layer_name: str = 'features.denseblock4',
    target_class: Optional[int] = None,
    device: torch.device = None,
) -> Tuple[np.ndarray, int]:
    """
    使用多个模型生成集成Grad-CAM热力图
    
    Args:
        models: 模型列表
        input_tensor: [1, C, D, H, W] 输入张量
        target_layer_name: 目标层名称
        target_class: 目标类别，None则使用集成预测
        device: 计算设备
    
    Returns:
        ensemble_cam: [D, H, W] 集成热力图
        predicted_class: 集成预测类别
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cams = []
    probs_sum = None
    
    for model in models:
        model = model.to(device)
        model.eval()
        
        input_tensor_dev = input_tensor.to(device)
        
        # 获取预测概率
        with torch.no_grad():
            logits = model(input_tensor_dev)
            probs = F.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = probs.cpu().numpy()
            else:
                probs_sum += probs.cpu().numpy()
        
        # 生成CAM
        gradcam = GradCAM3D(model, target_layer_name)
        
        # 如果没有指定target_class，先用当前模型预测
        if target_class is None:
            cam = gradcam.generate(input_tensor_dev, target_class=None)
        else:
            cam = gradcam.generate(input_tensor_dev, target_class=target_class)
        
        cams.append(cam)
    
    # 集成预测类别
    avg_probs = probs_sum / len(models)
    predicted_class = int(avg_probs.argmax())
    
    # 平均热力图
    ensemble_cam = np.mean(cams, axis=0)
    
    # 重新归一化
    if ensemble_cam.max() > ensemble_cam.min():
        ensemble_cam = (ensemble_cam - ensemble_cam.min()) / (ensemble_cam.max() - ensemble_cam.min())
    
    return ensemble_cam, predicted_class


# =============================================================================
# 逆变换函数
# =============================================================================

def inverse_transform_heatmap_to_hemis(
    heatmap: np.ndarray,
    transform_params: Dict[str, Any],
) -> np.ndarray:
    """
    将热力图逆变换到半球切割后的 MRA 空间
    
    流程:
    1. 逆Crop
    2. 逆Pad
    3. 逆BBox裁剪
    
    输出形状与半球 MRA 图像一致
    
    Args:
        heatmap: [D, H, W] 预处理空间中的热力图（160x160x160）
        transform_params: 预处理时记录的变换参数
    
    Returns:
        heatmap_hemis: 半球 MRA 空间中的热力图
    """
    # 1. 逆Crop
    crop_start = transform_params.get("crop_start", [0, 0, 0])
    shape_before_crop = transform_params.get("shape_before_crop", list(heatmap.shape))
    
    # 创建crop前大小的数组
    heatmap_uncropped = np.zeros(shape_before_crop, dtype=np.float32)
    cz, cy, cx = crop_start
    tz, ty, tx = heatmap.shape
    heatmap_uncropped[cz:cz+tz, cy:cy+ty, cx:cx+tx] = heatmap
    
    # 2. 逆Pad
    pad_before = transform_params.get("pad_before", [0, 0, 0])
    pad_after = transform_params.get("pad_after", [0, 0, 0])
    
    pz0, py0, px0 = pad_before
    pz1, py1, px1 = pad_after
    
    # 去除padding
    z, y, x = heatmap_uncropped.shape
    heatmap_unpadded = heatmap_uncropped[
        pz0:(z - pz1) if pz1 > 0 else z,
        py0:(y - py1) if py1 > 0 else y,
        px0:(x - px1) if px1 > 0 else x,
    ]
    
    # 3. 逆BBox裁剪
    hemi_shape = transform_params.get("hemi_shape_before_bbox", list(heatmap_unpadded.shape))
    bbox = transform_params.get("bbox")
    
    heatmap_hemis = np.zeros(hemi_shape, dtype=np.float32)
    if bbox is not None:
        zmin, zmax, ymin, ymax, xmin, xmax = bbox
        # 确保不超出边界
        dz = min(heatmap_unpadded.shape[0], zmax - zmin + 1)
        dy = min(heatmap_unpadded.shape[1], ymax - ymin + 1)
        dx = min(heatmap_unpadded.shape[2], xmax - xmin + 1)
        heatmap_hemis[zmin:zmin+dz, ymin:ymin+dy, xmin:xmin+dx] = heatmap_unpadded[:dz, :dy, :dx]
    else:
        heatmap_hemis = heatmap_unpadded
    
    return heatmap_hemis


def inverse_transform_heatmap(
    heatmap: np.ndarray,
    transform_params: Dict[str, Any],
    return_resampled_space: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将热力图从预处理空间逆变换到原始空间
    
    Args:
        heatmap: [D, H, W] 预处理空间中的热力图（160x160x160）
        transform_params: 预处理时记录的变换参数
        return_resampled_space: 是否返回重采样空间的热力图（不做最后的逆重采样）
    
    Returns:
        heatmap_original: 原始空间中的热力图
        full_brain_resampled: 重采样空间中的完整大脑热力图
    """
    # 1. 逆Crop
    crop_start = transform_params.get("crop_start", [0, 0, 0])
    shape_before_crop = transform_params.get("shape_before_crop", list(heatmap.shape))
    
    # 创建crop前大小的数组
    heatmap_uncropped = np.zeros(shape_before_crop, dtype=np.float32)
    cz, cy, cx = crop_start
    tz, ty, tx = heatmap.shape
    heatmap_uncropped[cz:cz+tz, cy:cy+ty, cx:cx+tx] = heatmap
    
    # 2. 逆Pad
    pad_before = transform_params.get("pad_before", [0, 0, 0])
    pad_after = transform_params.get("pad_after", [0, 0, 0])
    
    pz0, py0, px0 = pad_before
    pz1, py1, px1 = pad_after
    
    # 去除padding
    z, y, x = heatmap_uncropped.shape
    heatmap_unpadded = heatmap_uncropped[
        pz0:(z - pz1) if pz1 > 0 else z,
        py0:(y - py1) if py1 > 0 else y,
        px0:(x - px1) if px1 > 0 else x,
    ]
    
    # 3. 逆BBox裁剪
    hemi_shape = transform_params.get("hemi_shape_before_bbox", list(heatmap_unpadded.shape))
    bbox = transform_params.get("bbox")
    
    heatmap_hemi = np.zeros(hemi_shape, dtype=np.float32)
    if bbox is not None:
        zmin, zmax, ymin, ymax, xmin, xmax = bbox
        # 确保不超出边界
        dz = min(heatmap_unpadded.shape[0], zmax - zmin + 1)
        dy = min(heatmap_unpadded.shape[1], ymax - ymin + 1)
        dx = min(heatmap_unpadded.shape[2], xmax - xmin + 1)
        heatmap_hemi[zmin:zmin+dz, ymin:ymin+dy, xmin:xmin+dx] = heatmap_unpadded[:dz, :dy, :dx]
    else:
        heatmap_hemi = heatmap_unpadded
    
    # 4. 逆翻转（如果是右半球）
    is_right_hemi = transform_params.get("is_right_hemi", False)
    if is_right_hemi:
        heatmap_hemi = np.flip(heatmap_hemi, axis=0).copy()
    
    # 5. 放回完整大脑（重采样空间）
    resampled_shape = transform_params.get("resampled_shape", hemi_shape)
    split_idx = transform_params.get("split_idx", resampled_shape[0] // 2)
    
    full_brain_resampled = np.zeros(resampled_shape, dtype=np.float32)
    
    if is_right_hemi:
        # 右半球放到split_idx之后
        end_idx = min(split_idx + heatmap_hemi.shape[0], resampled_shape[0])
        actual_len = end_idx - split_idx
        full_brain_resampled[split_idx:end_idx, :heatmap_hemi.shape[1], :heatmap_hemi.shape[2]] = heatmap_hemi[:actual_len, :, :]
    else:
        # 左半球放到split_idx之前
        actual_len = min(split_idx, heatmap_hemi.shape[0])
        full_brain_resampled[:actual_len, :heatmap_hemi.shape[1], :heatmap_hemi.shape[2]] = heatmap_hemi[:actual_len, :, :]
    
    if return_resampled_space:
        return full_brain_resampled, full_brain_resampled
    
    # 6. 逆重采样到原始分辨率
    original_shape = transform_params.get("original_shape", resampled_shape)
    
    # 使用scipy zoom进行重采样（order=3 三次样条插值，减少横线伪影）
    zoom_factors = [o / r for o, r in zip(original_shape, resampled_shape)]
    heatmap_original = zoom(full_brain_resampled, zoom_factors, order=3)
    
    return heatmap_original, full_brain_resampled


def save_heatmap_as_nifti(
    heatmap: np.ndarray,
    affine: np.ndarray,
    output_path: str,
):
    """
    保存热力图为NIfTI文件
    
    Args:
        heatmap: [D, H, W] 热力图
        affine: 4x4 affine矩阵
        output_path: 输出路径
    """
    nii = nib.Nifti1Image(heatmap.astype(np.float32), affine=affine)
    nib.save(nii, output_path)
    print(f"[INFO] 热力图已保存: {output_path}")


def save_heatmap_as_nifti_uint8(
    heatmap: np.ndarray,
    affine: np.ndarray,
    output_path: str,
):
    """
    保存热力图为uint8格式NIfTI文件（0-255），便于在ITK-SNAP中叠加
    
    Args:
        heatmap: [D, H, W] 热力图，范围应为[0, 1]
        affine: 4x4 affine矩阵
        output_path: 输出路径
    """
    # 确保范围在[0, 1]
    heatmap_clipped = np.clip(heatmap, 0, 1)
    # 缩放到[0, 255]并转为uint8
    heatmap_uint8 = (heatmap_clipped * 255).astype(np.uint8)
    nii = nib.Nifti1Image(heatmap_uint8, affine=affine)
    nib.save(nii, output_path)
    print(f"[INFO] 热力图(uint8)已保存: {output_path}")


def inverse_transform_heatmap_to_original_hemi(
    heatmap: np.ndarray,
    transform_params: Dict[str, Any],
) -> np.ndarray:
    """
    将热力图逆变换到原图半脑空间（Canonical后，未重采样，已分割）
    
    流程:
    1. 逆Crop
    2. 逆Pad
    3. 逆BBox裁剪 → 半球空间 (1.0mm)
    4. 逆重采样 → 原图半脑空间
    
    Args:
        heatmap: [D, H, W] 预处理空间中的热力图（160x160x160）
        transform_params: 预处理时记录的变换参数
    
    Returns:
        heatmap_original_hemi: 原图半脑空间中的热力图
    """
    # 1-3. 先逆变换到半球 MRA 空间
    heatmap_hemis = inverse_transform_heatmap_to_hemis(heatmap, transform_params)
    
    # 4. 逆重采样到原图半脑空间
    original_hemi_shape = transform_params.get("original_hemi_shape")
    if original_hemi_shape is None:
        print("[WARN] 未找到original_hemi_shape参数，无法逆重采样")
        return heatmap_hemis
    
    # 计算zoom因子
    zoom_factors = [o / h for o, h in zip(original_hemi_shape, heatmap_hemis.shape)]
    
    # 使用scipy zoom进行逆重采样（order=3 三次样条插值，减少横线伪影）
    heatmap_original_hemi = zoom(heatmap_hemis, zoom_factors, order=3)
    
    # 确保形状匹配
    if list(heatmap_original_hemi.shape) != list(original_hemi_shape):
        # 如果有微小差异，进行裁剪或填充
        result = np.zeros(original_hemi_shape, dtype=np.float32)
        min_shape = [min(a, b) for a, b in zip(heatmap_original_hemi.shape, original_hemi_shape)]
        result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
            heatmap_original_hemi[:min_shape[0], :min_shape[1], :min_shape[2]]
        heatmap_original_hemi = result
    
    return heatmap_original_hemi


def get_original_hemi_data(
    mra_path: str,
    mask_path: str,
    hemi: str,
    transform_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从原图获取半脑数据（Canonical后，未重采样，已分割）
    
    Args:
        mra_path: 原始MRA文件路径
        mask_path: 原始Mask文件路径
        hemi: 'L' 或 'R'
        transform_params: 预处理时记录的变换参数
    
    Returns:
        hemi_mra: 原图半脑MRA数据
        affine: 原图affine矩阵
    """
    # 加载并Canonical
    mra_nii = nib.as_closest_canonical(nib.load(mra_path))
    
    # 获取分割索引
    split_idx_original = transform_params.get("split_idx_original")
    if split_idx_original is None:
        # 兼容旧版本：使用比例计算
        split_idx = transform_params.get("split_idx", 0)
        canonical_shape = transform_params.get("canonical_shape", list(mra_nii.shape))
        resampled_shape = transform_params.get("resampled_shape", canonical_shape)
        split_idx_original = int(split_idx * canonical_shape[0] / resampled_shape[0])
    
    mra_data = mra_nii.get_fdata(dtype=np.float32)
    
    # 分割半脑
    if hemi == 'L':
        hemi_mra = mra_data[:split_idx_original, :, :]
    else:  # R
        hemi_mra = mra_data[split_idx_original:, :, :]
        # 右半球翻转对齐
        hemi_mra = np.flip(hemi_mra, axis=0).copy()
    
    return hemi_mra, mra_nii.affine


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_heatmap_slices(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_path: str,
    spacing: Tuple[float, float, float] = None,
    hemi: str = None,
    alpha: float = 0.4,
    cmap: str = 'jet',
    title: str = None,
    percentile_clip: Tuple[float, float] = (1, 99),
):
    """
    生成三视图切片可视化（RAS+坐标系，符合标准医学图像方向）
    
    放射学约定: R(右侧)在图像左边，L(左侧)在图像右边
    
    注意：半脑数据在预处理时的处理不同：
    - 右半脑(R): 预处理时经过np.flip(axis=0)翻转，可视化时不需要再翻转R轴
    - 左半脑(L): 预处理时未翻转，可视化时需要翻转R轴
    
    Args:
        mra_data: [R, A, S] MRA数据（RAS+坐标系）
        heatmap_data: [R, A, S] 热力图数据（RAS+坐标系）
        output_path: 输出PNG路径
        spacing: 体素间距 (spacing_R, spacing_A, spacing_S)，用于计算正确的显示比例
        hemi: 半脑类型 'L' 或 'R'，用于确定R轴翻转方向
        alpha: 热力图透明度
        cmap: 热力图颜色映射
        title: 图像标题
        percentile_clip: MRA显示的分位数clip
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # 找到热力图最大激活位置
    max_pos = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    r_max, a_max, s_max = max_pos  # RAS+坐标系
    
    # MRA归一化显示
    mra_display = mra_data.copy()
    lo = np.percentile(mra_display[mra_display > 0], percentile_clip[0]) if (mra_display > 0).any() else 0
    hi = np.percentile(mra_display[mra_display > 0], percentile_clip[1]) if (mra_display > 0).any() else 1
    mra_display = np.clip(mra_display, lo, hi)
    mra_display = (mra_display - lo) / (hi - lo + 1e-8)
    
    # 默认spacing（各向同性）
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing_r, spacing_a, spacing_s = spacing
    
    # RAS+坐标系下的切片方向和aspect ratio计算:
    # 转置后: Axial=(A行,R列), Coronal=(S行,R列), Sagittal=(S行,A列)
    # aspect = 行spacing / 列spacing
    aspect_axial = spacing_a / spacing_r     # Axial: A行/R列
    aspect_coronal = spacing_s / spacing_r   # Coronal: S行/R列
    aspect_sagittal = spacing_s / spacing_a  # Sagittal: S行/A列
    
    # 根据半脑类型决定R轴翻转
    # 放射学约定: R(右)在图像左边
    # - 右半脑(R): 预处理时已翻转，不需要再翻转R轴
    # - 左半脑(L): 需要翻转R轴使R在左
    flip_r = (hemi == 'L') if hemi else True  # 默认翻转（兼容旧代码）
    
    # 创建图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 定义切片获取函数
    def get_axial_slice(data, idx):
        # data[:,:,s]=(R,A) -> .T=(A,R)
        slice_2d = data[:, :, idx].T
        if flip_r:
            slice_2d = slice_2d[:, ::-1]  # 翻转R列
        return slice_2d
    
    def get_coronal_slice(data, idx):
        # data[:,a,:]=(R,S) -> .T=(S,R)
        slice_2d = data[:, idx, :].T
        if flip_r:
            slice_2d = slice_2d[:, ::-1]  # 翻转R列
        return slice_2d
    
    def get_sagittal_slice(data, idx):
        # data[r,:,:]=(A,S) -> .T=(S,A) -> 翻转A列(使A前在左)
        slice_2d = data[idx, :, :].T[:, ::-1]
        return slice_2d
    
    # 第一行：原始MRA
    axes[0, 0].imshow(get_axial_slice(mra_display, s_max), cmap='gray', origin='lower', aspect=aspect_axial)
    axes[0, 0].set_title(f'Axial (s={s_max})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(get_coronal_slice(mra_display, a_max), cmap='gray', origin='lower', aspect=aspect_coronal)
    axes[0, 1].set_title(f'Coronal (a={a_max})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(get_sagittal_slice(mra_display, r_max), cmap='gray', origin='lower', aspect=aspect_sagittal)
    axes[0, 2].set_title(f'Sagittal (r={r_max})')
    axes[0, 2].axis('off')
    
    # 第二行：MRA + 热力图叠加
    # Axial
    axes[1, 0].imshow(get_axial_slice(mra_display, s_max), cmap='gray', origin='lower', aspect=aspect_axial)
    im1 = axes[1, 0].imshow(get_axial_slice(heatmap_data, s_max), cmap=cmap, alpha=alpha, origin='lower',
                            aspect=aspect_axial, norm=Normalize(vmin=0, vmax=1))
    axes[1, 0].set_title(f'Axial + Heatmap')
    axes[1, 0].axis('off')
    
    # Coronal
    axes[1, 1].imshow(get_coronal_slice(mra_display, a_max), cmap='gray', origin='lower', aspect=aspect_coronal)
    axes[1, 1].imshow(get_coronal_slice(heatmap_data, a_max), cmap=cmap, alpha=alpha, origin='lower',
                      aspect=aspect_coronal, norm=Normalize(vmin=0, vmax=1))
    axes[1, 1].set_title(f'Coronal + Heatmap')
    axes[1, 1].axis('off')
    
    # Sagittal
    axes[1, 2].imshow(get_sagittal_slice(mra_display, r_max), cmap='gray', origin='lower', aspect=aspect_sagittal)
    axes[1, 2].imshow(get_sagittal_slice(heatmap_data, r_max), cmap=cmap, alpha=alpha, origin='lower',
                      aspect=aspect_sagittal, norm=Normalize(vmin=0, vmax=1))
    axes[1, 2].set_title(f'Sagittal + Heatmap')
    axes[1, 2].axis('off')
    
    # 添加colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(im1, cax=cbar_ax, label='Activation')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 可视化已保存: {output_path}")


def visualize_preprocessed_space(
    preprocessed_mra: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
    cmap: str = 'jet',
    title: str = None,
):
    """
    在预处理空间生成三视图可视化（RAS+坐标系，符合标准医学图像方向）
    
    预处理空间使用1.0mm等向性spacing，aspect ratio均为1.0
    
    Args:
        preprocessed_mra: [R, A, S] 预处理后的MRA（RAS+坐标系）
        heatmap: [R, A, S] 热力图（RAS+坐标系）
        output_path: 输出路径
        alpha: 热力图透明度
        cmap: 颜色映射
        title: 标题
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # 找到热力图最大激活位置
    max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    r_max, a_max, s_max = max_pos  # RAS+坐标系
    
    # MRA归一化显示
    mra_display = preprocessed_mra.copy()
    # 预处理后的MRA已经归一化，直接映射到[0,1]显示
    mra_min, mra_max = mra_display.min(), mra_display.max()
    mra_display = (mra_display - mra_min) / (mra_max - mra_min + 1e-8)
    
    # 创建图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # RAS+坐标系下的标准医学图像方向（预处理空间是1.0mm等向性，aspect=1）
    # imshow: 第一维=行(垂直), 第二维=列(水平)
    # 需要转置(.T)使行列对应正确，然后翻转R轴使R(右)在左
    # - Axial: data[:,:,s].T -> (A,R) -> [:,::-1] 翻转R列
    # - Coronal: data[:,a,:].T -> (S,R) -> [:,::-1] 翻转R列
    # - Sagittal: data[r,:,:].T -> (S,A) -> [:,::-1] 翻转A列(使A前在左)
    
    # 第一行：原始MRA（预处理空间）
    axes[0, 0].imshow(mra_display[:, :, s_max].T[:, ::-1], cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Axial (s={s_max})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mra_display[:, a_max, :].T[:, ::-1], cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Coronal (a={a_max})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mra_display[r_max, :, :].T[:, ::-1], cmap='gray', origin='lower')
    axes[0, 2].set_title(f'Sagittal (r={r_max})')
    axes[0, 2].axis('off')
    
    # 第二行：MRA + 热力图叠加
    axes[1, 0].imshow(mra_display[:, :, s_max].T[:, ::-1], cmap='gray', origin='lower')
    im1 = axes[1, 0].imshow(heatmap[:, :, s_max].T[:, ::-1], cmap=cmap, alpha=alpha, origin='lower',
                            norm=Normalize(vmin=0, vmax=1))
    axes[1, 0].set_title(f'Axial + Heatmap')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mra_display[:, a_max, :].T[:, ::-1], cmap='gray', origin='lower')
    axes[1, 1].imshow(heatmap[:, a_max, :].T[:, ::-1], cmap=cmap, alpha=alpha, origin='lower',
                      norm=Normalize(vmin=0, vmax=1))
    axes[1, 1].set_title(f'Coronal + Heatmap')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mra_display[r_max, :, :].T[:, ::-1], cmap='gray', origin='lower')
    axes[1, 2].imshow(heatmap[r_max, :, :].T[:, ::-1], cmap=cmap, alpha=alpha, origin='lower',
                      norm=Normalize(vmin=0, vmax=1))
    axes[1, 2].set_title(f'Sagittal + Heatmap')
    axes[1, 2].axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(im1, cax=cbar_ax, label='Activation')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 预处理空间可视化已保存: {output_path}")


def export_axial_slices(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_dir: str,
    spacing: Tuple[float, float, float] = None,
    hemi: str = None,
    alpha: float = 0.4,
    cmap: str = 'jet',
    prefix: str = 'slice',
    percentile_clip: Tuple[float, float] = (1, 99),
):
    """
    导出所有三个轴位的切片为单独的PNG文件（RAS+坐标系，符合标准医学图像方向）
    
    放射学约定: R(右侧)在图像左边，L(左侧)在图像右边
    
    注意：半脑数据在预处理时的处理不同：
    - 右半脑(R): 预处理时经过np.flip(axis=0)翻转，可视化时不需要再翻转R轴
    - 左半脑(L): 预处理时未翻转，可视化时需要翻转R轴
    
    Args:
        mra_data: [R, A, S] MRA数据（RAS+坐标系）
        heatmap_data: [R, A, S] 热力图数据（RAS+坐标系）
        output_dir: 输出目录
        spacing: 体素间距 (spacing_R, spacing_A, spacing_S)，用于计算正确的显示比例
        hemi: 半脑类型 'L' 或 'R'，用于确定R轴翻转方向
        alpha: 热力图叠加透明度
        cmap: 热力图颜色映射
        prefix: 文件名前缀
        percentile_clip: MRA显示的分位数clip
    
    输出:
        output_dir/
        ├── axial/
        │   ├── mra/, heatmap/, overlay/
        ├── coronal/
        │   ├── mra/, heatmap/, overlay/
        └── sagittal/
            ├── mra/, heatmap/, overlay/
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import os
    
    # MRA归一化显示
    mra_display = mra_data.copy()
    if (mra_display > 0).any():
        lo = np.percentile(mra_display[mra_display > 0], percentile_clip[0])
        hi = np.percentile(mra_display[mra_display > 0], percentile_clip[1])
    else:
        lo, hi = 0, 1
    mra_display = np.clip(mra_display, lo, hi)
    mra_display = (mra_display - lo) / (hi - lo + 1e-8)
    
    # 默认spacing（各向同性）
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing_r, spacing_a, spacing_s = spacing
    
    # 根据半脑类型决定R轴翻转
    # 放射学约定: R(右)在图像左边
    # - 右半脑(R): 预处理时已翻转，不需要再翻转R轴
    # - 左半脑(L): 需要翻转R轴使R在左
    flip_r = (hemi == 'L') if hemi else True  # 默认翻转（兼容旧代码）
    
    # RAS+坐标系下的切片方向和aspect ratio计算:
    # imshow: 第一维=行(垂直), 第二维=列(水平)
    # 标准显示: Axial=(A行,R列), Coronal=(S行,R列), Sagittal=(S行,A列)
    
    # 定义切片获取函数
    def get_sagittal_slice(data, i):
        # data[r,:,:]=(A,S) -> .T=(S,A) -> 翻转A列(使A前在左)
        return data[i, :, :].T[:, ::-1]
    
    def get_coronal_slice(data, i):
        # data[:,a,:]=(R,S) -> .T=(S,R)
        slice_2d = data[:, i, :].T
        if flip_r:
            slice_2d = slice_2d[:, ::-1]  # 翻转R列
        return slice_2d
    
    def get_axial_slice(data, i):
        # data[:,:,s]=(R,A) -> .T=(A,R)
        slice_2d = data[:, :, i].T
        if flip_r:
            slice_2d = slice_2d[:, ::-1]  # 翻转R列
        return slice_2d
    
    # 定义三个轴位 (基于RAS+坐标系，符合标准医学图像方向)
    orientations = {
        'sagittal': {
            'axis': 0,
            'num_slices': mra_data.shape[0],
            'get_slice': get_sagittal_slice,
            'aspect': spacing_s / spacing_a,  # 行是S，列是A
            'figsize': (8, 8),
        },
        'coronal': {
            'axis': 1,
            'num_slices': mra_data.shape[1],
            'get_slice': get_coronal_slice,
            'aspect': spacing_s / spacing_r,  # 行是S，列是R
            'figsize': (8, 8),
        },
        'axial': {
            'axis': 2,
            'num_slices': mra_data.shape[2],
            'get_slice': get_axial_slice,
            'aspect': spacing_a / spacing_r,  # 行是A，列是R
            'figsize': (8, 8),
        },
    }
    
    for orient_name, orient_info in orientations.items():
        orient_dir = os.path.join(output_dir, orient_name)
        
        # 创建三个子文件夹
        mra_dir = os.path.join(orient_dir, "mra")
        heatmap_dir = os.path.join(orient_dir, "heatmap")
        overlay_dir = os.path.join(orient_dir, "overlay")
        os.makedirs(mra_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        
        num_slices = orient_info['num_slices']
        get_slice = orient_info['get_slice']
        figsize = orient_info['figsize']
        aspect = orient_info['aspect']
        
        print(f"[INFO] 正在导出 {num_slices} 个 {orient_name} 切片 (aspect={aspect:.3f})...")
        
        for i in range(num_slices):
            mra_slice = get_slice(mra_display, i)
            heatmap_slice = get_slice(heatmap_data, i)
            
            # 1. 保存MRA切片
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(mra_slice, cmap='gray', origin='lower', aspect=aspect)
            ax.axis('off')
            plt.savefig(os.path.join(mra_dir, f"{prefix}_{i:03d}.png"), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 2. 保存热力图切片
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(heatmap_slice, cmap=cmap, origin='lower', aspect=aspect,
                      norm=Normalize(vmin=0, vmax=1))
            ax.axis('off')
            plt.savefig(os.path.join(heatmap_dir, f"{prefix}_{i:03d}.png"), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 3. 保存叠加切片
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(mra_slice, cmap='gray', origin='lower', aspect=aspect)
            ax.imshow(heatmap_slice, cmap=cmap, alpha=alpha, origin='lower', aspect=aspect,
                      norm=Normalize(vmin=0, vmax=1))
            ax.axis('off')
            plt.savefig(os.path.join(overlay_dir, f"{prefix}_{i:03d}.png"), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 进度显示
            if (i + 1) % 50 == 0 or i == num_slices - 1:
                print(f"[INFO] {orient_name}: 已导出 {i + 1}/{num_slices} 切片")
    
    print(f"[INFO] 所有切片已导出到: {output_dir}")


def visualize_3d_heatmap(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_path: str,
    threshold: float = 0.3,
    alpha: float = 0.6,
    title: str = None,
):
    """
    生成3D可视化：MRA等值面 + 热力图高激活区域
    
    Args:
        mra_data: [D, H, W] MRA数据
        heatmap_data: [D, H, W] 热力图数据
        output_path: 输出路径 (.html 交互式 或 .png 静态)
        threshold: 热力图显示阈值 (0-1)
        alpha: 热力图透明度
        title: 图像标题
    """
    try:
        import pyvista as pv
        _visualize_3d_pyvista(mra_data, heatmap_data, output_path, threshold, alpha, title)
    except ImportError:
        print("[WARN] PyVista未安装，使用matplotlib生成3D视图")
        _visualize_3d_matplotlib(mra_data, heatmap_data, output_path, threshold, alpha, title)


def _visualize_3d_pyvista(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_path: str,
    threshold: float = 0.3,
    alpha: float = 0.6,
    title: str = None,
):
    """使用PyVista生成交互式3D可视化"""
    import pyvista as pv
    
    # MRA归一化
    mra_norm = mra_data.copy().astype(np.float32)
    if (mra_norm > 0).any():
        lo = np.percentile(mra_norm[mra_norm > 0], 5)
        hi = np.percentile(mra_norm[mra_norm > 0], 95)
        mra_norm = np.clip(mra_norm, lo, hi)
        mra_norm = (mra_norm - lo) / (hi - lo + 1e-8)
    
    # 创建网格 - 使用point_data而不是cell_data（contour需要point_data）
    grid_mra = pv.ImageData(dimensions=mra_data.shape)
    grid_mra.point_data["MRA"] = mra_norm.flatten(order="F")
    
    grid_heatmap = pv.ImageData(dimensions=heatmap_data.shape)
    grid_heatmap.point_data["Heatmap"] = heatmap_data.flatten(order="F")
    
    # 创建plotter
    plotter = pv.Plotter(off_screen=True)
    
    # MRA等值面（显示脑组织轮廓）
    mra_thresh = 0.2
    try:
        mra_contour = grid_mra.contour([mra_thresh], scalars="MRA")
        if mra_contour.n_points > 0:
            plotter.add_mesh(
                mra_contour,
                color="lightgray",
                opacity=0.15,
                smooth_shading=True,
            )
    except Exception:
        pass
    
    # 热力图等值面（高激活区域）- 红色
    try:
        contour = grid_heatmap.contour([threshold], scalars="Heatmap")
        if contour.n_points > 0:
            plotter.add_mesh(
                contour,
                color="red",
                opacity=alpha,
                smooth_shading=True,
            )
    except Exception:
        pass
    
    # 添加更高阈值的等值面（核心区域）- 黄色
    if threshold < 0.6:
        try:
            contour_high = grid_heatmap.contour([0.6], scalars="Heatmap")
            if contour_high.n_points > 0:
                plotter.add_mesh(
                    contour_high,
                    color="yellow",
                    opacity=0.9,
                    smooth_shading=True,
                )
        except Exception:
            pass
    
    # 最高激活区域 - 白色
    try:
        contour_max = grid_heatmap.contour([0.8], scalars="Heatmap")
        if contour_max.n_points > 0:
            plotter.add_mesh(
                contour_max,
                color="white",
                opacity=1.0,
                smooth_shading=True,
            )
    except Exception:
        pass
    
    if title:
        plotter.add_title(title, font_size=12)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.set_background('black')
    
    # 保存
    if output_path.endswith('.html'):
        plotter.export_html(output_path)
    else:
        plotter.screenshot(output_path)
    
    plotter.close()
    print(f"[INFO] 3D可视化已保存: {output_path}")


def _visualize_3d_matplotlib(
    mra_data: np.ndarray,
    heatmap_data: np.ndarray,
    output_path: str,
    threshold: float = 0.3,
    alpha: float = 0.6,
    title: str = None,
):
    """使用matplotlib生成3D正交切片视图"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    
    # MRA归一化
    mra_display = mra_data.copy()
    if (mra_display > 0).any():
        lo = np.percentile(mra_display[mra_display > 0], 1)
        hi = np.percentile(mra_display[mra_display > 0], 99)
        mra_display = np.clip(mra_display, lo, hi)
        mra_display = (mra_display - lo) / (hi - lo + 1e-8)
    
    D, H, W = mra_data.shape
    
    # 找到热力图最大激活位置
    max_pos = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    z_max, y_max, x_max = max_pos
    
    fig = plt.figure(figsize=(16, 12))
    
    # ===== 左上: 3D正交切片 =====
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # 创建坐标网格
    # Axial切片 (z = z_max)
    X_ax, Y_ax = np.meshgrid(np.arange(W), np.arange(H))
    Z_ax = np.ones_like(X_ax) * z_max
    ax1.plot_surface(X_ax, Y_ax, Z_ax, 
                     facecolors=plt.cm.gray(mra_display[z_max, :, :]),
                     alpha=0.7, shade=False)
    
    # Coronal切片 (y = y_max)
    X_cor, Z_cor = np.meshgrid(np.arange(W), np.arange(D))
    Y_cor = np.ones_like(X_cor) * y_max
    ax1.plot_surface(X_cor, Y_cor, Z_cor,
                     facecolors=plt.cm.gray(mra_display[:, y_max, :]),
                     alpha=0.7, shade=False)
    
    # Sagittal切片 (x = x_max)
    Y_sag, Z_sag = np.meshgrid(np.arange(H), np.arange(D))
    X_sag = np.ones_like(Y_sag) * x_max
    ax1.plot_surface(X_sag, Y_sag, Z_sag,
                     facecolors=plt.cm.gray(mra_display[:, :, x_max]),
                     alpha=0.7, shade=False)
    
    # 高激活区域散点
    high_activation = heatmap_data > threshold
    if high_activation.any():
        z_pts, y_pts, x_pts = np.where(high_activation)
        colors = heatmap_data[high_activation]
        scatter = ax1.scatter(x_pts, y_pts, z_pts, c=colors, cmap='hot', 
                             alpha=0.5, s=1, vmin=0, vmax=1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Orthogonal Slices + High Activation')
    
    # ===== 右上: 最大强度投影 (MIP) =====
    ax2 = fig.add_subplot(2, 2, 2)
    
    # 热力图MIP (沿z轴)
    heatmap_mip = np.max(heatmap_data, axis=0)
    mra_mip = np.max(mra_display, axis=0)
    
    ax2.imshow(mra_mip.T, cmap='gray', origin='lower')
    ax2.imshow(heatmap_mip.T, cmap='jet', alpha=0.5, origin='lower',
               norm=Normalize(vmin=0, vmax=1))
    ax2.set_title('Maximum Intensity Projection (Axial)')
    ax2.axis('off')
    
    # ===== 左下: Coronal MIP =====
    ax3 = fig.add_subplot(2, 2, 3)
    
    heatmap_mip_cor = np.max(heatmap_data, axis=1)
    mra_mip_cor = np.max(mra_display, axis=1)
    
    ax3.imshow(mra_mip_cor.T, cmap='gray', origin='lower')
    ax3.imshow(heatmap_mip_cor.T, cmap='jet', alpha=0.5, origin='lower',
               norm=Normalize(vmin=0, vmax=1))
    ax3.set_title('Maximum Intensity Projection (Coronal)')
    ax3.axis('off')
    
    # ===== 右下: Sagittal MIP =====
    ax4 = fig.add_subplot(2, 2, 4)
    
    heatmap_mip_sag = np.max(heatmap_data, axis=2)
    mra_mip_sag = np.max(mra_display, axis=2)
    
    ax4.imshow(mra_mip_sag.T, cmap='gray', origin='lower')
    ax4.imshow(heatmap_mip_sag.T, cmap='jet', alpha=0.5, origin='lower',
               norm=Normalize(vmin=0, vmax=1))
    ax4.set_title('Maximum Intensity Projection (Sagittal)')
    ax4.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 3D可视化已保存: {output_path}")
