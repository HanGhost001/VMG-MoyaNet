from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def _read_nii(path: str) -> np.ndarray:
    nii = nib.load(path)
    x = nii.get_fdata(dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _bbox_3d(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def _crop_with_margin(img: np.ndarray, msk: np.ndarray, margin: int) -> Tuple[np.ndarray, np.ndarray]:
    bb = _bbox_3d(msk)
    if bb is None:
        return img, msk
    zmin, zmax, ymin, ymax, xmin, xmax = bb
    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)
    zmax = min(img.shape[0] - 1, zmax + margin)
    ymax = min(img.shape[1] - 1, ymax + margin)
    xmax = min(img.shape[2] - 1, xmax + margin)
    sl = (slice(zmin, zmax + 1), slice(ymin, ymax + 1), slice(xmin, xmax + 1))
    return img[sl], msk[sl]


def _pad_or_crop_to_shape(x: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    tz, ty, tx = target
    z, y, xw = x.shape

    # pad
    pad_z0 = max(0, (tz - z) // 2)
    pad_z1 = max(0, tz - z - pad_z0)
    pad_y0 = max(0, (ty - y) // 2)
    pad_y1 = max(0, ty - y - pad_y0)
    pad_x0 = max(0, (tx - xw) // 2)
    pad_x1 = max(0, tx - xw - pad_x0)

    if pad_z0 or pad_z1 or pad_y0 or pad_y1 or pad_x0 or pad_x1:
        x = np.pad(x, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant", constant_values=0)

    # crop center
    z, y, xw = x.shape
    z0 = max(0, (z - tz) // 2)
    y0 = max(0, (y - ty) // 2)
    x0 = max(0, (xw - tx) // 2)
    return x[z0 : z0 + tz, y0 : y0 + ty, x0 : x0 + tx]


def _histogram_match_mra(mra: np.ndarray, mask: np.ndarray, 
                         reference_bins: np.ndarray, 
                         reference_cdf: np.ndarray) -> np.ndarray:
    """
    使用直方图匹配将mra匹配到参考直方图
    
    Args:
        mra: MRA图像
        mask: Mask（用于提取前景）
        reference_bins: 参考直方图的bins (shape: [n_bins+1])
        reference_cdf: 参考直方图的累积分布函数 (shape: [n_bins])
    
    Returns:
        匹配后的MRA图像
    """
    # 提取前景区域
    fg = mra[mask > 0]
    if fg.size < 32:
        return mra  # 如果前景太小，不进行匹配
    
    # 1%-99%分位数clip（与normalize一致）
    lo = np.quantile(fg, 0.01)
    hi = np.quantile(fg, 0.99)
    fg_clipped = np.clip(fg, lo, hi)
    
    # 计算当前图像的直方图和CDF
    src_hist, src_bins = np.histogram(fg_clipped, bins=len(reference_bins)-1, 
                                      range=(reference_bins[0], reference_bins[-1]))
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / (src_cdf[-1] + 1e-10)  # 归一化
    
    # 直方图匹配：对于源图像的每个像素值，找到对应的参考值
    # 使用线性插值
    mra_matched = mra.copy()
    fg_flat = mra_matched[mask > 0]
    
    # Clip到参考范围内
    fg_flat = np.clip(fg_flat, reference_bins[0], reference_bins[-1])
    
    # 对每个像素值，找到其在源CDF中的位置，然后映射到参考直方图
    # 使用np.interp进行双向插值
    src_bin_centers = (src_bins[:-1] + src_bins[1:]) / 2
    ref_bin_centers = (reference_bins[:-1] + reference_bins[1:]) / 2
    
    # 源值 -> 源CDF -> 参考CDF -> 参考值
    fg_cdf_values = np.interp(fg_flat, src_bin_centers, src_cdf)
    fg_matched = np.interp(fg_cdf_values, reference_cdf, ref_bin_centers)
    
    # 将匹配后的值写回
    mra_matched[mask > 0] = fg_matched
    
    return mra_matched.astype(np.float32)


def _normalize_mra(mra: np.ndarray, mask: np.ndarray, 
                   use_global_norm: bool = False,
                   global_mean: Optional[float] = None,
                   global_std: Optional[float] = None,
                   use_nonzero_norm: bool = False) -> np.ndarray:
    """
    归一化MRA图像
    
    Args:
        mra: MRA图像数组
        mask: Mask数组
        use_global_norm: 是否使用全局标准化（推荐用于消除域偏移）
        global_mean: 全局均值（use_global_norm=True时必需）
        global_std: 全局标准差（use_global_norm=True时必需）
        use_nonzero_norm: 是否使用非零像素归一化（类似单通道的全局归一化）
    
    Returns:
        归一化后的MRA图像
    """
    if use_nonzero_norm:
        # 使用非零像素归一化（类似单通道的全局归一化，不使用mask前景）
        non_zero = mra[mra > 0]
        if non_zero.size >= 32:
            # 使用非零像素计算均值和标准差
            mean = float(non_zero.mean())
            std = float(non_zero.std() + 1e-6)
            mra = (mra - mean) / std
        else:
            # 如果没有足够的非零像素，使用全局均值和标准差
            mean = float(mra.mean())
            std = float(mra.std() + 1e-6)
            mra = (mra - mean) / std
    elif use_global_norm:
        # 全局标准化：使用预先计算的训练集统计量
        if global_mean is None or global_std is None:
            raise ValueError("use_global_norm=True requires global_mean and global_std")
        
        # 仍然进行1%-99%分位数clip（去除异常值）
        fg = mra[mask > 0]
        if fg.size >= 32:
            lo = np.quantile(fg, 0.01)
            hi = np.quantile(fg, 0.99)
            mra = np.clip(mra, lo, hi)
        
        # 使用全局统计量标准化
        mra = (mra - global_mean) / (global_std + 1e-6)
    else:
        # Per-sample标准化：使用当前样本的前景统计量
        fg = mra[mask > 0]
        if fg.size >= 32:
            lo = np.quantile(fg, 0.01)
            hi = np.quantile(fg, 0.99)
            mra = np.clip(mra, lo, hi)
            mean = float(fg.mean())
            std = float(fg.std() + 1e-6)
            mra = (mra - mean) / std
        else:
            mean = float(mra.mean())
            std = float(mra.std() + 1e-6)
            mra = (mra - mean) / std
    
    return mra.astype(np.float32)


@dataclass
class Item:
    patient_id: str
    hemi: str
    label3: int
    mra_path: str
    mask_path: str


class HemiFusionDataset(Dataset):
    """
    读取 split CSV（来自 scripts/02_make_splits_patient_grouped.py 输出），返回 2通道输入：
      x: [2, D, H, W]  (mra, mask)
      y: int64
    """

    def __init__(
        self,
        split_csv: str,
        target_shape: Tuple[int, int, int] = (160, 160, 160),
        crop_margin_vox: int = 10,
        augment: bool = False,
        seed: int = 0,
        use_histogram_matching: bool = False,
        reference_histogram_bins: Optional[np.ndarray] = None,
        reference_histogram_cdf: Optional[np.ndarray] = None,
        use_global_normalization: bool = False,
        global_mean: Optional[float] = None,
        global_std: Optional[float] = None,
        use_nonzero_normalization: bool = False,
    ):
        self.split_csv = split_csv
        self.target_shape = target_shape
        self.crop_margin_vox = int(crop_margin_vox)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.use_histogram_matching = use_histogram_matching
        self.reference_histogram_bins = reference_histogram_bins
        self.reference_histogram_cdf = reference_histogram_cdf
        self.use_global_normalization = use_global_normalization
        self.global_mean = global_mean
        self.global_std = global_std
        self.use_nonzero_normalization = use_nonzero_normalization
        
        if use_histogram_matching and (reference_histogram_bins is None or reference_histogram_cdf is None):
            raise ValueError("use_histogram_matching=True requires reference_histogram_bins and reference_histogram_cdf")
        if use_global_normalization and (global_mean is None or global_std is None):
            raise ValueError("use_global_normalization=True requires global_mean and global_std")
        if use_nonzero_normalization and use_global_normalization:
            raise ValueError("use_nonzero_normalization and use_global_normalization cannot be both True")

        self.items: List[Item] = []
        with open(split_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(
                    Item(
                        patient_id=str(row["patient_id"]),
                        hemi=str(row["hemi"]),
                        label3=int(row["label3"]),
                        mra_path=str(row["mra_path"]),
                        mask_path=str(row["mask_path"]),
                    )
                )
        if len(self.items) == 0:
            raise ValueError(f"empty split csv: {split_csv}")

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, mra: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强（增强版：根据用户建议增加RandomGammaCorrection等医学影像常用增强）
        - 翻转：Y轴和Z轴翻转（各50%概率）
        - 强度扰动：轻微缩放和偏移（30%概率）
        - Gamma校正：随机Gamma校正（40%概率，医学影像常用）
        - 噪声：轻微高斯噪声（20%概率）
        """
        # 注意：半脑已对齐到同一侧坐标系，仍建议避免沿LR轴再翻转（会引入镜像差异）。
        # 仅对另外两轴做翻转增强。
        if self.rng.random() < 0.5:
            mra = np.flip(mra, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.rng.random() < 0.5:
            mra = np.flip(mra, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        # 轻微强度扰动
        if self.rng.random() < 0.3:
            scale = float(self.rng.uniform(0.9, 1.1))
            shift = float(self.rng.uniform(-0.1, 0.1))
            mra = mra * scale + shift
        
        # RandomGammaCorrection（医学影像常用，增加G2样本的变体）
        if self.rng.random() < 0.4:
            gamma = float(self.rng.uniform(0.8, 1.2))
            # 归一化到[0, 1]范围进行gamma校正
            mra_min, mra_max = mra.min(), mra.max()
            if mra_max > mra_min:
                mra_norm = (mra - mra_min) / (mra_max - mra_min + 1e-6)
                mra_norm = np.power(mra_norm, gamma)
                mra = mra_norm * (mra_max - mra_min) + mra_min
        
        # 轻微高斯噪声（增加鲁棒性）
        if self.rng.random() < 0.2:
            noise_std = float(self.rng.uniform(0.01, 0.05))
            noise = self.rng.normal(0, noise_std, size=mra.shape).astype(np.float32)
            mra = mra + noise
        
        return mra, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        mra = _read_nii(it.mra_path)
        mask = _read_nii(it.mask_path)
        mask = (mask > 0.5).astype(np.uint8)

        # mask bbox crop（减背景稀释）
        mra, mask = _crop_with_margin(mra, mask, margin=self.crop_margin_vox)

        # pad/crop到固定shape
        mra = _pad_or_crop_to_shape(mra, self.target_shape)
        mask = _pad_or_crop_to_shape(mask, self.target_shape)

        # histogram matching (在normalize之前)
        if self.use_histogram_matching:
            mra = _histogram_match_mra(mra, mask, 
                                      self.reference_histogram_bins,
                                      self.reference_histogram_cdf)

        # normalize
        mra = _normalize_mra(mra, mask, 
                            use_global_norm=self.use_global_normalization,
                            global_mean=self.global_mean,
                            global_std=self.global_std,
                            use_nonzero_norm=self.use_nonzero_normalization)
        mask = mask.astype(np.float32)

        if self.augment:
            mra, mask = self._augment(mra, mask)

        x = np.stack([mra, mask], axis=0).astype(np.float32)  # [2,D,H,W]
        y = np.int64(it.label3)

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.long),
        }


def apply_tta_transforms(x: torch.Tensor) -> list:
    """
    应用测试时增强（TTA）变换
    返回3种变换：原图、Y轴翻转、轻微Gamma校正
    
    注意：此函数与训练时的数据增强逻辑保持一致：
    - 训练时使用Y轴和Z轴翻转（axis=1和axis=2），TTA使用Y轴翻转（axis=1）
    - 训练时使用随机Gamma校正（0.8-1.2），TTA使用固定Gamma=0.95（轻微变亮）
    
    Args:
        x: 输入张量 [B, C, D, H, W]，其中C=2（mra, mask）
    
    Returns:
        list of transformed tensors (same shape as x)
    """
    transforms = []
    
    # 1. 原图
    transforms.append(x)
    
    # 2. Y轴翻转（与训练时的翻转增强一致，axis=1对应H维度）
    # x的形状是[B, C, D, H, W]，所以axis=3对应H维度（Y轴）
    x_flip_y = torch.flip(x, dims=[3])
    transforms.append(x_flip_y)
    
    # 3. 轻微Gamma校正（gamma=0.95，轻微变亮，与训练时的Gamma校正逻辑一致）
    # 只对MRA通道（channel 0）进行Gamma校正，mask通道（channel 1）保持不变
    x_gamma = x.clone()
    mra_channel = x_gamma[:, 0:1, :, :, :]  # [B, 1, D, H, W]
    mask_channel = x_gamma[:, 1:2, :, :, :]  # [B, 1, D, H, W]
    
    # 归一化MRA通道到[0, 1]范围
    mra_min = mra_channel.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0]
    mra_max = mra_channel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
    mra_normalized = (mra_channel - mra_min) / (mra_max - mra_min + 1e-8)
    
    # Gamma校正（gamma=0.95，轻微变亮）
    gamma = 0.95
    mra_gamma_corrected = torch.pow(mra_normalized, gamma)
    
    # 恢复原始范围
    mra_corrected = mra_gamma_corrected * (mra_max - mra_min) + mra_min
    
    # 重新组合
    x_gamma = torch.cat([mra_corrected, mask_channel], dim=1)
    transforms.append(x_gamma)
    
    return transforms


def compute_sample_weights(ds: HemiFusionDataset, num_classes: int = 3) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for it in ds.items:
        if 0 <= int(it.label3) < num_classes:
            counts[int(it.label3)] += 1
    counts = np.maximum(counts, 1)
    weights_per_class = counts.sum() / (num_classes * counts.astype(np.float32))
    w = np.array([weights_per_class[int(it.label3)] for it in ds.items], dtype=np.float32)
    return torch.from_numpy(w)
