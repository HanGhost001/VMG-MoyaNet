from __future__ import annotations

import csv
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
        x = np.pad(
            x,
            ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="constant",
            constant_values=0,
        )

    # crop center
    z, y, xw = x.shape
    z0 = max(0, (z - tz) // 2)
    y0 = max(0, (y - ty) // 2)
    x0 = max(0, (xw - tx) // 2)
    return x[z0 : z0 + tz, y0 : y0 + ty, x0 : x0 + tx]


def _normalize_mra(mra: np.ndarray, mask: np.ndarray, use_nonzero_norm: bool = False) -> np.ndarray:
    """
    归一化MRA图像
    
    Args:
        mra: MRA图像数组
        mask: Mask数组
        use_nonzero_norm: 是否使用非零像素归一化（与单通道的全局归一化一致）
            - False: 使用mask前景做稳健归一化（默认，原有方式）
            - True: 使用mra>0非零像素归一化
    """
    if use_nonzero_norm:
        # 使用非零像素归一化（与单通道的全局归一化一致，不使用mask前景）
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
    else:
        # 原有方式：使用mask前景做稳健归一化；若mask为空则全局z-score
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
    label3: int
    mra_path: str
    mask_path: str


class FullFusionDataset(Dataset):
    """
    读取 split CSV，返回 2通道输入：
      x: [2, D, H, W]  (mra, mask)
      y: int64

    期望CSV列：patient_id,label3,mra_path,mask_path
    """

    def __init__(
        self,
        split_csv: str,
        target_shape: Tuple[int, int, int] = (160, 160, 160),
        crop_margin_vox: int = 10,
        augment: bool = False,
        seed: int = 0,
        allow_flip_lr: bool = True,
        use_nonzero_normalization: bool = False,
    ):
        self.split_csv = split_csv
        self.target_shape = target_shape
        self.crop_margin_vox = int(crop_margin_vox)
        self.augment = augment
        self.allow_flip_lr = bool(allow_flip_lr)
        self.use_nonzero_normalization = bool(use_nonzero_normalization)
        self.rng = np.random.default_rng(seed)

        self.items: List[Item] = []
        with open(split_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(
                    Item(
                        patient_id=str(row["patient_id"]),
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
        # 全脑任务：默认允许沿LR轴翻转（与半脑不同）
        if self.allow_flip_lr and self.rng.random() < 0.5:
            mra = np.flip(mra, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
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
        return mra, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        
        # 判断文件类型：支持npy（预处理）和nii.gz（实时处理）
        if it.mra_path.endswith('.npy') and it.mask_path.endswith('.npy'):
            # 预处理好的npy文件：直接加载，已归一化和调整尺寸
            mra = np.load(it.mra_path).astype(np.float32)  # [D, H, W]
            mask = np.load(it.mask_path).astype(np.float32)  # [D, H, W]
        else:
            # 实时处理NIfTI文件（向后兼容）
            mra = _read_nii(it.mra_path)
            mask = _read_nii(it.mask_path)
            mask = (mask > 0.5).astype(np.uint8)

            # mask bbox crop（减背景稀释）
            mra, mask = _crop_with_margin(mra, mask, margin=self.crop_margin_vox)

            # pad/crop到固定shape
            mra = _pad_or_crop_to_shape(mra, self.target_shape)
            mask = _pad_or_crop_to_shape(mask, self.target_shape)

            # normalize
            mra = _normalize_mra(mra, mask, use_nonzero_norm=self.use_nonzero_normalization)
            mask = mask.astype(np.float32)

        if self.augment:
            mra, mask = self._augment(mra, mask)

        x = np.stack([mra, mask], axis=0).astype(np.float32)  # [2,D,H,W]
        y = np.int64(it.label3)
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.long),
        }


def compute_sample_weights(ds: FullFusionDataset, num_classes: int = 3) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for it in ds.items:
        if 0 <= int(it.label3) < num_classes:
            counts[int(it.label3)] += 1
    counts = np.maximum(counts, 1)
    weights_per_class = counts.sum() / (num_classes * counts.astype(np.float32))
    w = np.array([weights_per_class[int(it.label3)] for it in ds.items], dtype=np.float32)
    return torch.from_numpy(w)

