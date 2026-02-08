"""
Vessel Segmentation Module

This module provides brain vessel segmentation using the COSTA model,
which is based on nnUNet with Swin Transformer for cerebral vessel extraction.

The segmentation pipeline includes:
1. Skull stripping using BET2 (FSL)
2. Histogram standardization for intensity normalization
3. COSTA model inference for vessel segmentation

Reference:
- COSTA: https://github.com/iMED-Lab/COSTA
- nnUNet: https://github.com/MIC-DKFZ/nnUNet
"""

from .segment_vessels import run_vessel_segmentation

__all__ = ['run_vessel_segmentation']
