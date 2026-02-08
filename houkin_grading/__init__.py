"""
Houkin Grading Module

Deep learning-based Houkin grade classification for Moyamoya Disease (MMD).

This module performs hemisphere-level classification of Houkin grades using
dual-channel 3D CNNs (MRA + Vessel Mask).

Classification Task:
- Grade 1 (G1): Mild stenosis
- Grade 2 (G2): Moderate stenosis
- Grade 3-4 (G3-4): Severe stenosis (combined due to data scarcity)

Architecture:
- DenseNet-121 3D (primary model)
- Dual-channel input: MRA (160x160x160) + Vessel Mask
- Focal Loss for class imbalance handling
"""

from .model_3d_densenet import densenet121_3d, DenseNet121_3D
from .model_3d_fusion import resnet3d10, resnet3d18, resnet3d34, resnet3d50, load_pretrained_weights
from .dataset_hemi_fusion import HemiFusionDataset, compute_sample_weights
from .utils_metrics import summarize_metrics

__all__ = [
    'densenet121_3d', 'DenseNet121_3D',
    'resnet3d10', 'resnet3d18', 'resnet3d34', 'resnet3d50',
    'load_pretrained_weights',
    'HemiFusionDataset', 'compute_sample_weights',
    'summarize_metrics',
]
