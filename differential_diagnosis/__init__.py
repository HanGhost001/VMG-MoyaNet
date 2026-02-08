"""
Differential Diagnosis Module

Deep learning-based differential diagnosis for distinguishing:
- MMD (Moyamoya Disease)
- ICAS (Intracranial Atherosclerotic Stenosis)
- NC (Normal Control)

This module performs full-brain level 3-class classification using
dual-channel 3D CNNs (MRA + Vessel Mask).

Architecture:
- DenseNet-121 3D (primary model)
- Dual-channel input: MRA (224x224x224) + Vessel Mask
- Focal Loss for class imbalance handling
"""

from .model_3d_densenet import densenet121_3d, DenseNet121_3D
from .dataset_full_fusion import FullFusionDataset, compute_sample_weights
from .utils_metrics import summarize_metrics

__all__ = [
    'densenet121_3d', 'DenseNet121_3D',
    'FullFusionDataset', 'compute_sample_weights',
    'summarize_metrics',
]
