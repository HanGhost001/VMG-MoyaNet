"""
Houkin Grading Evaluation Script

Evaluate a trained model checkpoint on the test set.

Usage:
    python -m houkin_grading.eval_hemi_fusion --run_dir <path_to_run>
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset_hemi_fusion import HemiFusionDataset
from .model_3d_densenet import densenet121_3d
from .model_3d_fusion import resnet3d10, resnet3d18, resnet3d34, resnet3d50
from .utils_metrics import summarize_metrics


def build_model_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    """Build model from saved config.json."""
    # Support both flat config (old) and nested config (YAML-based)
    if 'model' in config and isinstance(config['model'], dict):
        # New YAML-based config
        arch = config['model']['architecture']
        num_classes = config['task']['num_classes']
        in_channels = config['data']['in_channels']
        dropout = config['model']['dropout']
        growth_rate = config['model'].get('growth_rate', 56)
    else:
        # Legacy flat config
        arch = config.get('model_type', config.get('architecture', 'resnet3d18'))
        num_classes = config.get('num_classes', 3)
        in_channels = config.get('in_channels', 2)
        dropout = config.get('dropout', 0.5)
        growth_rate = config.get('growth_rate', 56)
        # Map legacy model_depth to architecture name
        if arch in ('resnet18', 'resnet') or config.get('model_depth') == 18:
            arch = 'resnet3d18'
        elif config.get('model_depth') == 10:
            arch = 'resnet3d10'
        elif config.get('model_depth') == 34:
            arch = 'resnet3d34'
        elif config.get('model_depth') == 50:
            arch = 'resnet3d50'

    if arch == 'densenet121':
        model = densenet121_3d(num_classes=num_classes, in_channels=in_channels,
                               growth_rate=growth_rate, dropout=dropout)
    elif arch == 'resnet3d10':
        model = resnet3d10(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    elif arch == 'resnet3d18':
        model = resnet3d18(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    elif arch == 'resnet3d34':
        model = resnet3d34(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    elif arch == 'resnet3d50':
        model = resnet3d50(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    else:
        raise ValueError(f"不支持的模型架构: {arch}")

    return model.to(device)


def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device,
          num_classes: int = 3) -> Dict[str, Any]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            ys.append(y.detach().cpu().numpy())
            ps.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.zeros((0,), dtype=np.int64)
    return summarize_metrics(y_true, y_pred, num_classes=num_classes)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Houkin grading model")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--test_csv", type=str, default=None, help="Override test CSV path")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(config_path, "r", encoding="utf-8"))

    # Determine test CSV
    if args.test_csv:
        test_csv = args.test_csv
    elif 'paths' in cfg:
        test_csv = cfg['paths'].get('test_csv', '')
    else:
        test_csv = os.path.join(cfg.get('splits_dir', ''), "test.csv")

    if not os.path.exists(test_csv):
        print(f"[ERROR] 测试集不存在: {test_csv}")
        return

    # Extract data config
    if 'data' in cfg and isinstance(cfg['data'], dict):
        target_shape = tuple(cfg['data']['input_shape'])
        crop_margin = int(cfg['data']['crop_margin_vox'])
        use_nonzero = cfg['data'].get('use_nonzero_normalization', False)
        num_classes = cfg['task']['num_classes']
        num_workers = cfg.get('dataloader', {}).get('num_workers', 4)
        batch_size = cfg['training']['batch_size']
        seed = cfg['training']['seed']
    else:
        target_shape = tuple(cfg.get('target_shape', [160, 160, 160]))
        crop_margin = int(cfg.get('crop_margin_vox', 10))
        use_nonzero = cfg.get('use_nonzero_normalization', False)
        num_classes = cfg.get('num_classes', 3)
        num_workers = cfg.get('num_workers', 4)
        batch_size = cfg.get('batch_size', 4)
        seed = cfg.get('seed', 0)

    ds = HemiFusionDataset(
        test_csv, target_shape=target_shape, crop_margin_vox=crop_margin,
        augment=False, seed=seed, use_nonzero_normalization=use_nonzero,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, device)

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    m = infer(model, loader, device, num_classes)
    out_path = os.path.join(run_dir, "test_metrics_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] {m}")
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
