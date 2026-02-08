"""
Differential Diagnosis Ensemble Evaluation

Load multiple trained models and evaluate with averaged predictions.

Usage:
    python -m differential_diagnosis.eval_ensemble_full <run_dir1> <run_dir2> ... [test_csv]
"""
from __future__ import annotations

import os
import sys
import json
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset_full_fusion import FullFusionDataset
from .model_3d_densenet import densenet121_3d
from .model_3d_fusion import resnet3d10, resnet3d18, resnet3d34, resnet3d50
from .utils_metrics import summarize_metrics


def build_model_from_config(config: dict, device: torch.device) -> nn.Module:
    """Build model from saved config.json."""
    # Support both flat config (old) and nested config (YAML-based)
    if 'model' in config and isinstance(config['model'], dict):
        arch = config['model']['architecture']
        num_classes = config['task']['num_classes']
        in_channels = config['data']['in_channels']
        dropout = config['model']['dropout']
        growth_rate = config['model'].get('growth_rate', 56)
    else:
        arch = config.get('model_type', config.get('architecture', 'densenet121'))
        num_classes = config.get('num_classes', 3)
        in_channels = config.get('in_channels', 2)
        dropout = config.get('dropout', 0.5)
        growth_rate = config.get('growth_rate', 56)
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


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    config_path = checkpoint_path.replace("best.pt", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate_ensemble(models: List[nn.Module], test_loader: DataLoader,
                      device: torch.device, num_classes: int) -> Dict[str, Any]:
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            model_probs = []
            for model in models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu().numpy())
            avg_probs = np.mean(model_probs, axis=0)
            all_probs.append(avg_probs)
            all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(probs, axis=1)
    return summarize_metrics(labels, preds, num_classes=num_classes)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m differential_diagnosis.eval_ensemble_full <run_dir1> [run_dir2] ... [--test_csv PATH]")
        sys.exit(1)

    # Parse arguments
    run_dirs = []
    test_csv = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--test_csv' and i + 1 < len(sys.argv):
            test_csv = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].endswith('.csv'):
            test_csv = sys.argv[i]
            i += 1
        else:
            run_dirs.append(sys.argv[i])
            i += 1

    if not run_dirs:
        print("Error: 至少需要一个 run_dir")
        sys.exit(1)

    # Auto-detect test_csv from first config
    if test_csv is None:
        config_path = os.path.join(run_dirs[0], "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if 'paths' in config:
            test_csv = config['paths'].get('test_csv', '')
        else:
            splits_dir = config.get('splits_dir', '')
            test_csv = os.path.join(splits_dir, "test.csv")

    if not os.path.exists(test_csv):
        print(f"Error: 测试集不存在: {test_csv}")
        sys.exit(1)

    print(f"[INFO] 加载 {len(run_dirs)} 个模型")
    print(f"[INFO] 测试集: {test_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    models = []
    for run_dir in run_dirs:
        ckpt_path = os.path.join(run_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] 跳过 {run_dir}: best.pt 不存在")
            continue
        print(f"[INFO] 加载: {run_dir}")
        models.append(load_model_from_checkpoint(ckpt_path, device))

    if not models:
        print("Error: 没有成功加载任何模型")
        sys.exit(1)

    # Load dataset config from first run
    with open(os.path.join(run_dirs[0], "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if 'data' in cfg and isinstance(cfg['data'], dict):
        target_shape = tuple(cfg['data']['input_shape'])
        crop_margin = int(cfg['data']['crop_margin_vox'])
        allow_flip = cfg['data'].get('allow_flip_lr', True)
        use_nonzero = cfg['data'].get('use_nonzero_normalization', False)
        num_classes = cfg['task']['num_classes']
    else:
        target_shape = tuple(cfg.get('target_shape', [224, 224, 224]))
        crop_margin = int(cfg.get('crop_margin_vox', 10))
        allow_flip = cfg.get('allow_flip_lr', True)
        use_nonzero = cfg.get('use_nonzero_normalization', False)
        num_classes = cfg.get('num_classes', 3)

    test_ds = FullFusionDataset(
        test_csv, target_shape=target_shape, crop_margin_vox=crop_margin,
        augment=False, seed=0, allow_flip_lr=allow_flip,
        use_nonzero_normalization=use_nonzero,
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print("[INFO] 开始集成评估...")
    metrics = evaluate_ensemble(models, test_loader, device, num_classes)

    output_path = "ensemble_test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[RESULTS]")
    print(f"  Acc:      {metrics['acc']:.3f}")
    print(f"  Bal Acc:  {metrics['bal_acc']:.3f}")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    main()
