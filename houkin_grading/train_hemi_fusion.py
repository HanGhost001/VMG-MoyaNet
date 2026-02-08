"""
Houkin Grading Training Script (YAML-driven)

Train a DenseNet121 (or ResNet) model for hemisphere-level Houkin grade classification.

Usage:
    python -m houkin_grading.train_hemi_fusion --config configs/houkin_grading.yaml
"""
from __future__ import annotations

import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset_hemi_fusion import HemiFusionDataset, compute_sample_weights
from .model_3d_densenet import densenet121_3d
from .model_3d_fusion import resnet3d10, resnet3d18, resnet3d34, resnet3d50, load_pretrained_weights
from .utils_metrics import summarize_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, num_classes=3, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha] * num_classes, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build model based on config."""
    arch = cfg['model']['architecture']
    num_classes = cfg['task']['num_classes']
    in_channels = cfg['data']['in_channels']
    dropout = cfg['model']['dropout']

    if arch == 'densenet121':
        growth_rate = cfg['model'].get('growth_rate', 56)
        model = densenet121_3d(
            num_classes=num_classes,
            in_channels=in_channels,
            growth_rate=growth_rate,
            dropout=dropout,
        )
        print(f"[INFO] 使用 DenseNet-121 3D (growth_rate={growth_rate}, in_channels={in_channels})")
    elif arch == 'resnet3d10':
        model = resnet3d10(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
        print(f"[INFO] 使用 ResNet-10 3D (in_channels={in_channels})")
    elif arch == 'resnet3d18':
        model = resnet3d18(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
        print(f"[INFO] 使用 ResNet-18 3D (in_channels={in_channels})")
    elif arch == 'resnet3d34':
        model = resnet3d34(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
        print(f"[INFO] 使用 ResNet-34 3D (in_channels={in_channels})")
    elif arch == 'resnet3d50':
        model = resnet3d50(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
        print(f"[INFO] 使用 ResNet-50 3D (in_channels={in_channels})")
    else:
        raise ValueError(f"不支持的模型架构: {arch}")

    model = model.to(device)

    # Load pretrained weights (ResNet only)
    if cfg['model'].get('use_pretrained', False) and arch.startswith('resnet'):
        pretrained_path = cfg['model'].get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            model = load_pretrained_weights(model, pretrained_path, in_channels=in_channels, strict=False)
        else:
            print(f"[WARN] 预训练权重不存在: {pretrained_path}，使用随机初始化")

    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, Any]:
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))
            ys.append(y.detach().cpu().numpy())
            ps.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.zeros((0,), dtype=np.int64)
    m = summarize_metrics(y_true, y_pred, num_classes=num_classes)
    m["loss"] = loss_sum / max(1, n)
    return m


def main():
    parser = argparse.ArgumentParser(description="Train Houkin grading model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Extract config values
    seed = cfg['training']['seed']
    set_seed(seed)

    num_classes = cfg['task']['num_classes']
    target_shape = tuple(cfg['data']['input_shape'])
    crop_margin_vox = cfg['data']['crop_margin_vox']
    use_nonzero = cfg['data'].get('use_nonzero_normalization', False)

    batch_size = cfg['training']['batch_size']
    grad_accum = cfg['training'].get('gradient_accumulation_steps', 1)
    epochs = cfg['training']['epochs']
    patience_limit = cfg['training']['patience']
    lr = cfg['training']['optimizer']['lr']
    wd = cfg['training']['optimizer']['weight_decay']

    use_sampler = cfg['training'].get('use_weighted_sampler', True)
    use_amp = cfg['training'].get('use_amp', True)
    augment = cfg['data'].get('use_augmentation', True)

    num_workers = cfg['dataloader'].get('num_workers', 4)
    persistent_workers = cfg['dataloader'].get('persistent_workers', True)
    pin_memory = cfg['dataloader'].get('pin_memory', True)
    prefetch_factor = cfg['dataloader'].get('prefetch_factor', 2)

    # Paths
    paths = cfg['paths']
    train_csv = paths['train_csv']
    val_csv = paths['val_csv']
    test_csv = paths.get('test_csv')
    output_dir = paths['output_dir']

    has_test = test_csv is not None and os.path.exists(test_csv)

    # Run directory
    run_tag = f"{cfg['task']['name']}_{cfg['model']['architecture']}"
    run_name = f"{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Datasets
    train_ds = HemiFusionDataset(
        train_csv, target_shape=target_shape, crop_margin_vox=crop_margin_vox,
        augment=augment, seed=seed, use_nonzero_normalization=use_nonzero,
    )
    val_ds = HemiFusionDataset(
        val_csv, target_shape=target_shape, crop_margin_vox=crop_margin_vox,
        augment=False, seed=seed, use_nonzero_normalization=use_nonzero,
    )
    test_ds = HemiFusionDataset(
        test_csv, target_shape=target_shape, crop_margin_vox=crop_margin_vox,
        augment=False, seed=0, use_nonzero_normalization=use_nonzero,
    ) if has_test else None

    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds) if test_ds else 'N/A'}")

    # Sampler
    if use_sampler:
        weights = compute_sample_weights(train_ds, num_classes=num_classes)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    def make_loader(ds, train: bool):
        return DataLoader(
            ds, batch_size=batch_size,
            shuffle=(shuffle and train),
            sampler=(sampler if train else None),
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=False,
        )

    train_loader = make_loader(train_ds, train=True)
    val_loader = make_loader(val_ds, train=False)
    test_loader = make_loader(test_ds, train=False) if test_ds else None

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(cfg, device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loss
    loss_cfg = cfg['training']['loss']
    if loss_cfg['type'] == 'focal':
        criterion = FocalLoss(
            alpha=loss_cfg.get('focal_alpha'),
            gamma=loss_cfg.get('focal_gamma', 2.0),
            num_classes=num_classes,
        )
        print(f"[INFO] Focal Loss (alpha={loss_cfg.get('focal_alpha')}, gamma={loss_cfg.get('focal_gamma')})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[INFO] CrossEntropyLoss")

    # LR Scheduler
    sched_cfg = cfg['training'].get('lr_scheduler', {})
    sched_type = sched_cfg.get('type', 'cosine')
    use_warmup = sched_cfg.get('warmup_epochs', 0) > 0
    warmup_epochs = sched_cfg.get('warmup_epochs', 0)
    warmup_start_lr = sched_cfg.get('warmup_start_lr', 1e-6)
    effective_t_max = sched_cfg.get('t_max', epochs) - warmup_epochs

    scheduler = None
    if sched_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, effective_t_max), eta_min=sched_cfg.get('eta_min', 2e-6))
        print(f"[INFO] CosineAnnealingLR (T_max={effective_t_max})")
    elif sched_type == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    if use_warmup:
        print(f"[INFO] Warmup: {warmup_epochs} epochs, {warmup_start_lr:.2e} -> {lr:.2e}")

    # Training loop
    best_val, best_bal_acc, best_loss = -1.0, -1.0, float('inf')
    best_path = os.path.join(run_dir, "best.pt")
    history_path = os.path.join(run_dir, "history.jsonl")
    patience = 0

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        loss_sum, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum

            scaler.scale(loss).backward()
            loss_sum += float(loss.item() * grad_accum) * int(y.size(0))
            n += int(y.size(0))

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        train_loss = loss_sum / max(1, n)

        # Validate
        val_m = evaluate(model, val_loader, device, num_classes)

        # LR update
        if use_warmup and epoch <= warmup_epochs:
            warmup_lr = warmup_start_lr + (lr - warmup_start_lr) * (epoch - 1) / max(1, warmup_epochs - 1)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            if scheduler is not None:
                if sched_type == 'cosine':
                    scheduler.step()
                elif sched_type == 'plateau':
                    scheduler.step(val_m['bal_acc'])
            current_lr = optimizer.param_groups[0]['lr']

        # Log
        epoch_s = time.perf_counter() - epoch_t0
        record = {"epoch": epoch, "train_loss": train_loss, "val": val_m, "lr": current_lr}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.3f} val_bal_acc={val_m['bal_acc']:.3f} "
            f"val_f1={val_m['macro_f1']:.3f} | lr={current_lr:.2e} | {epoch_s:.1f}s",
            flush=True,
        )

        # Best model / early stopping
        should_update = False
        if val_m["acc"] > best_val:
            should_update = True
        elif val_m["acc"] == best_val:
            if val_m["bal_acc"] > best_bal_acc:
                should_update = True
            elif val_m["bal_acc"] == best_bal_acc and val_m["loss"] < best_loss:
                should_update = True

        if should_update:
            best_val = float(val_m["acc"])
            best_bal_acc = float(val_m["bal_acc"])
            best_loss = float(val_m["loss"])
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val,
                "best_val_bal_acc": best_bal_acc,
                "best_val_loss": best_loss,
                "config": cfg,
            }, best_path)
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"[EARLY STOP] epoch={epoch} best_val_acc={best_val:.3f} best_bal_acc={best_bal_acc:.3f}")
                break

    # Final evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if test_loader is not None:
        test_m = evaluate(model, test_loader, device, num_classes)
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_m, f, indent=2, ensure_ascii=False)
        print(f"[TEST] acc={test_m['acc']:.3f} bal_acc={test_m['bal_acc']:.3f} f1={test_m['macro_f1']:.3f}")
    else:
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"note": "test set not available"}, f, indent=2, ensure_ascii=False)
        print("[INFO] 跳过测试集评估")

    print(f"[DONE] run_dir={run_dir}")


if __name__ == "__main__":
    main()
