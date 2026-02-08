from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Tuple[float, float]:
    acc = float(np.trace(cm) / (cm.sum() + 1e-9))
    per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-9)
    bal_acc = float(per_class.mean())
    return acc, bal_acc


def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-9)
    return float(np.nanmean(f1))


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    acc, bal_acc = metrics_from_cm(cm)
    macro_f1 = macro_f1_from_cm(cm)
    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "cm": cm.tolist(),
    }
