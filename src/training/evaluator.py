"""Evaluation Engine for Brain Tumor MRI Classification: Multi-class metrics, confusion matrices, and latency tracking."""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4
) -> Dict[str, Any]:
    """
    Computes multi-class classification metrics: Accuracy, Macro Precision/Recall/F1,
    per-class Precision/Recall/F1, and Confusion Matrix.
    """
    # Build confusion matrix: rows = true, cols = pred
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total_samples = len(y_true)
    correct_samples = np.trace(cm)
    accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1)

    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(per_class_f1))

    per_class_dict = {}
    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        per_class_dict[name] = {
            "precision": round(per_class_precision[c], 4),
            "recall": round(per_class_recall[c], 4),
            "f1_score": round(per_class_f1[c], 4),
            "support": int(np.sum(cm[c, :]))
        }

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "total_samples": total_samples
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates model performance on a given DataLoader.
    
    Returns:
        Dict containing accuracy, macro & per-class metrics, confusion matrix, and inference latency.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    total_eval_time = time.perf_counter() - start_time
    total_samples = len(all_targets)
    latency_ms_per_sample = float((total_eval_time / total_samples) * 1000) if total_samples > 0 else 0.0

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    metrics["inference_time_seconds"] = round(total_eval_time, 4)
    metrics["latency_ms_per_sample"] = round(latency_ms_per_sample, 4)

    return metrics
