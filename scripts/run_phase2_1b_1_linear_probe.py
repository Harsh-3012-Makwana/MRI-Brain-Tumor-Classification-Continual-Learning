"""Execution script for Phase 2.1B-1: ResNet18 Linear Probe Baseline Training & Dual Evaluation."""

import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline_factory import get_mri_dataloaders
from src.models.resnet18_baseline import build_resnet18_baseline
from src.training.evaluator import evaluate_model
from src.training.trainer import ModelTrainer

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def main():
    print("=" * 70)
    print("PHASE 2.1B-1: RESNET18 LINEAR PROBE BASELINE TRAINING")
    print("=" * 70)

    # 1. Setup device and random seeds
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")

    print(f"Target Device: {device}")

    # 2. Get DataLoaders (num_workers=0 for fast single-thread loading on Windows)
    print("\n[Pipeline] Constructing PyTorch DataLoaders...")
    dataloaders = get_mri_dataloaders(
        target_size=(224, 224),
        padding=10,
        batch_size=32,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        seed=seed
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    benchmark_loader = dataloaders["benchmark_test"]
    filtered_loader = dataloaders["filtered_test"]

    print(f"  - Train Samples          : {len(train_loader.dataset):,}")
    print(f"  - Val Samples            : {len(val_loader.dataset):,}")
    print(f"  - Benchmark Test Samples : {len(benchmark_loader.dataset):,}")
    print(f"  - Filtered Test Samples  : {len(filtered_loader.dataset):,}")

    # 3. Instantiate Model (Linear Probe: Frozen Backbone)
    print("\n[Model] Building ResNet18 (IMAGENET1K_V1, Linear Probe)...")
    model = build_resnet18_baseline(strategy="linear_probe", pretrained=True)
    param_counts = model.get_parameter_counts()
    print(f"  - Total Parameters      : {param_counts['total_parameters']:,}")
    print(f"  - Trainable Parameters  : {param_counts['trainable_parameters']:,}")
    print(f"  - Frozen Parameters     : {param_counts['frozen_parameters']:,}")

    # 4. Setup Optimizer & Trainer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.0001
    )
    
    checkpoint_dir = Path("artifacts/checkpoints/resnet18_baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        max_epochs=25,
        patience=7,
        checkpoint_dir=checkpoint_dir,
        seed=seed
    )

    # 5. Run Training Loop
    print("\n[Training] Starting Linear Probe training loop...")
    fit_res = trainer.fit()

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )

    best_ckpt_path = Path(fit_res["checkpoint_path"])
    checkpoint_size_bytes = best_ckpt_path.stat().st_size if best_ckpt_path.exists() else 0

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Epoch                : {fit_res['best_epoch']}")
    print(f"Best Validation Loss      : {fit_res['best_val_loss']}")
    print(f"Total Training Duration   : {fit_res['total_train_time_seconds']:.2f} seconds")
    print(f"Peak GPU VRAM Allocated   : {peak_vram_mb:.2f} MB")
    print(f"Checkpoint File Size      : {checkpoint_size_bytes / (1024**2):.2f} MB")

    # 6. Save Training History CSV
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_csv = reports_dir / "linear_probe_training_history.csv"

    history = fit_res["history"]
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in fieldnames})
    print(f"[Artifacts] Saved history CSV to: {history_csv}")

    # 7. Plot Training Curves Figure
    figures_dir = Path("artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    curves_fig_path = figures_dir / "linear_probe_training_curves.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = history["epoch"]
    
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="navy")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="crimson")
    ax1.axvline(fit_res["best_epoch"], color="gray", linestyle="--", label=f"Best Epoch ({fit_res['best_epoch']})")
    ax1.set_title("Linear Probe Loss Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="navy")
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Accuracy", color="crimson")
    ax2.plot(epochs, history["val_macro_f1"], "^-", label="Val Macro F1", color="forestgreen")
    ax2.axvline(fit_res["best_epoch"], color="gray", linestyle="--", label=f"Best Epoch ({fit_res['best_epoch']})")
    ax2.set_title("Linear Probe Metric Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(curves_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifacts] Saved training curves to: {curves_fig_path}")

    # 8. Reload Best Checkpoint for Evaluation
    print("\n[Evaluation] Reloading best validation checkpoint for dual test set evaluation...")
    trainer.load_best_checkpoint()

    # Evaluation A: Benchmark Test Set
    bench_metrics = evaluate_model(model, benchmark_loader, device)
    bench_metrics["evaluation_protocol"] = "Original Benchmark Test Set (1,311 images)"
    bench_json_path = reports_dir / "linear_probe_benchmark_test_eval.json"
    with open(bench_json_path, "w", encoding="utf-8") as f:
        json.dump(bench_metrics, f, indent=2)

    # Evaluation B: Filtered Test Set
    filt_metrics = evaluate_model(model, filtered_loader, device)
    filt_metrics["evaluation_protocol"] = "Leakage-Controlled Filtered Test Set (1,208 images)"
    filt_json_path = reports_dir / "linear_probe_filtered_test_eval.json"
    with open(filt_json_path, "w", encoding="utf-8") as f:
        json.dump(filt_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("DUAL EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"1. BENCHMARK TEST SET (1,311 images):")
    print(f"   - Accuracy         : {bench_metrics['accuracy'] * 100:.2f}%")
    print(f"   - Macro Precision  : {bench_metrics['macro_precision'] * 100:.2f}%")
    print(f"   - Macro Recall     : {bench_metrics['macro_recall'] * 100:.2f}%")
    print(f"   - Macro F1-Score   : {bench_metrics['macro_f1'] * 100:.2f}%")
    print(f"   - Latency          : {bench_metrics['latency_ms_per_sample']:.2f} ms/sample")
    
    print(f"\n2. LEAKAGE-CONTROLLED FILTERED TEST SET (1,208 images):")
    print(f"   - Accuracy         : {filt_metrics['accuracy'] * 100:.2f}%")
    print(f"   - Macro Precision  : {filt_metrics['macro_precision'] * 100:.2f}%")
    print(f"   - Macro Recall     : {filt_metrics['macro_recall'] * 100:.2f}%")
    print(f"   - Macro F1-Score   : {filt_metrics['macro_f1'] * 100:.2f}%")
    print(f"   - Latency          : {filt_metrics['latency_ms_per_sample']:.2f} ms/sample")
    print("=" * 70)

    # 9. Plot Confusion Matrices Figure
    cm_fig_path = figures_dir / "linear_probe_confusion_matrices.png"
    fig, (ax_b, ax_f) = plt.subplots(1, 2, figsize=(12, 5))

    cm_b = np.array(bench_metrics["confusion_matrix"])
    cm_f = np.array(filt_metrics["confusion_matrix"])

    im1 = ax_b.imshow(cm_b, cmap="Blues")
    ax_b.set_title("Benchmark Test Set (1,311 images)\nAccuracy: {:.2f}% | Macro F1: {:.2f}%".format(
        bench_metrics["accuracy"]*100, bench_metrics["macro_f1"]*100
    ), fontsize=11, fontweight="bold")
    ax_b.set_xticks(range(4))
    ax_b.set_yticks(range(4))
    ax_b.set_xticklabels(CLASS_NAMES, rotation=45)
    ax_b.set_yticklabels(CLASS_NAMES)
    ax_b.set_xlabel("Predicted Class")
    ax_b.set_ylabel("True Class")
    for i in range(4):
        for j in range(4):
            ax_b.text(j, i, str(cm_b[i, j]), ha="center", va="center", color="white" if cm_b[i, j] > cm_b.max()/2 else "black")

    im2 = ax_f.imshow(cm_f, cmap="Greens")
    ax_f.set_title("Filtered Test Set (1,208 images)\nAccuracy: {:.2f}% | Macro F1: {:.2f}%".format(
        filt_metrics["accuracy"]*100, filt_metrics["macro_f1"]*100
    ), fontsize=11, fontweight="bold")
    ax_f.set_xticks(range(4))
    ax_f.set_yticks(range(4))
    ax_f.set_xticklabels(CLASS_NAMES, rotation=45)
    ax_f.set_yticklabels(CLASS_NAMES)
    ax_f.set_xlabel("Predicted Class")
    ax_f.set_ylabel("True Class")
    for i in range(4):
        for j in range(4):
            ax_f.text(j, i, str(cm_f[i, j]), ha="center", va="center", color="white" if cm_f[i, j] > cm_f.max()/2 else "black")

    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifacts] Saved confusion matrices to: {cm_fig_path}")

    # 10. Save Combined Experiment Result JSON
    summary_json_path = reports_dir / "baseline_resnet18_results.json"
    summary_data = {
        "experiment_name": "resnet18_frozen_linear_probe",
        "phase": "2.1B-1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "peak_vram_mb": round(peak_vram_mb, 2)
        },
        "model": {
            "architecture": "resnet18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "parameter_counts": param_counts,
            "checkpoint_path": str(best_ckpt_path),
            "checkpoint_size_mb": round(checkpoint_size_bytes / (1024**2), 2)
        },
        "training": {
            "max_epochs": 25,
            "best_epoch": fit_res["best_epoch"],
            "best_val_loss": fit_res["best_val_loss"],
            "total_train_time_seconds": fit_res["total_train_time_seconds"],
            "early_stopping_triggered": (len(history["epoch"]) < 25)
        },
        "evaluations": {
            "benchmark_test": bench_metrics,
            "filtered_test": filt_metrics
        }
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[Artifacts] Saved master experiment JSON to: {summary_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
