"""Execution script for Phase 2.1B-2: ResNet18 Full Fine-Tuning Baseline & Dual Evaluation."""

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
    print("=" * 70, flush=True)
    print("PHASE 2.1B-2: RESNET18 FULL FINE-TUNING BASELINE TRAINING", flush=True)
    print("=" * 70, flush=True)

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

    print(f"Target Device: {device}", flush=True)

    # 2. Get DataLoaders (same deterministic data pipeline)
    print("\n[Pipeline] Constructing PyTorch DataLoaders...", flush=True)
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

    print(f"  - Train Samples          : {len(train_loader.dataset):,}", flush=True)
    print(f"  - Val Samples            : {len(val_loader.dataset):,}", flush=True)
    print(f"  - Benchmark Test Samples : {len(benchmark_loader.dataset):,}", flush=True)
    print(f"  - Filtered Test Samples  : {len(filtered_loader.dataset):,}", flush=True)

    # 3. Instantiate Model (Full Fine-Tuning: Unfrozen Backbone)
    print("\n[Model] Building ResNet18 (IMAGENET1K_V1, Full Fine-Tuning)...", flush=True)
    model = build_resnet18_baseline(strategy="full_finetuning", pretrained=True)
    param_counts = model.get_parameter_counts()
    print(f"  - Total Parameters      : {param_counts['total_parameters']:,}", flush=True)
    print(f"  - Trainable Parameters  : {param_counts['trainable_parameters']:,}", flush=True)
    print(f"  - Frozen Parameters     : {param_counts['frozen_parameters']:,}", flush=True)

    # Verify BatchNorm behavior during full fine-tuning
    model.train()
    bn_train_status = model.backbone.layer1[0].bn1.training
    model.eval()
    bn_eval_status = model.backbone.layer1[0].bn1.training
    print(f"  - BatchNorm train() mode status : {bn_train_status} (Expected: True)", flush=True)
    print(f"  - BatchNorm eval() mode status  : {bn_eval_status} (Expected: False)", flush=True)

    # 4. Setup Optimizer & Trainer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001
    )
    
    checkpoint_dir = Path("artifacts/checkpoints/resnet18_full_finetuning")
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

    # 5. Run Custom Training Loop with Explicit Unbuffered Logging
    print("\n[Training] Starting Full Fine-Tuning training loop...", flush=True)
    print("-" * 70, flush=True)
    header = f"{'Epoch':<7} | {'Train Loss':<10} | {'Train Acc':<9} | {'Val Loss':<9} | {'Val Acc':<8} | {'Val F1':<8} | {'LR':<8} | {'Time (s)':<8}"
    print(header, flush=True)
    print("-" * 70, flush=True)

    start_train_time = time.perf_counter()
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
        "epoch_time_seconds": []
    }

    for epoch in range(1, 26):
        epoch_start = time.perf_counter()

        train_loss, train_acc = trainer.train_epoch()

        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, device)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        # Compute validation loss
        model.eval()
        val_running_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = trainer.criterion(logits, targets)
                val_running_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
        val_loss = val_running_loss / val_samples if val_samples > 0 else 0.0

        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["lr"].append(current_lr)
        history["epoch_time_seconds"].append(round(epoch_time, 2))

        # Checkpoint saving & early stopping
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            is_best = True

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": val_acc,
                "best_val_f1": val_f1
            }, trainer.best_checkpoint_path)
        else:
            patience_counter += 1

        best_tag = " [BEST]" if is_best else ""
        log_line = f"{epoch:<7} | {train_loss:<10.4f} | {train_acc:<9.4f} | {val_loss:<9.4f} | {val_acc:<8.4f} | {val_f1:<8.4f} | {current_lr:<8.6f} | {epoch_time:<8.2f}{best_tag}"
        print(log_line, flush=True)

        if patience_counter >= 7:
            print(f"\n[Trainer] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).", flush=True)
            break

    total_train_duration = time.perf_counter() - start_train_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print("=" * 70, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Best Epoch                : {best_epoch}", flush=True)
    print(f"Best Validation Loss      : {best_val_loss:.4f}", flush=True)
    print(f"Total Training Duration   : {total_train_duration:.2f} seconds", flush=True)
    print(f"Peak GPU VRAM Allocated   : {peak_vram_mb:.2f} MB", flush=True)

    # 6. Save Training History CSV & Plots
    reports_dir = Path("artifacts/reports")
    figures_dir = Path("artifacts/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "full_finetuning_training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_time_seconds"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["val_macro_f1"][i],
                history["lr"][i],
                history["epoch_time_seconds"][i]
            ])
    print(f"[Artifacts] Saved history CSV to: {csv_path}", flush=True)

    # Plot Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = history["epoch"]

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_title("ResNet18 Full Fine-Tuning: Loss Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs_range, history["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2)
    ax2.plot(epochs_range, history["val_macro_f1"], label="Val Macro F1", color="#9467bd", linestyle=":", linewidth=2)
    ax2.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax2.set_title("ResNet18 Full Fine-Tuning: Accuracy & F1 Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curves_path = figures_dir / "full_finetuning_training_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[Artifacts] Saved training curves to: {curves_path}", flush=True)

    # 7. Dual Test Set Evaluation
    print("\n[Evaluation] Reloading best validation checkpoint for dual test set evaluation...", flush=True)
    best_checkpoint = torch.load(trainer.best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\n  - Running Evaluation 1: Raw Benchmark Test Set (1,311 images)...", flush=True)
    benchmark_results = evaluate_model(model, benchmark_loader, device)
    
    print("  - Running Evaluation 2: Leakage-Controlled Filtered Test Set (1,208 images)...", flush=True)
    filtered_results = evaluate_model(model, filtered_loader, device)

    # Print Dual Summary
    print("\n" + "=" * 70, flush=True)
    print("DUAL EVALUATION RESULTS SUMMARY (FULL FINE-TUNING)", flush=True)
    print("=" * 70, flush=True)
    print("1. BENCHMARK TEST SET (1,311 images):", flush=True)
    print(f"   - Accuracy         : {benchmark_results['accuracy'] * 100:.2f}%", flush=True)
    print(f"   - Macro Precision  : {benchmark_results['macro_precision'] * 100:.2f}%", flush=True)
    print(f"   - Macro Recall     : {benchmark_results['macro_recall'] * 100:.2f}%", flush=True)
    print(f"   - Macro F1-Score   : {benchmark_results['macro_f1'] * 100:.2f}%", flush=True)
    print(f"   - Latency          : {benchmark_results['latency_ms_per_sample']:.2f} ms/sample", flush=True)

    print("\n2. LEAKAGE-CONTROLLED FILTERED TEST SET (1,208 images):", flush=True)
    print(f"   - Accuracy         : {filtered_results['accuracy'] * 100:.2f}%", flush=True)
    print(f"   - Macro Precision  : {filtered_results['macro_precision'] * 100:.2f}%", flush=True)
    print(f"   - Macro Recall     : {filtered_results['macro_recall'] * 100:.2f}%", flush=True)
    print(f"   - Macro F1-Score   : {filtered_results['macro_f1'] * 100:.2f}%", flush=True)
    print(f"   - Latency          : {filtered_results['latency_ms_per_sample']:.2f} ms/sample", flush=True)
    print("=" * 70, flush=True)

    # Save Individual Evaluation JSONs
    benchmark_json_path = reports_dir / "full_finetuning_benchmark_test_eval.json"
    benchmark_results["evaluation_protocol"] = "Original Benchmark Test Set (1,311 images)"
    with open(benchmark_json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    filtered_json_path = reports_dir / "full_finetuning_filtered_test_eval.json"
    filtered_results["evaluation_protocol"] = "Leakage-Controlled Filtered Test Set (1,208 images)"
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_results, f, indent=2)

    # 8. Plot Dual Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm_bench = np.array(benchmark_results["confusion_matrix"])
    im1 = ax1.imshow(cm_bench, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Benchmark Test Set (1,311 images)\nFull Fine-Tuning ResNet18", fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=ax1, shrink=0.8)
    tick_marks = np.arange(len(CLASS_NAMES))
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    for i in range(cm_bench.shape[0]):
        for j in range(cm_bench.shape[1]):
            ax1.text(j, i, str(cm_bench[i, j]), ha="center", va="center",
                     color="white" if cm_bench[i, j] > cm_bench.max() / 2 else "black")

    cm_filt = np.array(filtered_results["confusion_matrix"])
    im2 = ax2.imshow(cm_filt, interpolation="nearest", cmap=plt.cm.Greens)
    ax2.set_title("Filtered Test Set (1,208 images)\nFull Fine-Tuning ResNet18", fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    for i in range(cm_filt.shape[0]):
        for j in range(cm_filt.shape[1]):
            ax2.text(j, i, str(cm_filt[i, j]), ha="center", va="center",
                     color="white" if cm_filt[i, j] > cm_filt.max() / 2 else "black")

    plt.tight_layout()
    cm_path = figures_dir / "full_finetuning_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Artifacts] Saved confusion matrices to: {cm_path}", flush=True)

    # 9. Save Master Summary JSON
    checkpoint_size_mb = os.path.getsize(trainer.best_checkpoint_path) / (1024 ** 2)
    master_results = {
        "experiment_name": "resnet18_full_finetuning",
        "phase": "2.1B-2",
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
            "strategy": "full_finetuning",
            "parameter_counts": param_counts,
            "checkpoint_path": str(trainer.best_checkpoint_path),
            "checkpoint_size_mb": round(checkpoint_size_mb, 2)
        },
        "training": {
            "max_epochs": 25,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 32,
            "total_train_time_seconds": round(total_train_duration, 2),
            "early_stopping_triggered": (patience_counter >= 7)
        },
        "evaluations": {
            "benchmark_test": benchmark_results,
            "filtered_test": filtered_results
        }
    }

    master_json_path = reports_dir / "full_finetuning_resnet18_results.json"
    with open(master_json_path, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"[Artifacts] Saved master experiment JSON to: {master_json_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
