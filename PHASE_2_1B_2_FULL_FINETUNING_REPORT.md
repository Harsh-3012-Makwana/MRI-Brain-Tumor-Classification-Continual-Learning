# Phase 2.1B-2 — ResNet18 Full Fine-Tuning Baseline & Dual Evaluation Report

## Project Context
**Project Title**: Adaptive and Compute-Efficient Class-Incremental Learning for Brain Tumor MRI Classification Using Transfer Learning  
**Phase**: Phase 2.1B-2 — ResNet18 Full Fine-Tuning Baseline  
**Date**: October 5, 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 2.1B-2 established the fully fine-tuned pretrained `ResNet18` baseline performance ceiling. Unlike Phase 2.1B-1 (which trained only the 2,052 parameters of the classification head), Phase 2.1B-2 unfroze all 11,178,564 parameters across the entire ResNet18 architecture for end-to-end backpropagation using ImageNet initialization (`IMAGENET1K_V1`).

Training was executed on an **NVIDIA GeForce RTX 5060 Laptop GPU** (`cuda:0`) with learning rate `0.0001` and weight decay `0.0001`. Early stopping triggered at **Epoch 14** (patience=7), selecting **Epoch 7** as the best validation checkpoint with a **Validation Loss of 0.0436** (a 5.48x reduction compared to linear probing's 0.2388).

Dual test set evaluation yielded near-perfect classification metrics:
- **Raw Benchmark Test Set (1,311 images)**: **99.01% Accuracy**, **98.96% Macro F1-Score**
- **Leakage-Controlled Filtered Test Set (1,208 images)**: **98.92% Accuracy**, **98.93% Macro F1-Score**

---

## 2. Experimental Setup & Hardware Configuration

| Setting / Parameter | Value |
| :--- | :--- |
| **Model Architecture** | ResNet18 (`torchvision.models.resnet18`) |
| **Pretrained Initialization** | `ResNet18_Weights.IMAGENET1K_V1` |
| **Transfer Strategy** | Full Fine-Tuning (`full_finetuning`, all layers unfrozen) |
| **Total Parameters** | 11,178,564 |
| **Trainable Parameters** | 11,178,564 (100.0%) |
| **Frozen Parameters** | 0 (0.0%) |
| **BatchNorm Behavior** | Unfrozen; `train()` updates running statistics; `eval()` static during validation/test |
| **Optimizer** | `AdamW` (`lr=0.0001`, `weight_decay=0.0001`) |
| **Loss Function** | `CrossEntropyLoss()` |
| **Batch Size** | 32 |
| **Max Epochs / Patience** | 25 epochs max / 7 epochs early stopping patience (`val_loss`) |
| **Random Seed** | 42 (deterministic training & split loading) |
| **Hardware Device** | `cuda:0` (NVIDIA GeForce RTX 5060 Laptop GPU) |
| **PyTorch Version** | `2.12.0.dev20260408+cu128` |
| **Peak GPU VRAM Allocated** | 952.27 MB |
| **Total Training Time** | 951.01 seconds (~15.85 minutes) |

---

## 3. Pre-Flight Verification & Safety Audit

Before training, pre-flight checks confirmed complete consistency with Phase 2.1B-1:
1. **Dataset Integrity**: Reused exact `train_validation_manifest.csv` (4,855 train / 857 val) and `evaluation_manifest.csv` (1,311 benchmark / 1,208 filtered test).
2. **Preprocessing Pipeline**: Contour-based crop + 10px padding + 224x224 resize + ImageNet RGB normalization.
3. **Class Mapping**: `{"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}`.
4. **BatchNorm Verification**: Confirmed `model.train()` puts BatchNorm layers in `train()` mode (`training=True`) and `model.eval()` puts them in `eval()` mode (`training=False`).
5. **Linear Probe Artifact Preservation**: Confirmed [best_model.pt](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/checkpoints/resnet18_baseline/best_model.pt) and [baseline_resnet18_results.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/baseline_resnet18_results.json) remained intact and untouched in `artifacts/checkpoints/resnet18_baseline/`.

---

## 4. Comprehensive Compute & Performance Comparison

The table below contrasts the frozen linear-probe baseline (Phase 2.1B-1) against the full fine-tuning baseline (Phase 2.1B-2):

| Metric / Dimension | Phase 2.1B-1 (Linear Probe) | Phase 2.1B-2 (Full Fine-Tuning) | Delta / Shift (Fine-Tuning vs Linear Probe) |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | 2,052 (0.018%) | 11,178,564 (100.0%) | **+11,176,512 (+5,446.6x)** |
| **Best Validation Loss** | 0.2388 | 0.0436 | **-0.1952 (-81.75%)** |
| **Best Epoch** | Epoch 23 | Epoch 7 | **-16 epochs (Faster convergence)** |
| **Training Duration** | 1,495.87 s (24.93 min) | 951.01 s (15.85 min) | **-544.86 s (-36.42% overall time)** |
| **Peak GPU VRAM** | 353.86 MB | 952.27 MB | **+598.41 MB (+268.58%)** |
| **Checkpoint File Size** | 42.74 MB | 128.06 MB | **+85.32 MB (+299.77%)** |
| **Benchmark Accuracy (1,311 img)** | 87.72% | 99.01% | **+11.29%** |
| **Benchmark Macro F1** | 87.18% | 98.96% | **+11.78%** |
| **Filtered Accuracy (1,208 img)** | 87.67% | 98.92% | **+11.25%** |
| **Filtered Macro F1** | 87.58% | 98.93% | **+11.35%** |
| **Inference Latency (Filtered)** | 5.51 ms / sample | 6.37 ms / sample | **+0.86 ms / sample (+15.61%)** |

---

## 5. Performance-Compute Trade-Off Analysis

1. **Accuracy Ceiling**: Unfreezing all backbone layers resolves subtle feature boundary overlaps in brain MRI slices, elevating classification accuracy from **87.67% to 98.92%** (+11.25%) on the leakage-controlled test set.
2. **Convergence Speed**: Full fine-tuning converged significantly faster to its optimal validation state (**Epoch 7** vs Epoch 23), triggering early stopping at Epoch 14 and reducing total training time from 24.9 minutes to 15.9 minutes.
3. **Memory & Storage Overhead**:
   - **VRAM**: Full fine-tuning requires tracking gradient states and optimizer momentum across all 11.18M parameters, increasing peak GPU VRAM from **354 MB to 952 MB** (+268.6%).
   - **Disk Storage**: Checkpoint file size increases from **42.7 MB to 128.1 MB** because optimizer momentum states for all 11.18M parameters must be persisted for exact checkpoint reloading.
4. **Conclusion for Continual Learning**: Full fine-tuning provides an exceptionally high non-continual upper bound (98.92% F1). However, updating all 11.18M parameters during class-incremental steps without regularization or replay buffer management would lead to severe catastrophic forgetting. This trade-off will serve as the core benchmark comparison in Phase 3.

---

## 6. Detailed Dual Evaluation Breakdown

### 6.1 Per-Class Performance Comparison

#### A. Raw Benchmark Test Set (1,311 images)
| Class Name | Support | Precision | Recall | F1-Score | Correct Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 300 | 99.66% | 97.33% | 98.48% | 292 / 300 |
| **Meningioma** | 306 | 97.12% | 99.02% | 98.06% | 303 / 306 |
| **Notumor** | 405 | 99.26% | 100.00% | 99.63% | 405 / 405 |
| **Pituitary** | 300 | 100.00% | 99.33% | 99.67% | 298 / 300 |

#### B. Leakage-Controlled Filtered Test Set (1,208 images)
| Class Name | Support | Precision | Recall | F1-Score | Correct Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 300 | 99.66% | 97.33% | 98.48% | 292 / 300 |
| **Meningioma** | 304 | 97.10% | 99.01% | 98.05% | 301 / 304 |
| **Notumor** | 309 | 99.04% | 100.00% | 99.52% | 309 / 309 |
| **Pituitary** | 295 | 100.00% | 99.32% | 99.66% | 293 / 295 |

---

### 6.2 Confusion Matrices

#### A. Raw Benchmark Test Set Confusion Matrix (1,311 images)
| True \ Predicted | Glioma | Meningioma | Notumor | Pituitary |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | **292** | 7 | 1 | 0 |
| **Meningioma** | 1 | **303** | 2 | 0 |
| **Notumor** | 0 | 0 | **405** | 0 |
| **Pituitary** | 0 | 2 | 0 | **298** |

#### B. Leakage-Controlled Filtered Test Set Confusion Matrix (1,208 images)
| True \ Predicted | Glioma | Meningioma | Notumor | Pituitary |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | **292** | 7 | 1 | 0 |
| **Meningioma** | 1 | **301** | 2 | 0 |
| **Notumor** | 0 | 0 | **309** | 0 |
| **Pituitary** | 0 | 2 | 0 | **293** |

---

## 7. Artifact Inventory & Safety Verification

All Phase 2.1B-2 artifacts were saved into dedicated paths to guarantee zero overwriting of Phase 2.1B-1 baseline files:

1. **Fine-Tuning Checkpoint**: [best_model.pt](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/checkpoints/resnet18_full_finetuning/best_model.pt) (128.06 MB)
2. **Training History CSV**: [full_finetuning_training_history.csv](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/full_finetuning_training_history.csv)
3. **Training Curves Plot**: [full_finetuning_training_curves.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/full_finetuning_training_curves.png)
4. **Benchmark Evaluation Report**: [full_finetuning_benchmark_test_eval.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/full_finetuning_benchmark_test_eval.json)
5. **Filtered Evaluation Report**: [full_finetuning_filtered_test_eval.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/full_finetuning_filtered_test_eval.json)
6. **Confusion Matrices Plot**: [full_finetuning_confusion_matrices.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/full_finetuning_confusion_matrices.png)
7. **Master Summary JSON**: [full_finetuning_resnet18_results.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/full_finetuning_resnet18_results.json)

---

## 8. Test Suite Verification

The full pytest suite was executed post-training:
- **Command**: `.venv\Scripts\python.exe -m pytest`
- **Result**: `25 passed in 18.83s` (100% pass rate)

---

## 9. Conclusion & Next Phase Readiness

Phase 2.1B-2 complete. Both baseline models (frozen linear probe at 87.67% F1 and full fine-tuning at 98.93% F1) are now fully trained, evaluated, and documented under leakage-controlled conditions. Continual learning experiments (Phase 3) can now proceed with established upper and lower baseline bounds.
