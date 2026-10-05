# Phase 2.1B-1 — ResNet18 Linear Probe Baseline Training & Dual Evaluation Report

## Project Context
**Project Title**: Adaptive and Compute-Efficient Class-Incremental Learning for Brain Tumor MRI Classification Using Transfer Learning  
**Phase**: Phase 2.1B-1 — ResNet18 Frozen Linear Probe Baseline Training  
**Date**: October 4, 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 2.1B-1 established the official frozen-backbone linear probe baseline using a torchvision `ResNet18` model pretrained on ImageNet (`IMAGENET1K_V1`). Only the 4-class classification head (`fc` layer with 2,052 trainable parameters) was updated, while all 11,176,512 backbone parameters remained strictly frozen. 

Training completed across 25 epochs on an **NVIDIA GeForce RTX 5060 Laptop GPU** (`cuda:0`). The best validation state was reached at **Epoch 23** with a **Validation Loss of 0.2388**. 

Independent evaluation on both test protocols yielded robust classification performance:
- **Raw Benchmark Test Set (1,311 images)**: **87.72% Accuracy**, **87.18% Macro F1-Score**
- **Leakage-Controlled Filtered Test Set (1,208 images)**: **87.67% Accuracy**, **87.58% Macro F1-Score**

The dual evaluation confirms that removing the 103 cross-split duplicate test images does not degrade classification performance (+0.40% Macro F1 improvement on clean test data), proving that the linear probe generalises effectively without reliance on data leakage.

---

## 2. Experimental Setup & Hardware Configuration

| Component / Setting | Value / Detail |
| :--- | :--- |
| **Model Architecture** | ResNet18 (`torchvision.models.resnet18`) |
| **Pretrained Weights** | `ResNet18_Weights.IMAGENET1K_V1` |
| **Transfer Strategy** | Frozen Backbone Linear Probe (`fc` head replaced for 4 classes) |
| **Total Parameters** | 11,178,564 |
| **Trainable Parameters** | 2,052 (0.018% of total) |
| **Frozen Parameters** | 11,176,512 (99.982% of total) |
| **Optimizer** | `AdamW` (`lr=0.001`, `weight_decay=0.0001`) |
| **Loss Function** | `CrossEntropyLoss()` |
| **Batch Size** | 32 |
| **Max Epochs / Patience** | 25 epochs / 7 epochs patience (`val_loss` selection) |
| **Random Seed** | 42 (deterministic training & dataset split) |
| **Hardware Device** | `cuda:0` (NVIDIA GeForce RTX 5060 Laptop GPU, 7.96 GB VRAM) |
| **PyTorch Version** | `2.12.0.dev20260408+cu128` |
| **Peak VRAM Allocated** | 353.86 MB |
| **Total Training Time** | 1,495.87 seconds (~24.93 minutes) |

---

## 3. Dataset Split Verification

Training and validation splits were generated deterministically from the 5,712 training images using stratified 85/15 sampling. Test sets were strictly isolated.

| Split Name | Image Count | SHA-256 Overlap with Training | Purpose |
| :--- | :---: | :---: | :--- |
| **Train Split** | 4,855 | N/A (Source) | Gradient updates on `fc` head |
| **Validation Split** | 857 | 0 | Model selection & early stopping |
| **Benchmark Test Set** | 1,311 | 103 exact duplicates | Legacy benchmark comparison |
| **Filtered Test Set** | 1,208 | 0 exact duplicates | Leakage-controlled scientific benchmark |

---

## 4. Training Progression & Best Checkpoint

The linear probe model was trained for 25 full epochs. Gradient updates were strictly confined to the `fc` layer, while `BatchNorm` layers were kept in strict evaluation mode (`eval()`) with static running statistics (`running_mean` and `running_var`).

- **Best Epoch**: 23
- **Best Validation Loss**: `0.2388`
- **Best Validation Accuracy**: `90.67%` (Epoch 23)
- **Checkpoint Location**: [best_model.pt](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/checkpoints/resnet18_baseline/best_model.pt) (42.74 MB)

---

## 5. Dual Evaluation Results Summary

After training completed, the best model checkpoint (`best_model.pt` from Epoch 23) was reloaded and evaluated independently on both test sets using evaluation mode (`eval()`) with disabled gradients (`torch.no_grad()`).

### 5.1 Overall Benchmark Performance

| Metric | Raw Benchmark Test Set (1,311 images) | Leakage-Controlled Filtered Test Set (1,208 images) | Difference (Filtered - Benchmark) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **87.72%** | **87.67%** | -0.05% |
| **Macro Precision** | **87.45%** | **87.71%** | +0.26% |
| **Macro Recall** | **87.02%** | **87.63%** | +0.61% |
| **Macro F1-Score** | **87.18%** | **87.58%** | **+0.40%** |
| **Inference Latency** | 5.43 ms / sample | 5.51 ms / sample | +0.08 ms |
| **Total Evaluation Time** | 7.12 seconds | 6.66 seconds | -0.46 s |

---

### 5.2 Detailed Per-Class Breakdown

#### A. Raw Benchmark Test Set (1,311 images)
| Class Name | Support | Precision | Recall | F1-Score | Correct Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 300 | 89.13% | 82.00% | 85.42% | 246 / 300 |
| **Meningioma** | 306 | 75.87% | 78.10% | 76.97% | 239 / 306 |
| **Notumor** | 405 | 92.20% | 96.30% | 94.20% | 390 / 405 |
| **Pituitary** | 300 | 92.59% | 91.67% | 92.13% | 275 / 300 |

#### B. Leakage-Controlled Filtered Test Set (1,208 images)
| Class Name | Support | Precision | Recall | F1-Score | Correct Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 300 | 90.11% | 82.00% | 85.86% | 246 / 300 |
| **Meningioma** | 304 | 77.78% | 78.29% | 78.03% | 238 / 304 |
| **Notumor** | 309 | 90.50% | 98.71% | 94.43% | 305 / 309 |
| **Pituitary** | 295 | 92.47% | 91.53% | 91.99% | 270 / 295 |

---

### 5.3 Confusion Matrices

#### A. Raw Benchmark Test Set Confusion Matrix
| True \ Predicted | Glioma | Meningioma | Notumor | Pituitary |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | **246** | 46 | 3 | 5 |
| **Meningioma** | 20 | **239** | 30 | 17 |
| **Notumor** | 5 | 10 | **390** | 0 |
| **Pituitary** | 5 | 20 | 0 | **275** |

#### B. Filtered Test Set Confusion Matrix
| True \ Predicted | Glioma | Meningioma | Notumor | Pituitary |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | **246** | 46 | 3 | 5 |
| **Meningioma** | 20 | **238** | 29 | 17 |
| **Notumor** | 2 | 2 | **305** | 0 |
| **Pituitary** | 5 | 20 | 0 | **270** |

---

## 6. Scientific Insights & Findings

1. **Robust Feature Extraction**: Frozen ImageNet features achieve ~87.7% accuracy across 4 tumor classes without updating any backbone layers, proving that early transfer features possess strong semantic representational power for brain MRI slices.
2. **Minimal Leakage Impact**: Filtering the 103 duplicate test images resulted in virtually unchanged performance (**87.67% vs 87.72% accuracy**; **87.58% vs 87.18% F1-score**). This demonstrates that the model is not memorising duplicate test instances.
3. **Class-Specific Distinctions**:
   - **Notumor** (F1 94.4%) and **Pituitary** (F1 92.0%) are easily separated by frozen linear features.
   - **Meningioma** (F1 78.0%) shows the highest confusion, primarily misclassified into Glioma (46 cases) and Pituitary (20 cases), indicating subtle boundary overlaps that will require backbone fine-tuning.

---

## 7. Artifact Manifest & Verification

All experiment artifacts have been generated, saved, and verified:

1. **Model Checkpoint**: [best_model.pt](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/checkpoints/resnet18_baseline/best_model.pt) (42.74 MB)
2. **Training History**: [linear_probe_training_history.csv](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/linear_probe_training_history.csv)
3. **Training Curves Plot**: [linear_probe_training_curves.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/linear_probe_training_curves.png)
4. **Benchmark Evaluation Report**: [linear_probe_benchmark_test_eval.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/linear_probe_benchmark_test_eval.json)
5. **Filtered Evaluation Report**: [linear_probe_filtered_test_eval.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/linear_probe_filtered_test_eval.json)
6. **Confusion Matrices Plot**: [linear_probe_confusion_matrices.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/linear_probe_confusion_matrices.png)
7. **Master Summary JSON**: [baseline_resnet18_results.json](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/baseline_resnet18_results.json)

---

## 8. Test Suite Verification

The full project unit and integration test suite was executed:
- **Command**: `.venv\Scripts\python.exe -m pytest`
- **Result**: `25 passed in 14.83s` (100% pass rate)

---

## 9. Next Steps

Phase 2.1B-1 linear probe baseline training is complete. The next phase is **Phase 2.1B-2 — ResNet18 Full Fine-Tuning Baseline**, where the entire ResNet18 backbone (all 11.17M parameters) will be un-frozen and fine-tuned end-to-end to establish the ceiling non-continual transfer learning benchmark.
