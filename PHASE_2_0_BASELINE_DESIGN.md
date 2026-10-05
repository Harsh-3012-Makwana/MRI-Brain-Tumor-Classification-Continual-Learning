# PHASE 2.0 — ResNet18 Baseline Experiment Design Report

**Project Title:** Brain Tumor MRI Classification & Continual Learning  
**Phase:** 2.0 (ResNet18 Baseline Experiment Design)  
**Date:** October 4, 2026  
**Auditor:** Senior Computer Vision Research Engineer  
**Status:** Experiment Design Complete (Zero Training Executed)  

---

## 1. Executive Summary & Objective

Phase 2.0 defines the formal scientific experiment design for establishing a **ResNet18 Transfer Learning Baseline**. 

* **Phase Scope:** Design and architectural verification ONLY. No model weights were downloaded, and zero training epochs were executed.
* **Goal:** Create a clean, reproducible transfer learning baseline using PyTorch and `torchvision.models.resnet18` pretrained on ImageNet-1k, evaluated under the Dual-Evaluation Protocol established in Phase 1.2D.

---

## 2. Model Architecture & Torchvision API (Task 2)

### 2.1 Pretrained Backbone & Torchvision API
* **Torchvision Model API:** `torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)`
* **Pretrained Dataset:** ImageNet-1k ($1.28 \times 10^6$ images, 1,000 visual classes).
* **Backbone Architecture:** 18-layer Residual Network with 4 residual stage blocks (`layer1`, `layer2`, `layer3`, `layer4`) and an Adaptive Average Pooling layer (`avgpool`).

### 2.2 Classification Head & Parameter Breakdown
The default 1,000-class ImageNet fully connected layer (`model.fc`) is replaced with a 4-class linear classification head:

$$\text{Head: } \mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{4 \times 512}, \mathbf{b} \in \mathbb{R}^4$$

* **Class Mapping:** `{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}`
* **Loss Function:** `torch.nn.CrossEntropyLoss()`

#### Parameter Count Verification Table:

| Component / Layer | Parameter Type | Weight Shape | Bias Shape | Parameter Count |
| :--- | :--- | :---: | :---: | :---: |
| **ResNet18 Backbone** | Convolutional & BatchNorm Layers | Various | Various | 11,176,512 |
| **New Classification Head (`model.fc`)** | Linear Layer | $(4, 512)$ | $(4,)$ | 2,052 |
| **Total ResNet18 Model** | **Backbone + Head** | — | — | **11,178,564** |

---

## 3. Training Strategy Design (Task 3)

Two distinct transfer learning strategies will be evaluated independently:

### Strategy A: Frozen-Backbone Linear Probe (`resnet18_frozen_linear_probe`)
* **Mechanism:** All backbone parameters are frozen (`param.requires_grad = False`). Only the 2,052 parameters of the linear classification head (`model.fc`) are updated during gradient descent.
* **Purpose:** Evaluates the quality and transferability of fixed ImageNet feature representations for brain MRI tumor classification.
* **Trainable Parameters:** `2,052` (0.018% of total model parameters).
* **Learning Rate:** `1.0e-3` (`AdamW` optimizer).

### Strategy B: Full Fine-Tuning (`resnet18_full_finetuning`)
* **Mechanism:** All 11,178,564 parameters are unfrozen (`param.requires_grad = True`). Low-level convolutional filters adapt to MRI contrast structures.
* **Purpose:** Evaluates domain adaptation performance when fine-tuning deep feature representations.
* **Trainable Parameters:** `11,178,564` (100% of model parameters).
* **Learning Rate:** `1.0e-4` (10x lower learning rate to prevent catastrophic forgetting of low-level edge detectors).

---

## 4. Reproducibility & Hyperparameter Protocol (Task 4)

### 4.1 Global Determinism & Hardware
* **Random Seed:** `42` (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`).
* **CUDNN Determinism:** `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
* **Hardware Execution:** NVIDIA GeForce RTX 5060 Laptop GPU (`cuda:0`, Compute Capability `sm_120`), 8 GB VRAM.

### 4.2 Hyperparameter Specification Table:

| Hyperparameter | Linear Probe (Strategy A) | Full Fine-Tuning (Strategy B) | Rationale |
| :--- | :---: | :---: | :--- |
| **Batch Size** | 32 | 32 | Standard batch size balancing GPU memory and gradient noise. |
| **Input Resolution** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | Native ImageNet pretrained resolution. |
| **Optimizer** | `AdamW` | `AdamW` | Weight decay decoupling prevents over-regularization. |
| **Base Learning Rate** | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-4}$ | Higher LR for single linear layer; lower LR for full network. |
| **Weight Decay** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | Standard L2 regularization. |
| **LR Scheduler** | Cosine Annealing | Cosine Annealing | Smooth learning rate decay to $1.0 \times 10^{-6}$. |
| **Maximum Epochs** | 25 | 25 | Sufficient convergence window. |
| **Early Stopping** | Patience $= 7$ | Patience $= 7$ | Monitors `val_loss`; stops if no improvement for 7 epochs. |
| **Checkpoint Selection** | Best `val_loss` | Best `val_loss` | Strictly validation-driven checkpoint selection. |

---

## 5. Dual-Evaluation Protocol & Metrics (Task 5)

Neither test set will be exposed to the model during training, validation, early stopping, or hyperparameter selection. Evaluation will be conducted post-training on both test sets:

### 5.1 Test Sets
1. **Raw Benchmark Test Set:** 1,311 images (Original Nickparvar test split).
2. **Leakage-Controlled Filtered Test Set:** 1,208 images (103 cross-split duplicates excluded).

### 5.2 Metrics Suite
* **Overall Accuracy:** $\frac{\text{Correct Predictions}}{\text{Total Samples}}$
* **Macro Precision / Recall / F1-Score:** Unweighted average across 4 classes.
* **Per-Class Metrics:** Precision, Recall, F1 for `glioma`, `meningioma`, `notumor`, `pituitary`.
* **Confusion Matrix:** $4 \times 4$ absolute count matrix.
* **Computational Cost:** Parameter count, total training time (seconds), per-sample inference latency (ms).

---

## 6. Machine-Readable Experiment Tracking Schema (Task 6)

Results will be stored in `artifacts/reports/baseline_resnet18_results.json`:

```json
{
  "experiment_name": "resnet18_baseline",
  "seed": 42,
  "timestamp": "ISO-8601-STRING",
  "strategies": {
    "linear_probe": {
      "best_epoch": 12,
      "best_val_loss": 0.215,
      "best_val_accuracy": 0.924,
      "best_val_macro_f1": 0.921,
      "benchmark_test_metrics": { ... },
      "filtered_test_metrics": { ... }
    },
    "full_finetuning": {
      "best_epoch": 18,
      "best_val_loss": 0.105,
      "best_val_accuracy": 0.965,
      "best_val_macro_f1": 0.963,
      "benchmark_test_metrics": { ... },
      "filtered_test_metrics": { ... }
    }
  }
}
```

---

## 7. Scientific Considerations & Disclaimers (Task 7)

1. **Image-Level vs. Patient-Level Separation:** Exact SHA-256 duplicate filtering prevents byte-identical data leakage. It does not guarantee patient-level slice independence because raw Kaggle JPEGs lack patient IDs.
2. **Dual-Score Reporting Requirement:** Both raw benchmark (1,311 images) and filtered test scores (1,208 images) must be reported side-by-side in all tables to maintain literature comparability while exposing leakage effects.
3. **Contour Cropping Unproven:** Contour-based skull stripping with 10 px padding is our working preprocessing hypothesis; minimal preprocessing vs contour crop ablation will be evaluated in subsequent experiments.
4. **Augmentation Ablation Required:** Data augmentations (horizontal flip, $\pm 15^\circ$ rotation, color jitter) require systematic ablation in future work.
5. **No Clinical Claims:** This study is a computer vision research benchmark; no clinical diagnostic validity or medical deployment suitability is claimed.

---

## 8. Detailed Implementation Plan for Phase 2.1

Upon approval of Phase 2.0, **Phase 2.1 (Baseline ResNet18 Training & Evaluation)** will proceed according to the following step-by-step roadmap:

```
[Phase 2.1 Implementation Roadmap]
│
├── Step 1: Model Builder (src/models/resnet18_baseline.py)
│   ├── Implement ResNet18 baseline model factory using torchvision weights API
│   └── Implement backbone freezing / unfreezing helper methods
│
├── Step 2: Training Engine (src/training/trainer.py)
│   ├── Implement PyTorch training loop with loss tracking & gradient clipping
│   ├── Implement validation monitoring, early stopping (patience=7), and checkpoint saving
│   └── Implement Learning Rate Scheduler stepping
│
├── Step 3: Evaluation Engine (src/training/evaluator.py)
│   ├── Implement multi-metric computation (Accuracy, Macro/Per-class Precision, Recall, F1)
│   ├── Implement confusion matrix computation
│   └── Implement JSON experiment results logger
│
├── Step 4: Execute Experiment Strategy A (Linear Probe)
│   ├── Train 2,052 trainable parameters (25 max epochs)
│   └── Evaluate on Validation, Raw Benchmark Test (1,311), and Filtered Test (1,208)
│
├── Step 5: Execute Experiment Strategy B (Full Fine-Tuning)
│   ├── Train 11,178,564 trainable parameters (25 max epochs)
│   └── Evaluate on Validation, Raw Benchmark Test (1,311), and Filtered Test (1,208)
│
└── Step 6: Artifact & Report Generation
    ├── Save model checkpoints to artifacts/checkpoints/resnet18_baseline/
    ├── Save evaluation metrics to artifacts/reports/baseline_resnet18_results.json
    └── Write PHASE_2_1_BASELINE_RESULTS.md summary report
```

---

**Status:** Phase 2.0 Experiment Design complete. Standing by for user authorization to begin Phase 2.1 model training.
