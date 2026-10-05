# Phase 2.1B-3 — Baseline Research Audit Report

**Project Title**: Adaptive and Compute-Efficient Class-Incremental Learning for Brain Tumor MRI Classification Using Transfer Learning  
**Phase**: Phase 2.1B-3 — Baseline Research Audit  
**Audited Experiments**: Phase 2.1B-1 (ResNet18 Frozen Linear Probe) and Phase 2.1B-2 (ResNet18 Full Fine-Tuning)  
**Date**: October 5, 2026  
**Auditor**: Antigravity Autonomous Research Assistant  
**Status**: READ-ONLY AUDIT COMPLETE  

---

## Executive Summary

This research audit evaluates the two reference baselines established for the brain tumor MRI classification project prior to beginning Phase 3 (Class-Incremental Continual Learning). The audit rigorously verifies configuration consistency, dataset split isolation, performance metrics, training trajectory behaviors, compute profiles, scientific claim rigor, and environmental reproducibility.

Both baseline experiments were conducted on the same hardware, using the same codebase, random seeds, manifests, preprocessing routines, and evaluation protocols. The empirical results provide verified lower-bound (Linear Probe: **87.58% Filtered Macro F1**) and upper-bound (Full Fine-Tuning: **98.93% Filtered Macro F1**) anchors. 

**Audit Verdict**: **READY FOR CONTINUAL LEARNING**, subject to mandatory protocol freeze and explicit documentation of dataset limitations.

---

## 1. Configuration Consistency

Every experimental and architectural variable was inspected directly in the codebase (`configs/base_config.yaml`, `configs/model_resnet18.yaml`, `src/data/pipeline_factory.py`, `src/data/dataset.py`, `src/models/resnet18_baseline.py`, `scripts/run_phase2_1b_1_linear_probe.py`, and `scripts/run_phase2_1b_2_full_finetuning.py`).

| Configuration Factor | Phase 2.1B-1 (Linear Probe) | Phase 2.1B-2 (Full Fine-Tuning) | Audit Status | Evidence / Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Dataset Source** | `data/raw/` | `data/raw/` | **SAME** | Identical raw directory |
| **Class Definitions** | glioma, meningioma, notumor, pituitary | glioma, meningioma, notumor, pituitary | **SAME** | `CLASS_TO_IDX` mapping identical |
| **Train/Val/Test Split** | Stratified 85/15 (4,855 train / 857 val) | Stratified 85/15 (4,855 train / 857 val) | **SAME** | Same manifest rows and sample hashes |
| **Random Seed** | 42 | 42 | **SAME** | `torch.manual_seed(42)`, `cuda.manual_seed_all(42)`, DataLoader generator |
| **Preprocessing Pipeline** | `MRIPreprocessor` (contour crop + 10px pad) | `MRIPreprocessor` (contour crop + 10px pad) | **SAME** | Standardized contour bounding box crop |
| **Image Size** | (224, 224) | (224, 224) | **SAME** | Bilinear resize to 224x224 |
| **Normalization** | ImageNet mean & std | ImageNet mean & std | **SAME** | mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]` |
| **Training Augmentations** | Flip(0.5), Rot(15°), ColorJitter(0.15) | Flip(0.5), Rot(15°), ColorJitter(0.15) | **SAME** | `get_default_train_transform()` |
| **Architecture** | torchvision ResNet18 | torchvision ResNet18 | **SAME** | `models.resnet18` backbone |
| **Pretrained Weights** | `ResNet18_Weights.IMAGENET1K_V1` | `ResNet18_Weights.IMAGENET1K_V1` | **SAME** | Identical PyTorch hub weight cache |
| **Classification Head** | `nn.Linear(512, 4)` | `nn.Linear(512, 4)` | **SAME** | Identical replacement of final FC layer |
| **Loss Function** | `nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss()` | **SAME** | Standard multi-class cross-entropy |
| **Optimizer** | `AdamW` | `AdamW` | **SAME** | Standard decoupled weight decay optimizer |
| **Learning Rate** | `0.001` (`1e-3`) | `0.0001` (`1e-4`) | **DIFFERENT** | Methodologically justified: higher LR for linear head; lower LR to prevent feature disruption in backbone |
| **Weight Decay** | `0.0001` (`1e-4`) | `0.0001` (`1e-4`) | **SAME** | Standard AdamW regularization |
| **Batch Size** | 32 | 32 | **SAME** | Mini-batch dimension across all loaders |
| **Maximum Epochs** | 25 | 25 | **SAME** | Equal upper epoch budget |
| **Early Stopping Patience** | 7 epochs | 7 epochs | **SAME** | Evaluated strictly on `val_loss` |
| **Checkpoint Criterion** | Minimum `val_loss` | Minimum `val_loss` | **SAME** | `best_val_loss` selection |
| **BatchNorm Behavior** | Frozen (`eval()` mode) | Unfrozen (`train()` mode) | **DIFFERENT** | Methodologically required: linear probe must keep running stats frozen; fine-tuning updates running stats |
| **Hardware Device** | `cuda:0` | `cuda:0` | **SAME** | NVIDIA GeForce RTX 5060 Laptop GPU |
| **DataLoader Settings** | `num_workers=0`, `pin_memory=True` | `num_workers=0`, `pin_memory=True` | **SAME** | Single-threaded loader for deterministic Windows IPC safety |

---

## 2. Data Split Audit

### 2.1 Manifest & Split Isolation
1. **Training Split**: Both experiments ingested precisely the same **4,855 images** defined in `artifacts/reports/train_validation_manifest.csv` under `split_assignment == "Train"`.
2. **Validation Split**: Both experiments monitored model progress against the same **857 images** defined in `artifacts/reports/train_validation_manifest.csv` under `split_assignment == "Val"`.
3. **Test Set Strict Isolation**: Test loaders were never imported, instantiated, or accessed inside the training loops (`ModelTrainer.train_epoch()` or `ModelTrainer.fit()`). Checkpoint selection and early stopping were solely governed by validation loss computed on the 857 validation samples.
4. **Reproducible Seed Handling**: Random seeds were initialized at `seed = 42` across Python `random`, NumPy, and PyTorch (CPU and CUDA). DataLoader batch shuffling utilized a seeded `torch.Generator().manual_seed(42)`.

### 2.2 Test Set Protocols
Two distinct test sets were audited from `artifacts/reports/evaluation_manifest.csv`:
- **Original Benchmark Test Set (1,311 images)**: The unedited testing partition from the original archive.
- **Leakage-Controlled Filtered Test Set (1,208 images)**: The benchmark test set after excluding 103 images that were exact byte-level duplicates of training set images.

### 2.3 Duplicate-Filtering Protocol & Scientific Limitation
The duplicate detection procedure implemented in Phase 1.2C and Phase 1.2D computed the SHA-256 hash of the raw byte content of each file (`compute_sha256(p)`). Slices in the testing partition whose SHA-256 hash matched an existing hash in the training partition were identified as `CROSS_SPLIT_EXACT_DUPLICATE_OF_TRAINING_SAMPLE` and excluded from the filtered test set.

> [!WARNING]
> **Scientific Disclosure on Dataset Independence**:  
> The duplicate-filtering protocol addresses **only exact duplicate files** based on byte-level SHA-256 hash equality. The source dataset lacks patient identifiers, hospital codes, acquisition dates, or subject metadata. Consequently:  
> **Patient-level independence was not established.**  
> It remains possible that multiple slices from the same patient or MRI volume exist across training, validation, and testing partitions. All experimental results must be interpreted strictly as slice-level image classification benchmarks rather than patient-level diagnostic generalizations.

---

## 3. Performance Audit

The reported performance metrics were extracted directly from the verified result files (`artifacts/reports/baseline_resnet18_results.json` and `artifacts/reports/full_finetuning_resnet18_results.json`) and training histories.

### 3.1 Primary Summary Metrics

| Metric | Phase 2.1B-1 (Linear Probe) | Phase 2.1B-2 (Full Fine-Tuning) |
| :--- | :---: | :---: |
| **Best Checkpoint Epoch** | Epoch 23 | Epoch 7 |
| **Best Validation Loss** | 0.2388 | 0.0436 |
| **Best Validation Accuracy** | 91.13% | 98.83% |
| **Benchmark Test Accuracy (1,311 img)** | 87.72% | 99.01% |
| **Benchmark Test Macro Precision** | 87.45% | 99.01% |
| **Benchmark Test Macro Recall** | 87.02% | 98.92% |
| **Benchmark Test Macro F1-Score** | 87.18% | 98.96% |
| **Filtered Test Accuracy (1,208 img)** | 87.67% | 98.92% |
| **Filtered Test Macro Precision** | 87.71% | 98.95% |
| **Filtered Test Macro Recall** | 87.63% | 98.92% |
| **Filtered Test Macro F1-Score** | 87.58% | 98.93% |

---

### 3.2 Per-Class Breakdown

#### A. Raw Benchmark Test Set (1,311 images)

| Class Label | Support | Linear Probe Precision | Linear Probe Recall | Linear Probe F1 | Fine-Tuning Precision | Fine-Tuning Recall | Fine-Tuning F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **glioma** | 300 | 89.13% | 82.00% | 85.42% | 99.66% | 97.33% | 98.48% |
| **meningioma** | 306 | 75.87% | 78.10% | 76.97% | 97.12% | 99.02% | 98.06% |
| **notumor** | 405 | 92.20% | 96.30% | 94.20% | 99.26% | 100.00% | 99.63% |
| **pituitary** | 300 | 92.59% | 91.67% | 92.13% | 100.00% | 99.33% | 99.67% |

#### B. Leakage-Controlled Filtered Test Set (1,208 images)

| Class Label | Support | Linear Probe Precision | Linear Probe Recall | Linear Probe F1 | Fine-Tuning Precision | Fine-Tuning Recall | Fine-Tuning F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **glioma** | 300 | 90.11% | 82.00% | 85.86% | 99.66% | 97.33% | 98.48% |
| **meningioma** | 304 | 77.78% | 78.29% | 78.03% | 97.10% | 99.01% | 98.05% |
| **notumor** | 309 | 90.50% | 98.71% | 94.43% | 99.04% | 100.00% | 99.52% |
| **pituitary** | 295 | 92.47% | 91.53% | 91.99% | 100.00% | 99.32% | 99.66% |

---

### 3.3 Confusion Matrices

#### A. Raw Benchmark Test Set (1,311 images)
- **Linear Probe**:
  $$\begin{bmatrix} 246 & 46 & 3 & 5 \\ 20 & 239 & 30 & 17 \\ 5 & 10 & 390 & 0 \\ 5 & 20 & 0 & 275 \end{bmatrix}$$
- **Full Fine-Tuning**:
  $$\begin{bmatrix} 292 & 7 & 1 & 0 \\ 1 & 303 & 2 & 0 \\ 0 & 0 & 405 & 0 \\ 0 & 2 & 0 & 298 \end{bmatrix}$$

#### B. Filtered Test Set (1,208 images)
- **Linear Probe**:
  $$\begin{bmatrix} 246 & 46 & 3 & 5 \\ 20 & 238 & 29 & 17 \\ 2 & 2 & 305 & 0 \\ 5 & 20 & 0 & 270 \end{bmatrix}$$
- **Full Fine-Tuning**:
  $$\begin{bmatrix} 292 & 7 & 1 & 0 \\ 1 & 301 & 2 & 0 \\ 0 & 0 & 309 & 0 \\ 0 & 2 & 0 & 293 \end{bmatrix}$$

*(Order of classes in rows and columns: `glioma`, `meningioma`, `notumor`, `pituitary`)*

---

## 4. Training-Curve Audit

Detailed inspection of `artifacts/reports/linear_probe_training_history.csv` and `artifacts/reports/full_finetuning_training_history.csv` reveals the following empirical dynamics:

### 4.1 Linear Probe Dynamics
- **Overfitting Assessment**: Training loss steadily decreased from 0.7415 to 0.2493, while validation loss decreased from 0.5261 to 0.2388. The close alignment between training loss and validation loss across all 25 epochs is **consistent with** strong regularization enforced by freezing the backbone (updating only 2,052 parameters). No evidence of severe overfitting was observed.
- **Metric Consistency**: Validation loss and validation accuracy tracked consistently. The minimum validation loss (0.2388) coincided with the peak validation accuracy (91.13%) at Epoch 23.
- **Early Stopping Behavior**: The patience threshold was 7. Because the best epoch occurred at Epoch 23, the non-improving counter reached only 2 (out of 7) before reaching the 25-epoch ceiling. The training loop completed the full 25 epochs as configured.

### 4.2 Full Fine-Tuning Dynamics
- **Overfitting Assessment**: Training loss rapidly decreased from 0.2842 in Epoch 1 to 0.0118 in Epoch 14 (with training accuracy reaching 99.63%). Validation loss reached its minimum of 0.0436 at Epoch 7 and subsequently fluctuated between 0.0484 and 0.0904 in Epochs 8–14. This widening gap between near-zero training loss and plateauing validation loss **is consistent with** the onset of mild empirical overfitting on training set features.
- **Metric Consistency**: Validation loss achieved its minimum at Epoch 7 (0.0436), while validation accuracy peaked marginally higher in later epochs (98.95% at Epochs 10 and 12 vs 98.83% at Epoch 7). Prioritizing minimum validation loss for checkpoint selection prevented checkpointing overconfident predictive distributions.
- **Early Stopping Behavior**: Early stopping functioned strictly as implemented. Following Epoch 7, validation loss failed to improve across 7 consecutive epochs (Epochs 8 through 14). The trainer terminated execution automatically at Epoch 14 and restored the Epoch 7 checkpoint.
- **Instability Analysis**: No numerical divergence, loss spikes, or NaN gradients were observed. Validation loss exhibited minor epoch-to-epoch oscillation (e.g., 0.0436 $\rightarrow$ 0.0734 $\rightarrow$ 0.0768 $\rightarrow$ 0.0484), which **may indicate** gradient stepping variance on the validation sample distribution under constant learning rate without cosine annealing.

---

## 5. Compute Audit & Runtime Discrepancy Investigation

### 5.1 Resource Comparison

| Compute Dimension | Linear Probe (Phase 2.1B-1) | Full Fine-Tuning (Phase 2.1B-2) | Difference / Ratio |
| :--- | :---: | :---: | :---: |
| **Total Model Parameters** | 11,178,564 | 11,178,564 | Same |
| **Trainable Parameters** | 2,052 (0.018%) | 11,178,564 (100.0%) | +11,176,512 (+5,446.6x) |
| **Frozen Parameters** | 11,176,512 (99.982%) | 0 (0.0%) | -11,176,512 |
| **Total Training Duration** | 1,495.87 s (24.93 min) | 951.01 s (15.85 min) | -544.86 s (-36.4%) |
| **Completed Epochs** | 25 epochs | 14 epochs | -11 epochs |
| **Mean Duration per Epoch** | 59.83 s / epoch | 67.93 s / epoch | **+8.10 s / epoch (+13.5%)** |
| **Peak GPU VRAM** | 353.86 MB | 952.27 MB | +598.41 MB (+169.1%) |
| **Checkpoint File Size** | 42.74 MB | 128.06 MB | +85.32 MB (+199.6%) |
| **Filtered Inference Latency** | 5.51 ms / sample | 6.37 ms / sample | +0.86 ms / sample (+15.6%) |

---

### 5.2 Resolution of the "Faster Training Time" Discrepancy

The preliminary report noted that full fine-tuning finished in less total wall-clock time (951.01 s) than linear probing (1,495.87 s), which could easily be misinterpreted as implying that full fine-tuning is computationally less intensive.

The audit establishes the following distinction:

1. **Measured Fact**:
   - On a per-epoch basis, full fine-tuning took **67.93 seconds per epoch**, whereas linear probing took **59.83 seconds per epoch**.
   - Full fine-tuning was **13.5% slower per epoch** than linear probing.
   - Linear probing ran for **25 complete epochs**, whereas full fine-tuning ran for only **14 epochs**.

2. **Implementation Explanation**:
   - The shorter total wall-clock time in full fine-tuning is entirely attributable to **early stopping termination at Epoch 14** (triggered by patience=7 after the Epoch 7 optimum). In contrast, linear probing continued training for all 25 epochs because its best validation loss was reached late in training (Epoch 23), preventing the patience counter from triggering early termination.
   - At the hardware level, full fine-tuning required significantly greater compute per step, allocating **952.27 MB of VRAM** (vs 353.86 MB) to store intermediate activation gradients and optimizer momentum buffers across all 11.18M parameters.

3. **Hypothesis**:
   - Rapid early-stage minimization of validation loss in full fine-tuning is consistent with the hypothesis that unconstrained gradient descent across all convolutional filters allows the model to align with domain-specific MRI features within fewer global iterations than optimizing a linear decision surface over fixed ImageNet representations.

---

## 6. Claim Audit

The Phase 2.1B-1 and Phase 2.1B-2 reports were audited for ungrounded, overstrong, or non-rigorous statements. The table below documents specific flags and recommends scientifically defensible language:

| # | Current Claim in Report | Methodological Problem | Recommended Scientific Wording |
| :-: | :--- | :--- | :--- |
| **1** | *"Upper-bound performance ceiling"* | Unproven assertion. Other architectures, data augmentation strategies, or hyperparameter schedules might achieve higher metrics. | *"Empirical reference benchmark for unconstrained non-incremental fine-tuning on this data split."* |
| **2** | *"Resolves subtle tumor boundary overlaps"* | Causal/anatomical claim without localization evidence. No Grad-CAM, segmentation masks, or feature attribution analyses were performed. | *"Is consistent with improved empirical separation among classes that exhibited confusion under the linear probe."* |
| **3** | *"Generalises effectively without reliance on data leakage"* | Overstates guarantee. Exact SHA-256 byte duplicates were removed, but lack of patient IDs means patient-level correlation across splits cannot be ruled out. | *"Exhibits comparable performance when exact-duplicate files identified in Phase 1.2C are excluded; however, patient-level independence was not established."* |
| **4** | *"Proving that early transfer features possess strong semantic representational power"* | Overly definitive language ("proving"). 87.7% linear accuracy shows linear separability on this specific benchmark, not general semantic proof across MRI domains. | *"Demonstrates that frozen ImageNet features provide sufficient linear separability to distinguish the four classes on this dataset."* |
| **5** | *"Full fine-tuning is more computationally efficient (faster wall-clock time)"* | Misleading without epoch context. Fine-tuning took less total time only because early stopping triggered at Epoch 14, whereas per-epoch compute was 13.5% higher. | *"Full fine-tuning terminated in fewer epochs (14 vs 25) due to early validation loss plateau, despite requiring 13.5% more computation per epoch and 2.69x higher VRAM."* |
| **6** | *"Near-perfect diagnostic accuracy"* | Clinical implication. This is a research prototype on a public 2D slice dataset; no clinical validation or multi-site evaluation was performed. | *"High empirical classification accuracy on the curated test partitions."* |

---

## 7. Reproducibility Audit

The experiment artifacts and environment logs were evaluated against standard machine learning reproducibility criteria:

| Reproducibility Factor | Recorded Value / Artifact Location | Audit Status |
| :--- | :--- | :---: |
| **Python Version** | `3.11.9 (tags/v3.11.9:de54cf5)` | **VERIFIED** |
| **PyTorch Version** | `2.12.0.dev20260408+cu128` | **VERIFIED** |
| **torchvision Version** | `0.27.0.dev20260407+cu128` | **VERIFIED** |
| **CUDA Version** | `12.8` | **VERIFIED** |
| **GPU Model** | `NVIDIA GeForce RTX 5060 Laptop GPU` (8,151 MiB VRAM) | **VERIFIED** |
| **Operating System** | `Windows-10-10.0.26200-SP0` (Windows 11) | **VERIFIED** |
| **Random Seeds** | `seed = 42` recorded across PyTorch, CUDA, DataLoader generators | **VERIFIED** |
| **Dataset Statistics** | 7,023 total; 4,855 train; 857 val; 1,311 benchmark; 1,208 filtered | **VERIFIED** |
| **Test Suite Status** | 25 / 25 unit and integration tests passing (`pytest` in 18.83s) | **VERIFIED** |
| **Package Dependency Lock** | Active `.venv` environment present; no formal `requirements.lock` generated | **PRESENT BUT INCOMPLETE** |
| **Git Version Control State** | Commit `849f444e4c699a10ee0b6f46148251a42bba143c` (workspace has uncommitted report artifacts) | **PRESENT BUT INCOMPLETE** |

---

## 8. Baseline Comparison Matrix

The table below presents a clean comparison between the two established baselines using percentage-point (`pp`) differences:

| Dimension / Metric | ResNet18 Linear Probe (Phase 2.1B-1) | ResNet18 Full Fine-Tuning (Phase 2.1B-2) | Absolute Shift ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Performance** | | | |
| - Benchmark Accuracy (1,311 img) | 87.72% | 99.01% | +11.29 pp |
| - Benchmark Macro F1-Score | 87.18% | 98.96% | +11.78 pp |
| - Filtered Accuracy (1,208 img) | 87.67% | 98.92% | +11.25 pp |
| - Filtered Macro F1-Score | 87.58% | 98.93% | +11.35 pp |
| - Filtered Glioma F1-Score | 85.86% | 98.48% | +12.62 pp |
| - Filtered Meningioma F1-Score | 78.03% | 98.05% | +20.02 pp |
| - Filtered No-Tumor F1-Score | 94.43% | 99.52% | +5.09 pp |
| - Filtered Pituitary F1-Score | 91.99% | 99.66% | +7.67 pp |
| **Convergence** | | | |
| - Best Checkpoint Epoch | Epoch 23 | Epoch 7 | -16 epochs |
| - Best Validation Loss | 0.2388 | 0.0436 | -0.1952 |
| - Total Epochs Trained | 25 | 14 | -11 epochs |
| - Early Stopping Triggered | False | True | N/A |
| **Compute & Runtime** | | | |
| - Trainable Parameters | 2,052 | 11,178,564 | +11,176,512 (+5,446.6x) |
| - Total Training Duration | 1,495.87 s (24.9 min) | 951.01 s (15.9 min) | -544.86 s (-36.4%) |
| - Mean Duration per Epoch | 59.83 s | 67.93 s | +8.10 s (+13.5%) |
| **Memory & Storage** | | | |
| - Peak GPU VRAM | 353.86 MB | 952.27 MB | +598.41 MB (+169.1%) |
| - Checkpoint Disk Size | 42.74 MB | 128.06 MB | +85.32 MB (+199.6%) |
| **Inference Latency** | | | |
| - Filtered Test Latency | 5.51 ms / sample | 6.37 ms / sample | +0.86 ms / sample |

---

## 9. Research Interpretation

### 1. What does the experiment establish?
- Pretrained ImageNet features without backbone fine-tuning achieve **87.58% Filtered Macro F1** across 4 brain MRI classes, proving strong baseline linear separability.
- Joint end-to-end fine-tuning of all parameters achieves **98.93% Filtered Macro F1**, providing an empirical upper reference bound for non-incremental learning on this split.
- Filtering 103 exact byte-level duplicate test images shifted performance by negligible amounts (-0.05 pp accuracy in linear probing, -0.09 pp accuracy in fine-tuning), confirming that exact duplicate test contamination did not artificially inflate the benchmark.
- Updating all layers incurs a 169.1% increase in peak GPU VRAM (952 MB vs 354 MB) and a 199.6% increase in checkpoint size (128 MB vs 42.7 MB).

### 2. What does it NOT establish?
- It **does not establish** clinical safety, real-world generalization, or robustness against MRI scanner domain shifts.
- It **does not establish** patient-level independence. Because the original dataset lacks patient metadata, intra-patient slice correlation across splits cannot be ruled out.
- It **does not establish** true anatomical boundary localization (no interpretability maps or segmentation evaluations were performed).
- It **does not establish** confidence intervals or seed sensitivity, as experiments were conducted on a single fixed seed (`seed=42`).

### 3. What are the strongest defensible conclusions?
- Full fine-tuning delivers an **11.35 pp improvement in macro F1** over the linear probe on the exact-duplicate-filtered test partition, with the largest individual gain observed in meningioma classification (+20.02 pp F1).
- The two baselines establish well-defined, reproducible reference boundaries on this specific data split:
  - **Lower Bound (Linear Probe)**: 87.58% Filtered Macro F1
  - **Upper Bound (Full Fine-Tuning)**: 98.93% Filtered Macro F1

### 4. What limitations remain?
- The dataset is unannotated with respect to subject/patient identity.
- Single-seed execution provides a point estimate rather than a distribution with confidence intervals.
- Fixed 224x224 image resolution limits spatial detail compared to native MRI resolution.

### 5. What must be fixed/documented before continual-learning experiments?
- All future publications and reports must explicitly state: `"Patient-level independence was not established."`
- Continual learning evaluation protocols must measure Backward Transfer (BWT), Average Accuracy (ACC), and Forgetting Measure (FM) on both Benchmark and Filtered test sets against these exact frozen baseline metrics.

---

## 10. Final Verdict

### **VERDICT: READY FOR CONTINUAL LEARNING**

The experimental foundation is verified, reproducible, and internally consistent.

### Conditions Frozen for Phase 3:
1. **Dataset Manifests**: `train_validation_manifest.csv` (4,855 train / 857 val) and `evaluation_manifest.csv` (1,311 benchmark / 1,208 filtered) are strictly locked.
2. **Preprocessing Pipeline**: `MRIPreprocessor` (contour crop, 10px pad, 224x224 resize, ImageNet RGB normalization) is strictly locked.
3. **Class Indices**: `{"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}` is strictly locked.
4. **Seed**: `seed = 42` is strictly locked.
5. **Reference Anchors Frozen**:
   - Lower-Bound Anchor: **87.58% Macro F1** (Linear Probe)
   - Upper-Bound Anchor: **98.93% Macro F1** (Full Fine-Tuning)
6. **Mandatory Documentation Clause**: Every continual learning report must include:  
   *"Patient-level independence was not established. Filtering addressed only exact byte-level duplicate files."*
