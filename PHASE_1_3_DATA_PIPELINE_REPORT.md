# PHASE 1.3 — Reproducible Data Pipeline Preparation Report

**Project Title:** Brain Tumor MRI Classification & Continual Learning  
**Phase:** 1.3 (Reproducible Data Pipeline Preparation)  
**Date:** October 4, 2026  
**Auditor:** Senior Computer Vision Research Engineer  
**Status:** Data Pipeline Implementation & Verification Complete  

---

## 1. Executive Summary & Architecture Overview

Phase 1.3 prepares a PyTorch data loading pipeline for ResNet18 transfer learning baseline models. The pipeline implements a 15% stratified train/validation split derived strictly from the 5,712 original training images, leaving the testing sets completely untouched during training and validation.

### Key Data Pipeline Capabilities:
1. **Stratified Train/Validation Split:** 4,855 training samples (85.0%) and 857 validation samples (15.0%) with preserved class proportions.
2. **Deterministic Manifest Control:** Governed by machine-readable CSV manifests saved in `artifacts/reports/`.
3. **ResNet18 Input Processing:** Preprocessing crops brain contours (safety padding $= 10\text{ px}$), resizes images to $224 \times 224$, and normalizes with ImageNet standard mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
4. **Data Augmentations:** Training loader applies conservative medical imaging augmentations (`torchvision.transforms`: horizontal flip $p=0.5$, random rotation $\pm 15^\circ$, brightness/contrast $\pm 0.15$). Validation and test loaders use deterministic, non-random evaluation transforms.
5. **Dual-Evaluation DataLoaders:** Separate PyTorch `DataLoader` instances for the Raw Benchmark Test Set (1,311 images) and the Filtered Test Set (1,208 images).

---

## 2. Dataset Split & Class Distribution (Requirement 1 & 6)

The class distribution across all data pipeline splits is documented in [data_pipeline_class_counts.csv](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/data_pipeline_class_counts.csv):

### Detailed Class Count Summary Table:

| Tumor Class | 85% Train Count | 15% Val Count | Original Train Total | Raw Benchmark Test Count | Excluded Test Images | Filtered Test Count | Total Unique Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `glioma` | 1,123 | 198 | **1,321** | 300 | 0 | **300** | 1,621 |
| `meningioma` | 1,138 | 201 | **1,339** | 306 | 2 | **304** | 1,643 |
| `notumor` | 1,356 | 239 | **1,595** | 405 | 96 | **309** | 1,904 |
| `pituitary` | 1,238 | 219 | **1,457** | 300 | 5 | **295** | 1,752 |
| **TOTAL** | **4,855** | **857** | **5,712** | **1,311** | **103** | **1,208** | **6,920** |

---

## 3. Data Pipeline Implementation Details (Requirements 3 & 4)

### 3.1 Preprocessing & Image Processing
* **Module:** `src/data/preprocessor.py` (`MRIPreprocessor`) & `src/data/dataset.py` (`MRIDataset`).
* **Input Resolution:** Original variable image sizes (e.g. 512x512, 225x225).
* **Skull Stripping:** OpenCV contour bounding box detection with 10 px safety padding and boundary clipping.
* **Target Model Input:** $224 \times 224 \times 3$ (`torch.float32`).
* **Normalization:** ImageNet channel normalization (`(x - mean) / std`).

### 3.2 PyTorch DataModule Factory
* **Module:** `src/data/pipeline_factory.py` (`get_mri_dataloaders`).
* **Batch Size:** Configurable (default 32).
* **Multiprocessing:** Configurable `num_workers` (default 2/4).
* **Memory Management:** `pin_memory=True` supported for GPU transfer.
* **Shuffling:** Training loader uses a `torch.Generator` initialized with `seed=42`.
* **Label Mapping:** `{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}`.

---

## 4. Integrity Verification & Automated Tests (Requirement 5 & 10)

All 20 unit tests across the codebase were executed via `pytest`:

```bash
.venv\Scripts\python.exe -m pytest
```

### Test Suite Results (`20 / 20 PASSED` in 11.30s):

1. **`test_no_train_val_path_overlap`**: PASSED (0 overlapping paths between 4,855 Train and 857 Val samples).
2. **`test_no_val_test_path_overlap`**: PASSED (0 overlapping paths between Validation and Testing sets).
3. **`test_zero_exact_hash_overlap_train_and_filtered_test`**: PASSED (0 SHA-256 hash overlap between Train and Filtered Test).
4. **`test_dataloader_tensor_properties_shape_dtype_finite`**: PASSED (Batch shape `(B, 3, 224, 224)`, dtype `torch.float32`, no NaN/Inf, labels `0..3`).
5. **`test_reproducible_split_generation`**: PASSED (Identical manifest files produced across separate runs with seed 42).
6. **`test_validation_and_test_transforms_determinism`**: PASSED (Consecutive fetches of same validation sample produce bit-identical tensors).
7. **`test_class_counts_match_generated_manifests`**: PASSED (Dataset lengths match CSV manifest counts).

---

## 5. Artifact Checklist

| Artifact Name | Path | Purpose |
| :--- | :--- | :--- |
| **Pipeline Report** | `PHASE_1_3_DATA_PIPELINE_REPORT.md` | Comprehensive documentation of data pipeline architecture. |
| **Train/Val Manifest** | `artifacts/reports/train_validation_manifest.csv` | Machine-readable manifest of 4,855 Train and 857 Val image paths. |
| **Class Counts Summary** | `artifacts/reports/data_pipeline_class_counts.csv` | Complete tabular breakdown across all pipeline splits. |
| **Dataset Module** | `src/data/dataset.py` | `MRIDataset` PyTorch class with preprocessing & transforms. |
| **Pipeline Factory** | `src/data/pipeline_factory.py` | DataLoader factory & manifest generator. |
| **Unit Test Suite** | `tests/test_data_pipeline_phase1_3.py` | Pytest integrity suite for pipeline verification. |

---

## 6. Scientific Disclaimers & Restrictions Compliance

1. **Patient-Level Disclaimer:** Image-level deduplication guarantees zero exact file-level byte overlap. It does not guarantee patient-level slice separation due to raw Kaggle JPEG metadata limitations.
2. **Restrictions Compliance:** Zero models were trained, no GPU benchmarks were executed, `data/raw/` was un-mutated, and original benchmark splits remain intact.

---

## 7. Unresolved Concerns & Next Steps

* **Unresolved Concerns:** None. Data pipeline is completely verified and ready.
* **Next Recommended Phase:** Proceed to **Phase 2 — Baseline ResNet18 Transfer Learning Model Development**, implementing the training loop, validation tracking, and model checkpointing.
