# PHASE 1.2B — Controlled Dataset Extraction Report

**Project Title:** Brain Tumor MRI Classification & Continual Learning  
**Phase:** 1.2B (Controlled Dataset Extraction & Filesystem Validation)  
**Date:** October 4, 2026  
**Auditor:** Senior Computer Vision Research Engineer  
**Status:** Extraction & Filesystem Verification Completed Successfully  

---

## 1. Pre-Extraction Safety & Extraction Status

Prior to dataset extraction, all pre-extraction safety protocols were executed and verified:

1. **Pre-Extraction Destination Check:** `data/raw/` was inspected and verified to be completely empty (0 files/directories).
2. **Archive Re-Verification:** Binary integrity of `data/downloads/MRIdatasetfull.zip` was re-verified via `zipfile.ZipFile.testzip()`. Result: `0` corrupted files.
3. **Path Traversal Security Audit:** All 7,023 entry paths in the ZIP archive were scanned for relative directory traversal (`..`), leading slashes (`/` or `\`), or non-standard file extensions. Result: `0` unsafe paths detected.
4. **Extraction Execution:** Controlled extraction was executed extracting directly into `data/raw/`. The original source ZIP archive at `data/downloads/MRIdatasetfull.zip` was preserved untouched.

---

## 2. Actual Filesystem Verification & Archive Comparison

An independent, recursive scan of `data/raw/` on disk was performed post-extraction to measure filesystem counts and compare them against archive specifications.

### 2.1 Summary Metrics
* **Total Extracted Files:** `7,023`
* **File Extensions Present:** `100% .jpg` (7,023 files)
* **Unexpected / Non-Image Files:** `0`
* **Missing Class Directories:** `0` (All 8 split-class subdirectories present)

### 2.2 Detailed Class Breakdown & Comparison Table

| Dataset Split | Class Directory | Archive Target Count | Actual Extracted Files | Count Match | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Training** | `glioma` | 1,321 | 1,321 | **MATCH** | PASS |
| **Training** | `meningioma` | 1,339 | 1,339 | **MATCH** | PASS |
| **Training** | `notumor` | 1,595 | 1,595 | **MATCH** | PASS |
| **Training** | `pituitary` | 1,457 | 1,457 | **MATCH** | PASS |
| **Training Total** | **4 Subdirectories** | **5,712** | **5,712** | **MATCH** | **PASS** |
| | | | | | |
| **Testing** | `glioma` | 300 | 300 | **MATCH** | PASS |
| **Testing** | `meningioma` | 306 | 306 | **MATCH** | PASS |
| **Testing** | `notumor` | 405 | 405 | **MATCH** | PASS |
| **Testing** | `pituitary` | 300 | 300 | **MATCH** | PASS |
| **Testing Total** | **4 Subdirectories** | **1,311** | **1,311** | **MATCH** | **PASS** |
| | | | | | |
| **Grand Total** | **All 8 Directories** | **7,023** | **7,023** | **MATCH** | **PASS** |

---

## 3. Data Integrity & Duplicate Audit Findings

The data integrity audit was conducted using `src/data/data_audit.py` alongside extended image tensor analysis:

### 3.1 Image Readability & Corruption
* **Corrupted / Unreadable Images:** `0`
* OpenCV (`cv2.imread`) successfully decoded 100% of the 7,023 image files without any header or bitstream decoding errors.

### 3.2 Exact Byte-Level Duplicates (SHA-256)
* **Within-Split Duplicate Hash Clusters:** `166` clusters (multiple files within either `Training/` or `Testing/` possessing identical SHA-256 hashes).
* **Cross-Split Duplicate Hash Clusters:** `79` clusters.
  - **Data Leakage Risk:** 79 distinct image content hashes exist simultaneously in BOTH `Training/` and `Testing/` splits.
  - **Impact:** Testing on these images would evaluate a model on training-seen data. This cross-split duplication must be handled before final model evaluation.

### 3.3 Image Resolution & Channel Distribution
* **Image Channels:** 100% of images (`7,023 / 7,023`) are formatted with `3` color channels in memory (`height x width x 3`).
* **Color Mode Analysis:**
  - **Monochrome stored as 3-Channel RGB:** `6,894` images (98.16%) have identical Red, Green, and Blue channel intensities across all pixels.
  - **True Color RGB / Overlays:** `129` images (1.84%) contain non-identical channel values (due to color burn-in, annotations, or colormaps in raw JPEGs).
* **Resolution Distribution (387 Unique Resolutions):**
  - `512 x 512`: 4,742 images (**67.52%**)
  - `225 x 225`: 332 images (**4.73%**)
  - `630 x 630`: 90 images (**1.28%**)
  - `236 x 236`: 81 images (**1.15%**)
  - `201 x 251`: 58 images (**0.83%**)
  - *Other (382 resolutions):* 1,690 images (**24.49%**)

---

## 4. Visual Sample Generation (Task 5)

A deterministic class-wise sample grid was generated and saved:

* **Artifact Path:** [dataset_samples.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/dataset_samples.png)
* **Configuration:**
  - **Grid Size:** $4 \times 4$ (4 classes $\times$ 4 random samples per class)
  - **Random Seed:** `42` (Fixed for reproducibility)
  - **Preprocessing Status:** None (Raw, original images displayed directly as loaded from disk).
  - **Annotations:** Displayed image filename and raw resolution $(W \times H)$ per sample tile.

---

## 5. Audit Module Capabilities & Limitations

### 5.1 Existing Audit Module Integrity
The project's primary audit module ([src/data/data_audit.py](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/src/data/data_audit.py)) was run without modifying its source code, adhering to project guidelines.

### 5.2 Identified Module Gaps
* **Image Resolution & Channel Tracking:** `data_audit.py` measures file existence, OpenCV readability, SHA-256 hashes, exact duplicates, and cross-split duplicates, but does not natively record width/height histograms or channel equality metrics. Extended verification was performed via dedicated inspection scripts without mutating `data_audit.py`.
* **Patient Metadata Absence:** Because raw Kaggle JPG files omit DICOM headers and patient IDs, patient-level slice clustering cannot be inferred from metadata alone; exact SHA-256 hash matching serves as the baseline anti-leakage barrier.

---

## 6. Remaining Blockers & Next Phase Recommendations

### 6.1 Blockers & Pre-Training Requirements
1. **Cross-Split Deduplication:** The 79 cross-split duplicate image clusters must be purged from `Testing/` or reassigned to ensure zero test set data contamination.
2. **Stratified Validation Split:** As declared in `configs/data_config.yaml` (`val_split_from_train: 0.15`), a 15% stratified validation set must be carved out of `Training/` before model training.

### 6.2 Recommendation for Next Phase
Proceed to **Phase 1.3 / Data Pipeline Preparation**:
1. Implement a clean dataset loader and data manifest generator that excludes cross-split duplicate hashes from validation/test evaluation.
2. Integrate `MRIPreprocessor` (with configurable padding $= 10\text{ px}$ as validated in Phase 1.1) into the PyTorch `Dataset` pipeline.
3. Prepare the stratified train/validation/test PyTorch `DataLoader` instances with conservative medical augmentations.
