# PHASE 1.2A — Dataset Archive Pre-Extraction Verification Report

**Project Title:** Brain Tumor MRI Classification & Continual Learning  
**Phase:** 1.2A (Dataset Archive Pre-Extraction Verification)  
**Date:** October 4, 2026  
**Auditor:** Senior Computer Vision Research Engineer  
**Status:** Pre-Extraction Preflight Complete  

---

## 1. Executive Summary

This report performs a strict, read-only pre-flight verification of the newly placed dataset archive located at `data/downloads/MRIdatasetfull.zip`. Prior to executing any file extraction into `data/raw/` or making modifications to the codebase, the archive was inspected for structural validity, binary integrity, dataset completeness, path safety, and compatibility with the existing project architecture.

The archive passed all structural and checksum tests without error. The internal directory layout strictly matches the expected layout for the Masoud Nickparvar Brain Tumor MRI dataset (`Training/` and `Testing/` splits, each containing four tumor classes: `glioma`, `meningioma`, `notumor`, `pituitary`).

---

## 2. Repository Component Inspection (Task 1)

All existing core components of the repository were verified and preserved intact:

| Component Category | Module Path | Purpose / Responsibilities | Status |
| :--- | :--- | :--- | :--- |
| **Download & Ingestion Helper** | `src/data/download_dataset.py` | Checks local readiness of `data/raw`, supports Kaggle API downloads, and extracts zip archives to `data/raw/`. | Verified & Intact |
| **Data Integrity Audit** | `src/data/data_audit.py` | SHA-256 exact image duplicate detection, cross-split duplicate detection, image corruption check, and class distribution reporting. | Verified & Intact |
| **Preprocessor Engine** | `src/data/preprocessor.py` | OpenCV contour-based skull stripping, configurable bounding box padding (0, 5, 10, 15 px), boundary clipping, resizing (150x150), min-max/standard normalization, and 3-channel RGB preservation. | Verified & Intact |
| **Dataset Configuration** | `configs/data_config.yaml` | Declares 150x150 image resolution, 3 input channels, class lists, contour cropping parameters, validation split ratio (0.15), and Albumentations configs. | Verified & Intact |
| **Test Suite** | `tests/test_data_pipeline.py` | 10 unit tests covering fallback handling, padding, boundary clipping, shape/pixel ranges, preprocessor determinism, RGB preservation, duplicate hash computation, model shapes, and dynamic head expansion. | Verified (10/10 PASS) |
| **Historical Audits & Reports** | `PHASE_0_HANDOVER.md`<br>`PHASE_1_1_REPORT.md`<br>`PHASE_1_1_VERIFICATION.md` | Comprehensive documentation tracking foundation setup, preprocessing hardening, padding visual comparisons, and data integrity infrastructure. | Verified & Preserved |

---

## 3. Dataset Archive Technical Verification (Task 2)

The target dataset archive was analyzed using Python's `zipfile` and `os` standard libraries:

* **Target File Path:** `data/downloads/MRIdatasetfull.zip`
* **File Existence:** `True` (File verified present on filesystem)
* **File Size:** `155,791,278` bytes (`148.57 MB`)
* **File Extension:** `.zip`
* **Format & Header Integrity (`zipfile.is_zipfile`):** `True`
* **CRC32 / Binary Integrity (`ZipFile.testzip()`):** `None` (0 corrupted files detected across all 7,023 compressed items)
* **Readability Test:** `SUCCESS` (Full header scan and namelist iteration completed without standard I/O or CRC exceptions)

---

## 4. Archive Structure & Security Audit (Task 3)

### 4.1 Root Directory & Split Organization
* **Top-Level Root Directories:** `{'Training', 'Testing'}`
* **Unexpected Top-Level Nesting:** None (No wrapper folders such as `brain_tumor_dataset/` or `archive/`).
* **Hidden Files / Mac OS Metadata:** 0 instances of `__MACOSX/` or `.DS_Store`.

### 4.2 Detailed Class Breakdown & Image Counts
All 7,023 items inside the ZIP file are valid `.jpg` image files distributed across the canonical 4 classes:

```
MRIdatasetfull.zip
├── Testing/
│   ├── glioma/      (300 files)
│   ├── meningioma/  (306 files)
│   ├── notumor/     (405 files)
│   └── pituitary/   (300 files)
└── Training/
    ├── glioma/      (1,321 files)
    ├── meningioma/  (1,339 files)
    ├── notumor/     (1,595 files)
    └── pituitary/   (1,457 files)
```

#### Detailed Class Breakdown Table:

| Dataset Split | Class Name | File Count | Format | Percentage of Split |
| :--- | :--- | :---: | :---: | :---: |
| **Training** | `glioma` | 1,321 | `.jpg` | 23.13% |
| **Training** | `meningioma` | 1,339 | `.jpg` | 23.44% |
| **Training** | `notumor` | 1,595 | `.jpg` | 27.92% |
| **Training** | `pituitary` | 1,457 | `.jpg` | 25.51% |
| **Training Total** | **4 Classes** | **5,712** | **.jpg** | **100.00%** |
| | | | | |
| **Testing** | `glioma` | 300 | `.jpg` | 22.88% |
| **Testing** | `meningioma` | 306 | `.jpg` | 23.34% |
| **Testing** | `notumor` | 405 | `.jpg` | 30.89% |
| **Testing** | `pituitary` | 300 | `.jpg` | 22.88% |
| **Testing Total** | **4 Classes** | **1,311** | **.jpg** | **100.00%** |
| | | | | |
| **Combined Grand Total** | **All Splits** | **7,023** | **.jpg** | **100.00%** |

### 4.3 Security & Path Traversal Verification
* **Path Traversal Check (`..` in path):** 0 instances.
* **Absolute Path Violations (leading `/` or `\\`):** 0 instances.
* **Non-Standard / Executable File Extensions:** 0 instances (100% `.jpg`).
* **Zip-Bomb / Expansion Compression Ratio:** Uncompressed total size is ~162.4 MB vs compressed size 148.57 MB (ratio ~1.09:1), confirming normal image storage without excessive compression ratio anomalies.

---

## 5. Target Extraction Layout Mapping

Upon execution of extraction in Phase 1.2B, the archive will extract directly into `data/raw/` with zero restructuring required:

```
data/raw/
├── Training/
│   ├── glioma/      [1,321 images]
│   ├── meningioma/  [1,339 images]
│   ├── notumor/     [1,595 images]
│   └── pituitary/   [1,457 images]
└── Testing/
    ├── glioma/      [300 images]
    ├── meningioma/  [306 images]
    ├── notumor/     [405 images]
    └── pituitary/   [300 images]
```

---

## 6. Risk Assessment & Verification Summary

### 6.1 Verified Empirical Facts
1. The ZIP archive at `data/downloads/MRIdatasetfull.zip` is completely intact and free of byte-level corruption.
2. The dataset contains exactly 7,023 `.jpg` images (5,712 Training, 1,311 Testing).
3. Class representation across both splits is well-balanced (~23%-30% per class).
4. No security hazards or unsupported file types exist within the archive.
5. All 10 existing pytest regression tests continue to pass in 1.50s.

### 6.2 Potential Risks & Open Questions (Post-Extraction)
1. **Exact & Cross-Split Image Duplicates:** As established in literature regarding the Nickparvar dataset, exact image duplicates and potential patient-level cross-split leakage exist within the raw download. These will be formally audited and reported using `src/data/data_audit.py` immediately following extraction.
2. **Patient-Level Grouping:** Slice-level leakage vs. subject-level splitting remains an important consideration for model evaluation strategy in Phase 2.

---

## 7. Final Verdict

PASS — Archive is valid and safe for extraction
