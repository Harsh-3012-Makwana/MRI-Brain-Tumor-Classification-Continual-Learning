# PHASE 1.2C — Dataset Duplicate Forensics Report

**Project Title:** Brain Tumor MRI Classification & Continual Learning  
**Phase:** 1.2C (Dataset Duplicate Forensics & Contamination Investigation)  
**Date:** October 4, 2026  
**Auditor:** Senior Computer Vision Research Engineer  
**Status:** Read-Only Duplicate Forensics Complete  

---

## 1. Audit Implementation Technical Inspection (Step 1)

A technical review of `src/data/data_audit.py` was conducted to clarify audit methodology and metric definitions:

1. **SHA-256 Hash Computation:** Hashes are computed on **raw file bytes** using `hashlib.sha256()`. The file is opened in binary read mode (`with open(file_path, "rb") as f:`) and updated in 64 KB chunks (`65,536` bytes).
2. **Hash Target:** Identifies exact byte-level image file identity.
3. **Duplicate Cluster Formation:** File paths are grouped into clusters indexed by identical SHA-256 string keys.
4. **Cross-Split Duplicate Identification:** Evaluated via set intersection between training hash keys and testing hash keys: `set(hashes_by_split["training"].keys()) & set(hashes_by_split["testing"].keys())`.
5. **Metric Clarification:**
   - In `data_audit.py`, reported counts represent **unique duplicate clusters (hashes)**, not individual file counts.
   - `Exact Duplicates Found` (166 in `data_audit.py`) represents clusters with $\ge 2$ copies within either the Training split (163) or Testing split (3).
   - `Cross-Split Duplicates` (79 in `data_audit.py`) represents unique image hashes present simultaneously in BOTH Training and Testing splits.

---

## 2. Duplicate Manifest & Forensic Statistics (Step 2 & Step 5)

A comprehensive CSV manifest was generated mapping all duplicate file entries across the dataset:

* **Manifest File Path:** [duplicate_manifest.csv](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/reports/duplicate_manifest.csv)
* **Manifest Schema:** `cluster_id`, `sha256_hash`, `total_copies`, `splits`, `classes`, `is_cross_split`, `is_label_conflict`, `file_paths`

### 2.1 Complete Dataset Forensic Summary

| Forensic Metric | Cluster / Hash Count | File / Image Count | Percentage of Total Images |
| :--- | :---: | :---: | :---: |
| **Total Extracted Images on Disk** | N/A | **7,023** | 100.00% |
| **Unique Image Contents (Unique SHA-256 Hashes)** | **6,726** | 6,726 | 95.77% |
| **Total Duplicate Clusters ($\ge 2$ Files)** | **194** | **497** | 7.08% |
| — Confined Entirely to Training Split | 112 | 254 | 3.62% |
| — Confined Entirely to Testing Split | 3 | 6 | 0.09% |
| — **Cross-Split Clusters (Training & Testing)** | **79** | **237** | **3.37%** |
| — In Training Split | — | 134 | 1.91% |
| — In Testing Split | — | 103 | 1.47% |
| **Label Conflict Clusters (Conflicting Classes)** | **0** | **0** | **0.00%** |
| **Affected Tumor Classes** | **4 / 4** | 7,023 | 100.00% |

---

## 3. Cross-Split Contamination Investigation (Step 3)

For all 79 cross-split duplicate clusters (involving 237 images: 134 in Training and 103 in Testing):

1. **Same Image, Same Class, Cross-Split:** `79 clusters` (100% of cross-split duplicate clusters share identical class labels in both Training and Testing splits).
2. **Same Image, Conflicting Labels:** `0 clusters` (Zero label contradictions exist; e.g., no image is labeled `glioma` in training and `meningioma` in testing).
3. **Multi-Copy Within-Split Breakdown:**
   - Of the 79 cross-split clusters, 51 clusters also contain multiple duplicate copies *within* the Training split.
   - Testing images impacted by cross-split duplication: 103 out of 1,311 test images (**7.86%** of the raw test set consists of images identical to training samples).

---

## 4. Visual Inspection Artifact (Step 4)

A visual contact sheet displaying representative duplicate clusters across splits was created:

* **Artifact Path:** [duplicate_inspection.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/duplicate_inspection.png)
* **Visual Verification Note:** Visual inspection confirms exact pixel identity across duplicate files. Byte identity establishes identical JPEG encodings. *Scientific Disclaimer:* Byte-level exact identity confirms file duplication; patient-level identity cannot be claimed from visual similarity alone without DICOM header patient IDs.

---

## 5. Methodological Research Tradeoffs (Step 6)

Before initiating Phase 2, three potential split management strategies are evaluated:

### Option A: Preserve Original Split & Disclose Contamination
* **Description:** Retain raw Kaggle `Training/` (5,712) and `Testing/` (1,311) split without modification. Report the 7.86% test-set leakage in research documentation.
* **Pros:** Preserves direct comparability with published baseline papers using the raw Nickparvar benchmark split.
* **Cons:** Test accuracy metrics will be artificially optimistic due to memorization of 103 leaked test images.

### Option B: Deterministic Deduplicated Research Split
* **Description:** Apply a documented, deterministic deduplication procedure (e.g., purge cross-split duplicate copies from `Testing/` or assign whole duplicate clusters strictly to `Training/`).
* **Pros:** Scientifically rigorous; guarantees zero data leakage between training and testing splits.
* **Cons:** Deviates slightly from raw Kaggle image counts, requiring clear documentation for external reproducibility.

### Option C: Dual-Evaluation Protocol (Recommended)
* **Description:** Retain original raw test set for direct benchmark comparability against literature, while simultaneously creating a leakage-filtered evaluation set to report true clinical generalization.
* **Pros:** Provides complete scientific transparency — enables both direct comparison with prior work and rigorous leak-free performance evaluation.
* **Cons:** Requires maintaining two evaluation pipelines in data loading infrastructure.

---

## 6. Audit Limitations & Remaining Blockers

### 6.1 Limitations
* **Exact vs. Near-Duplicates:** SHA-256 detects byte-identical files. Crop variants, resolution changes, or minor compression artifacts of identical MRI scans are not captured by cryptographic hashing.
* **Patient Metadata Absence:** Header-based patient grouping remains impossible due to JPEG stripping.

### 6.2 Remaining Blockers
* **Decision on Split Strategy:** A formal decision between Option A, Option B, or Option C is required before split manifest generation and PyTorch `DataLoader` instantiation.

---

## 7. Recommended Next Decision

**Recommendation:** Adopt **Option C (Dual-Evaluation Protocol)**. Build a deterministic manifest generator in Phase 1.3 that supports both raw benchmark evaluation and deduplicated leak-free evaluation.
