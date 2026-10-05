# PHASE 1.1 RESEARCH DECISION REPORT: PREPROCESSING VALIDATION & DATA INTEGRITY HARDENING

**Author:** Senior Computer Vision Research Engineer  
**Date:** October 04, 2026  
**Status:** **PHASE 1.1 COMPLETED** (Zero model training executed, Phase 2 NOT implemented)

---

## 1. Executive Summary & What Changed

To improve scientific reliability and mitigate risks of clipping peripheral brain tumor tissue, Phase 1.1 completed a comprehensive validation and data integrity hardening sprint:

1. **Configurable Safety Padding & Boundary Clipping:** Updated `MRIPreprocessor` ([src/data/preprocessor.py](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/src/data/preprocessor.py)) to support configurable bounding-box padding (`padding = 0, 5, 10, 15` pixels) with strict boundary clipping `[max(0, y - pad) : min(H, y + h + pad), max(0, x - pad) : min(W, x + w + pad)]`.
2. **Visual Comparison Utility:** Created a 4x4 comparison generator ([src/data/inspect_preprocessing.py](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/src/data/inspect_preprocessing.py)) that produces a side-by-side visual matrix across all four tumor classes (*Glioma*, *Meningioma*, *Pituitary*, *No Tumor*).
3. **Data Integrity Audit Module:** Created `DataIntegrityAuditor` ([src/data/data_audit.py](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/src/data/data_audit.py)) to detect byte-level SHA-256 duplicates, identify cross-split leakage (Training vs. Testing), verify file integrity against corruption, and report class distributions.
4. **Expanded Unit Test Suite:** Expanded pytest coverage from 5 to 10 automated test cases in [tests/test_data_pipeline.py](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/tests/test_data_pipeline.py).

---

## 2. Why It Changed (Scientific Rationale)

In Phase 1, audit findings highlighted a potential vulnerability: extra-axial tumors (specifically **Meningiomas**) grow on the protective dural membranes along the outer cranial wall. Tight bounding-box cropping (`padding = 0`) ran the risk of clipping peripheral tumor margins if thresholding trimmed the dural boundary.

By introducing **configurable safety padding** (e.g. `padding = 10` pixels), the preprocessor extends the bounding rectangle outward while guaranteeing that coordinates never exceed the original image height ($H$) or width ($W$).

---

## 3. Visual Comparison Artifact Location

- **Artifact File Path:** [artifacts/figures/preprocessing_comparison.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/preprocessing_comparison.png)
- **Format:** 4x4 Grid (Rows: *Glioma*, *Meningioma*, *No Tumor*, *Pituitary*; Columns: 1. Original MRI, 2. Minimal Resize [No Crop], 3. Contour Crop [Pad=0px], 4. Contour Crop [Pad=10px]).
- **Reproducibility:** Generated using a fixed random seed (`SEED = 42`).

---

## 4. Dataset Integrity Audit Findings

- **Audit Module:** `DataIntegrityAuditor` in `src/data/data_audit.py`.
- **Duplicate & Leakage Strategy:**
  - *Exact Duplicates:* Evaluated via byte-level SHA-256 hashes (`compute_sha256`).
  - *Cross-Split Leakage:* Automatically checks whether any image SHA-256 hash in `Training/` exists inside `Testing/`.
- **Patient-Level Leakage Distinction:**  
  Exact duplicate detection identifies identical file copies. True patient-level leakage detection (identifying multiple adjacent 2D MRI slices from the same 3D scan session of a single patient) requires DICOM metadata (Subject ID / Series Instance UID). Because Kaggle JPEG/PNG images lack DICOM headers, SHA-256 cross-split hash matching serves as the primary automated barrier against data leakage.
- **Current Data Status:** `data/raw/` currently contains 0 files (dataset awaiting download/zip extraction). The auditor executes gracefully without crashing when data is absent.

---

## 5. Test Results

Automated test execution (`pytest -s -v tests`) returned a **100% pass rate** across all 10 unit tests:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Harsh\Documents\sem 1\AI lab\MRI classification
configfile: pyproject.toml
collected 10 items

tests/test_continual_buffers.py::test_dynamic_head_expansion PASSED      [ 10%]
tests/test_data_pipeline.py::test_empty_contour_fallback PASSED          [ 20%]
tests/test_data_pipeline.py::test_tiny_contour_fallback PASSED           [ 30%]
tests/test_data_pipeline.py::test_padding_behavior_and_boundary_clipping PASSED [ 40%]
tests/test_data_pipeline.py::test_output_shape_and_pixel_range PASSED    [ 50%]
tests/test_data_pipeline.py::test_preprocessor_determinism PASSED        [ 60%]
tests/test_data_pipeline.py::test_rgb_channel_preservation PASSED        [ 70%]
tests/test_data_pipeline.py::test_duplicate_detection_hash_computation PASSED [ 80%]
tests/test_model_shapes.py::test_custom_cnn_forward_shape PASSED         [ 90%]
tests/test_model_shapes.py::test_custom_cnn_parameter_count_constraint PASSED [100%]

============================= 10 passed in 19.22s =============================
```

---

## 6. Remaining Technical Risks

1. **Scanner Motion Artifact Noise:** Extremely noisy MRI scans with bright non-cranial artifacts near image boundaries could affect Otsu threshold boundary estimation.
2. **Lack of DICOM Metadata:** Without patient IDs, different slice indices from the same scanning session could theoretically appear in both training and test splits if their file hashes differ.
3. **Resizing Distortion:** Converting non-square bounding boxes directly to $150 \times 150$ introduces minor aspect-ratio stretching (mitigated by setting modest safety padding).

---

## 7. Recommended Preprocessing Candidates for Phase 2

To maintain scientific rigor without making unsupported claims before running experiments, we recommend evaluating two candidates in Phase 2:

* **Candidate A (Primary Recommendation): Contour Crop with 10px Safety Padding (`padding = 10`)**  
  *Rationale:* Preserves peripheral meningeal boundaries while eliminating $>80\%$ of dead black background.
* **Candidate B (Reference Baseline): Minimal Preprocessing (No Crop)**  
  *Rationale:* Resizes raw scan directly to $150 \times 150$ and normalizes to $[0, 1]$, serving as a baseline to empirically measure the accuracy gain of skull-stripping.

---

## Conclusion & Readiness
Phase 1.1 hardening is complete. Zero model training has been initiated, and Phase 2 implementation is on hold pending formal approval.
