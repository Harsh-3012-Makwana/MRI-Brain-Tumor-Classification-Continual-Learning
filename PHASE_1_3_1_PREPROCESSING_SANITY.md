# PHASE 1.3.1 — Preprocessing & Data Pipeline Sanity Check Report

**Project Title:** Brain Tumor MRI Classification & Continual Learning  
**Phase:** 1.3.1 (Preprocessing & Data Pipeline Sanity Check)  
**Date:** October 4, 2026  
**Auditor:** Senior Computer Vision Research Engineer  
**Status:** Sanity Audit & Visual Inspection Complete  

---

## 1. Executive Summary & Objective

Before initiating baseline ResNet18 model training in Phase 2, a read-only scientific audit was conducted across the image preprocessing and data augmentation pipeline:

1. **Preprocessing Verification:** Inspected OpenCV contour-based skull stripping with 10 px safety padding and compared it against minimal direct resizing.
2. **Channel & Color Representation:** Verified 3-channel RGB image tensor generation, ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), and PyTorch memory layout (`CHW`).
3. **Augmentation Distortions Check:** Evaluated `torchvision` augmentations for potential anatomical distortion risks.
4. **Transform Determinism:** Confirmed that validation and testing data loading pipelines are 100% deterministic with zero random augmentations.

---

## 2. Technical Code & Configuration Audit

The following core modules and configuration files were inspected:

| File Path | Inspected Component | Verified Behavior | Status |
| :--- | :--- | :--- | :--- |
| `src/data/preprocessor.py` | `MRIPreprocessor` | Otsu thresholding + morphological closing/opening + 10px safety padding + boundary clipping. Fallback triggered if contour < 5% image area. | **Verified** |
| `src/data/dataset.py` | `MRIDataset` | Converts BGR $\rightarrow$ RGB, crops contour, resizes to target $(224, 224)$, applies optional training transforms, and normalizes to ImageNet stats. | **Verified** |
| `src/data/pipeline_factory.py` | `get_mri_dataloaders` | Builds reproducible PyTorch DataLoaders (Train 4,855; Val 857; Benchmark Test 1,311; Filtered Test 1,208) with generator seed 42. | **Verified** |
| `configs/model_resnet18.yaml` | Baseline Config | Specifies `IMAGENET1K_V1` weights, 4 output classes, AdamW optimizer, batch size 32, learning rate 0.0001. | **Verified** |

---

## 3. Comparative Preprocessing Analysis (Tasks 1, 2, 3)

### 3.1 Contour-Based Crop vs. Minimal Preprocessing
* **Option A — Minimal Preprocessing (Direct Resize to 224x224 RGB):**
  - *Mechanism:* BGR $\rightarrow$ RGB conversion followed immediately by `cv2.resize(img, (224, 224), INTER_AREA)`.
  - *Pros:* Retains 100% of peripheral cranial vault anatomy. No risk of accidental tumor region clipping.
  - *Cons:* Allocates significant pixel capacity to empty black background borders, reducing effective pixel resolution of internal brain tissue.
* **Option B — Contour Crop with 10px Safety Padding + Resize to 224x224 RGB (Current):**
  - *Mechanism:* Otsu binarization detects skull outer boundary, adds 10 px bounding padding, clips to image boundaries, crops, and resizes to $(224, 224)$.
  - *Pros:* Removes uninformative background black space, increasing spatial pixel density on brain parenchyma and tumor lesions.
  - *Anatomical Safety:* 10 px safety padding prevents edge clipping of superficial meningiomas or skull-adjacent lesions. Automatic fallback to uncropped image prevents failure on low-contrast scans.

---

## 4. Visual Inspection Grid & Augmentation Assessment (Tasks 4, 7, 8)

A 5-column visual comparison grid was generated across all 4 tumor classes:

* **Artifact Path:** [preprocessing_sanity_grid.png](file:///c:/Users/Harsh/Documents/sem%201/AI%20lab/MRI%20classification/artifacts/figures/preprocessing_sanity_grid.png)
* **Grid Layout ($4 \times 5$):**
  - Column 1: Raw Original MRI (Variable dimensions: 150x198 to 630x630).
  - Column 2: Minimal Preprocessing (Direct $224 \times 224$ RGB resize).
  - Column 3: Contour-Cropped Preprocessing (10 px padding + $224 \times 224$ RGB resize).
  - Column 4: Training Augmentation Sample #1 (Random horizontal flip, rotation $\pm 15^\circ$, color jitter).
  - Column 5: Training Augmentation Sample #2 (Second random variation of same sample).

### 4.1 Augmentation Safety Assessment:

| Augmentation Type | Parameter Range | Anatomical & Distortive Risk | Verdict |
| :--- | :---: | :--- | :---: |
| **Random Horizontal Flip** | $p=0.5$ | Brain anatomy is bilaterally symmetric across the sagittal plane. Preserves valid lesion structure. | **SAFE** |
| **Random Rotation** | $\pm 15^\circ$ | Simulates minor natural head tilt during scanner acquisition. $15^\circ$ limit prevents boundary truncation. | **SAFE** |
| **Color Jitter** | Brightness/Contrast $\pm 0.15$ | Simulates scanner intensity variations (e.g. gain differences, field heterogeneity) without altering tissue contrast hierarchy. | **SAFE** |
| **Random Vertical Flip** | $p=0.0$ (Not used) | Vertical flipping turns brain upside down, violating natural cranio-caudal orientation. | **UNSAFE — EXCLUDED** |

---

## 5. Normalization, Input Channels & Transform Determinism (Tasks 5, 6, 7)

1. **3-Channel RGB Input:** Grayscale MRI scans are correctly converted to 3-channel RGB (`cv2.COLOR_BGR2RGB`). Shape is `(3, 224, 224)` matching PyTorch ResNet18 `conv1` (`in_channels=3`).
2. **ImageNet Normalization:** Pixel values in $[0.0, 1.0]$ are normalized via `(x - mean) / std` using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.
3. **Transform Determinism:** `val_loader`, `benchmark_test_loader`, and `filtered_test_loader` use `transform=None`. Consecutive reads of identical samples yield bit-identical float32 tensors (`test_validation_and_test_transforms_determinism` PASSED).

---

## 6. Automated Unit Tests & Verification

Executed full repository test suite via pytest:

```bash
.venv\Scripts\python.exe -m pytest
```

* **Test Suite Result:** `20 / 20 PASSED` in 11.28 seconds.
* Zero failures or regressions.

---

## 7. Recommended Baseline Preprocessing Configuration & Next Steps

### Recommended Configuration for ResNet18 Baseline:
* **Preprocessing:** Contour crop with 10 px safety padding + resize to $224 \times 224$.
* **Normalization:** ImageNet mean/std normalization.
* **Augmentation (Train only):** Horizontal flip ($p=0.5$), rotation ($\pm 15^\circ$), brightness/contrast jitter ($\pm 0.15$).
* **Evaluation (Val/Test):** Deterministic contour crop + resize + ImageNet normalization.

### Scientific Disclaimer:
This sanity check verifies image processing correctness and anatomical safety visual plausibility. Comparative superiority of contour cropping over minimal preprocessing must be validated empirically during baseline experiments in Phase 2.

---

**Status:** Preprocessing and data pipeline are 100% verified and ready for Phase 2 baseline ResNet18 model training.
