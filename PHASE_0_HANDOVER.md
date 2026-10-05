# PHASE 0 HANDOVER REPORT

**Project:** Multi-Class Brain Tumor Detection and Classification in MRI Scans Using Lightweight CNNs and Adaptive Incremental Learning  
**Target Audience:** Senior AI/ML Engineer taking technical ownership of the project  
**Milestone:** Phase 0 — Scaffolding, Environment Setup, Architecture Definition, & Hardware Validation  
**Date:** October 04, 2026  
**Status:** **PHASE 0 COMPLETED & VERIFIED** (0 Code Modifications made during Handover report generation)

---

## 1. Complete Repository Tree

```text
MRI classification/
├── .gitignore                        # Git exclusion rules (ignores .venv, data, checkpoints, runs)
├── .venv/                            # Isolated Python 3.11.9 virtual environment (gitignored)
├── README.md                         # Project documentation and reproduction instructions
├── pyproject.toml                    # Package metadata (PEP 517/621) and pytest configuration
├── requirements.txt                  # Pinned dependency manifest with cu128 PyTorch index
├── PHASE_0_HANDOVER.md               # Senior ML Engineer Handover Report
├── configs/                          # Experiment configuration YAML files
│   ├── base_config.yaml              # Global seed (42), CUDA device, path mappings
│   ├── data_config.yaml              # 150x150x3 RGB, contour crop params, Albumentations
│   ├── incremental_config.yaml       # 2->1->1 Task splits, buffer size, distillation temp
│   ├── model_custom.yaml             # Custom lightweight CNN hyperparameters (<5M params)
│   ├── model_resnet18.yaml           # ResNet18 transfer baseline configuration
│   └── model_vgg16.yaml              # VGG16 transfer baseline configuration
├── data/                             # Dataset storage directories (gitignored content)
│   ├── raw/                          # Directory for pristine Kaggle MRI images (empty)
│   └── processed/                    # Directory for preprocessed image caches (empty)
├── artifacts/                        # Output deliverables (gitignored content)
│   ├── checkpoints/                  # Saved .pt / .ckpt model weights (empty)
│   ├── figures/                      # Saved PNG charts and confusion matrices (empty)
│   └── logs/                         # Training metrics CSV files (empty)
├── runs/                             # TensorBoard event log directory (empty)
├── app/                              # Interactive Gradio Application package
│   └── __init__.py                   # Package initializer (stub)
├── src/                              # Main Python source package
│   ├── __init__.py                   # Core package initializer
│   ├── data/                         # Preprocessing and dataset modules
│   │   ├── __init__.py               # Subpackage initializer
│   │   └── preprocessor.py           # OpenCV contour-based skull-stripping preprocessor
│   ├── models/                       # Model architectures and heads
│   │   ├── __init__.py               # Subpackage initializer
│   │   ├── custom_cnn.py             # <5M parameter Post-BN CNN (GAP + Dropout)
│   │   └── dynamic_head.py           # Universal dynamic expanding classification head
│   ├── continual/                    # Class-incremental learning strategies
│   │   └── __init__.py               # Subpackage initializer (stub for Phase 6)
│   ├── evaluation/                   # Metrics and visualization helpers
│   │   └── __init__.py               # Subpackage initializer (stub for Phase 7)
│   ├── explainability/               # Visual localization modules
│   │   └── __init__.py               # Subpackage initializer (stub for Phase 9)
│   ├── tuning/                       # Hyperparameter optimization modules
│   │   └── __init__.py               # Subpackage initializer (stub for Phase 5)
│   └── utils/                        # Utility modules
│       ├── __init__.py               # Subpackage initializer
│       ├── config_parser.py          # Safe YAML parser and config dictionary merger
│       └── seed.py                   # Deterministic seeding for torch/np/random
└── tests/                            # Automated test suite
    ├── __init__.py                   # Package initializer
    ├── check_gpu.py                  # Live Blackwell GPU hardware validation gate
    ├── test_continual_buffers.py     # Unit test for dynamic head class expansion
    ├── test_data_pipeline.py         # Unit test for contour crop border removal & scaling
    └── test_model_shapes.py          # Unit test for forward pass shapes & <5M param limit
```

---

## 2. Project Objective and Scope

### Objective
Build an ultra-lightweight, compact Convolutional Neural Network (<5M parameters) to classify brain MRI scans into 4 classes (**Glioma**, **Meningioma**, **Pituitary Tumor**, **No Tumor**), and extend it into a **Class-Incremental Learning (CIL)** setup that introduces tumor classes sequentially while minimizing **Catastrophic Forgetting**.

### Scope
- **Dataset:** Single dataset scope using the **Masoud Nickparvar Brain Tumor MRI Dataset** (7,023 images: 5,712 train / 1,311 test).
- **Architecture Constraint:** Trainable parameter count must strictly stay under **5.0 Million**.
- **Continual Learning Setup:** Class-incremental stream introducing classes in a $2 \rightarrow 1 \rightarrow 1$ split:
  - Task 1: `{notumor, meningioma}` (2 classes)
  - Task 2: `{+glioma}` (3 classes total)
  - Task 3: `{+pituitary}` (4 classes total)
- **Compared Methods:**
  1. *Naive Fine-Tuning* (Baseline failure mode showing catastrophic forgetting)
  2. *Experience Replay* (Memory buffer of 50 exemplars/class)
  3. *Knowledge Distillation / Learning without Forgetting (LwF)* (Teacher-student logit preservation without raw image memory)
- **Reference Baselines:** Fine-tuned ResNet18, VGG16, and Joint Training (theoretical upper bound).
- **Academic Context:** Research prototype for deep learning experimentation (NOT a clinical diagnostic device).

---

## 3. Current Architecture

```
                               ┌─────────────────────────┐
                               │     Input MRI Scan      │
                               │  (150 × 150 × 3 RGB)     │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │    ConvBlock 1 (3->32)  │
                               │ Conv(3x3) -> BN -> ReLU │
                               │     -> MaxPool(2x2)     │
                               └────────────┬────────────┘
                                            │ (75 × 75 × 32)
                                            ▼
                               ┌─────────────────────────┐
                               │    ConvBlock 2 (32->64) │
                               │ Conv(3x3) -> BN -> ReLU │
                               │     -> MaxPool(2x2)     │
                               └────────────┬────────────┘
                                            │ (37 × 37 × 64)
                                            ▼
                               ┌─────────────────────────┐
                               │   ConvBlock 3 (64->128) │
                               │ Conv(3x3) -> BN -> ReLU │
                               │     -> MaxPool(2x2)     │
                               └────────────┬────────────┘
                                            │ (18 × 18 × 128)
                                            ▼
                               ┌─────────────────────────┐
                               │  ConvBlock 4 (128->128) │
                               │ Conv(3x3) -> BN -> ReLU │
                               │     -> MaxPool(2x2)     │
                               └────────────┬────────────┘
                                            │ (9 × 9 × 128)
                                            ▼
                               ┌─────────────────────────┐
                               │ Global Average Pooling  │
                               │  AdaptiveAvgPool2d(1,1) │
                               └────────────┬────────────┘
                                            │ (128-dim vector)
                                            ▼
                               ┌─────────────────────────┐
                               │  Dense Classifier Head  │
                               │   Linear(128 -> 384)    │
                               │     -> ReLU -> Drop(0.4)│
                               │ DynamicLinearHead(N)    │
                               └────────────┬────────────┘
                                            │ (N-class logits)
                                            ▼
                               ┌─────────────────────────┐
                               │     Output Logits       │
                               │ (Task 1: 2, Task 2: 3,  │
                               │  Task 3: 4 logits)      │
                               └─────────────────────────┘
```

---

## 4. All Implemented Components

1. **Environment Setup & Isolation:**
   - Standalone CPython 3.11.9 isolated in user directory (`AppData\Local\Programs\Python\Python311`).
   - Virtual environment provisioned at `.venv`.
2. **Blackwell GPU Hardware Gate Verification:**
   - PyTorch CUDA 12.8 wheel (`cu128`) configured for NVIDIA GeForce RTX 5060 Laptop GPU (compute capability `sm_120`).
   - Active GPU tensor multiplication gate verified (`x = torch.randn(10, 10).cuda() * 2`).
3. **OpenCV Contour Preprocessor (`src/data/preprocessor.py`):**
   - Implemented `MRIPreprocessor` class supporting Grayscale conversion, Gaussian blur $(5\times5)$, Otsu thresholding, morphological closing/opening, contour detection, largest contour bounding-box cropping, resizing to $150\times150$, and $[0.0, 1.0]$ normalization.
4. **Custom Lightweight CNN Architecture (`src/models/custom_cnn.py`):**
   - Implemented `CustomLightweightCNN` with 4 Conv-PostBN-ReLU-MaxPool stages and a Global Average Pooling head. Total trainable parameters: **292,612** ($\approx 0.29\text{M}$ params).
5. **Universal Dynamic Classification Head (`src/models/dynamic_head.py`):**
   - Implemented `DynamicLinearHead` with `expand_classes()` method. Dynamically expands output logits from $N \rightarrow N+M$ while preserving existing class weights and biases.
6. **Configuration System (`configs/` & `src/utils/config_parser.py`):**
   - Structured YAML configs (`base_config`, `data_config`, `model_custom`, `model_resnet18`, `model_vgg16`, `incremental_config`).
   - Implemented `load_yaml` and recursive `merge_configs` functions.
7. **Determinism Engine (`src/utils/seed.py`):**
   - Implemented `seed_everything(seed=42)` locking Python `random`, `numpy`, `torch.manual_seed`, and cuDNN deterministic flags.
8. **Automated Unit Test Suite (`tests/`):**
   - 5 automated pytest tests covering forward shapes, parameter count constraint (<5M), contour cropping border removal, and dynamic head weight retention.

---

## 5. All Created Files and Their Responsibilities

| File Path | Implemented vs Planned | Exact Responsibility |
| :--- | :--- | :--- |
| `.gitignore` | **Implemented** | Defines git exclusions (`.venv`, `data/`, `artifacts/checkpoints/`, `runs/`). |
| `README.md` | **Implemented** | Research overview, architecture breakdown, hardware specs, reproduction guide. |
| `pyproject.toml` | **Implemented** | Package metadata (PEP 517/621) and pytest settings. |
| `requirements.txt` | **Implemented** | Manifest with exact version pins and PyTorch `cu128` index URL. |
| `PHASE_0_HANDOVER.md` | **Implemented** | Technical handover document for Senior ML Engineer taking project ownership. |
| `configs/base_config.yaml` | **Implemented** | Global seeds (42), CUDA device mapping, worker counts, path mappings. |
| `configs/data_config.yaml` | **Implemented** | Image size ($150\times150\times3$), contour crop parameters, Albumentations limits. |
| `configs/model_custom.yaml` | **Implemented** | Custom CNN architecture specifications and AdamW training hyperparameters. |
| `configs/model_resnet18.yaml` | **Implemented** | ResNet18 ImageNet pre-trained baseline settings. |
| `configs/model_vgg16.yaml` | **Implemented** | VGG16 ImageNet pre-trained baseline settings. |
| `configs/incremental_config.yaml` | **Implemented** | Task splits ($2\rightarrow1\rightarrow1$), replay buffer size (50/class), distillation $T=2.0$. |
| `src/__init__.py` | **Implemented** | Package initializer for core `src` package. |
| `src/data/__init__.py` | **Implemented** | Subpackage initializer for data modules. |
| `src/data/preprocessor.py` | **Implemented** | `MRIPreprocessor` OpenCV skull stripping, cropping, resizing, normalization. |
| `src/models/__init__.py` | **Implemented** | Subpackage initializer for neural network models. |
| `src/models/custom_cnn.py` | **Implemented** | `CustomLightweightCNN` architecture definition (<5M params). |
| `src/models/dynamic_head.py` | **Implemented** | `DynamicLinearHead` expandable output layer for CIL. |
| `src/continual/__init__.py` | **Implemented (Stub)** | Package stub reserved for Phase 6 incremental learning strategies. |
| `src/evaluation/__init__.py` | **Implemented (Stub)** | Package stub reserved for Phase 7 metrics & comparison matrix. |
| `src/explainability/__init__.py` | **Implemented (Stub)** | Package stub reserved for Phase 9 Grad-CAM module. |
| `src/tuning/__init__.py` | **Implemented (Stub)** | Package stub reserved for Phase 5 Optuna hyperparameter tuning. |
| `src/utils/__init__.py` | **Implemented** | Subpackage initializer for utilities. |
| `src/utils/config_parser.py` | **Implemented** | YAML file reader (`load_yaml`) and recursive dictionary merger (`merge_configs`). |
| `src/utils/seed.py` | **Implemented** | Deterministic seed lock function (`seed_everything`). |
| `app/__init__.py` | **Implemented (Stub)** | Package stub reserved for Phase 8 Gradio Web Application. |
| `tests/__init__.py` | **Implemented** | Test package initializer. |
| `tests/check_gpu.py` | **Implemented** | Hardware gate script for RTX 5060 `sm_120` CUDA 12.8 tensor execution test. |
| `tests/test_model_shapes.py` | **Implemented** | Pytest unit test for Custom CNN forward pass shape & <5M parameter assert. |
| `tests/test_data_pipeline.py` | **Implemented** | Pytest unit test for OpenCV contour crop border removal & $[0, 1]$ scaling. |
| `tests/test_continual_buffers.py`| **Implemented** | Pytest unit test for dynamic classification head expansion & weight locking. |
| `src/data/dataset.py` | **PLANNED (Phase 2)** | PyTorch `Dataset` with class filtering and Albumentations integration. |
| `src/data/datamodule.py` | **PLANNED (Phase 2)** | PyTorch Lightning `DataModule` for train/val/test data splits. |
| `src/models/baselines.py` | **PLANNED (Phase 4)** | Wrappers for pretrained ResNet18 and VGG16 models. |
| `src/continual/base_strategy.py`| **PLANNED (Phase 6)** | Abstract base class for incremental learning strategies. |
| `src/continual/naive_strategy.py`| **PLANNED (Phase 6a)**| Naive fine-tuning training strategy (forgetting baseline). |
| `src/continual/replay_strategy.py`| **PLANNED (Phase 6b)**| Experience Replay strategy with exemplar memory buffer. |
| `src/continual/distillation_strategy.py`| **PLANNED (Phase 6c)**| Knowledge Distillation (LwF) teacher-student training strategy. |
| `src/evaluation/metrics.py` | **PLANNED (Phase 7)** | Accuracy, F1, Precision, Recall, Latency, Parameter count functions. |
| `src/evaluation/continual_metrics.py`| **PLANNED (Phase 7/9)**| Task Matrix $R_{i,j}$, Average Accuracy $A_T$, Forgetting $F_T$, BWT. |
| `src/evaluation/visualizer.py` | **PLANNED (Phase 7)** | Confusion matrix and accuracy plot generators. |
| `src/explainability/gradcam.py` | **PLANNED (Phase 9)** | Grad-CAM heatmap generator hooked to ConvBlock 4. |
| `src/tuning/optuna_tuner.py` | **PLANNED (Phase 5)** | Optuna hyperparameter optimization study script. |
| `app/inference.py` | **PLANNED (Phase 8)** | Inference engine wrapping preprocessor, model weights, and Grad-CAM. |
| `app/ui.py` | **PLANNED (Phase 8)** | Gradio web UI layout and event handlers. |
| `app/main.py` | **PLANNED (Phase 8)** | Entry point for launching the Gradio web app. |

---

## 6. Dependencies and Versions

All dependencies are pinned in `requirements.txt` and verified in `.venv`:

```text
--extra-index-url https://download.pytorch.org/whl/nightly/cu128
torch==2.12.0.dev20260408+cu128
torchvision==0.27.0.dev20260407+cu128
pytorch-lightning==2.6.6
opencv-python-headless==5.0.0.93
albumentations==1.3.1
scikit-image==0.26.0
scikit-learn==1.9.1
scipy==1.17.1
numpy==2.4.6
optuna==5.0.0
matplotlib==3.11.2
seaborn==0.13.2
gradio==6.28.0
pyyaml==6.0.3
pydantic==2.13.5
tensorboard==2.21.0
kaggle==2.2.4
pytest==9.1.1
```

> **Dependency Rationale:**
> 1. `torch` and `torchvision` are pulled from `cu128` (CUDA 12.8) to support NVIDIA Blackwell `sm_120`.
> 2. `albumentations` is pinned to `1.3.1` to avoid `albucore`/`stringzilla` C++ compilation requirements on Windows systems lacking Visual Studio MSVC tools.

---

## 7. Dataset Location and Current Status

- **Dataset Identifier:** `masoudnickparvar/brain-tumor-mri-dataset`
- **Total Scan Count:** 7,023 images across 4 subdirectories: `glioma/`, `meningioma/`, `notumor/`, `pituitary/`.
- **Target Ingestion Path:** `data/raw/`
- **Current Ingestion Status:** **NOT YET INGESTED**. 
  - The local directory `data/raw/` exists in the filesystem but contains 0 image files.
  - Ingestion via Kaggle API CLI (`kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset`) or manual zip extraction is scheduled for **Phase 1**.

---

## 8. Model Architecture Decisions

1. **Global Average Pooling (GAP) Over Flattening:**
   - At the output of ConvBlock 4, feature maps have dimensions $9 \times 9 \times 128 = 10,368$ values.
   - *Flattening* to Dense(384) would require $10,368 \times 384 = \mathbf{3,981,312}$ parameters in the dense layer alone.
   - *Global Average Pooling* reduces spatial dimensions to $1 \times 1 \times 128$. Connecting 128 features to Dense(384) requires only $128 \times 384 = \mathbf{49,152}$ parameters.
   - Total model parameters: **292,612** ($\approx 0.29\text{M}$ params), comfortably satisfying the $<5\text{M}$ parameter constraint.
2. **Post-Batch Normalization & Convolutional Bias:**
   - Batch Normalization (`nn.BatchNorm2d`) is applied immediately after `nn.Conv2d` and prior to `nn.ReLU`.
   - `nn.Conv2d` uses `bias=False` because BatchNorm's subtraction of the mean renders convolutional bias mathematically redundant.
3. **Dropout Regularization:**
   - `Dropout(p=0.4)` is inserted before the final output layer to prevent co-adaptation of features on small medical MRI split datasets.
4. **Dynamic Expanding Head (`DynamicLinearHead`):**
   - Output layer expands dynamically from 2 to 3 to 4 logits when new tasks arrive in class-incremental training.
   - Previous class weights and biases are copied over without modification (`torch.no_grad()`), while new class logits are initialized with standard weights.

---

## 9. Current Configuration

All configuration files live in `configs/`:

- **`configs/base_config.yaml`:**
  - `seed`: 42
  - `device`: `"cuda:0"`
  - `deterministic`: `true`
  - `paths`: `data/raw`, `data/processed`, `artifacts/checkpoints`, `artifacts/logs`, `artifacts/figures`, `runs`
- **`configs/data_config.yaml`:**
  - `image_size`: `[150, 150]`
  - `in_channels`: 3
  - `classes`: `["glioma", "meningioma", "notumor", "pituitary"]`
  - `contour_crop`: `enabled: true`, `blur_kernel: [5, 5]`, `threshold_method: "otsu"`
  - `augmentations`: Rotation limit $\pm 15^\circ$, horizontal flip $p=0.5$, brightness/contrast jitter $0.15$.
- **`configs/model_custom.yaml`:**
  - Backbone: 4 ConvBlocks ($32, 64, 128, 128$ filters)
  - Head: GAP $\rightarrow$ Dense(384) $\rightarrow$ Dropout(0.4) $\rightarrow$ Dense(num_classes)
  - Defaults: `epochs: 25`, `batch_size: 32`, `learning_rate: 0.0005`, `optimizer: "adamw"`.
- **`configs/incremental_config.yaml`:**
  - Task Split: $2 \rightarrow 1 \rightarrow 1$ (Task 1: `{notumor, meningioma}`, Task 2: `{+glioma}`, Task 3: `{+pituitary}`).
  - `naive`: 15 epochs/task, $LR = 0.0003$.
  - `replay`: `buffer_size_per_class: 50`, `replay_batch_ratio: 0.3`.
  - `distillation`: `temperature: 2.0`, `alpha_kd: 0.5`, `alpha_ce: 0.5`.

---

## 10. Completed Tests and Their Actual Results

Executing `.\.venv\Scripts\pytest.exe tests` produces the following verified log output:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Harsh\Documents\sem 1\AI lab\MRI classification\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Harsh\Documents\sem 1\AI lab\MRI classification
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 5 items

tests/test_continual_buffers.py::test_dynamic_head_expansion PASSED      [ 20%]
tests/test_data_pipeline.py::test_contour_crop_removes_black_border PASSED [ 40%]
tests/test_data_pipeline.py::test_preprocess_output_shape_and_range PASSED [ 60%]
tests/test_model_shapes.py::test_custom_cnn_forward_shape PASSED         [ 80%]
tests/test_model_shapes.py::test_custom_cnn_parameter_count_constraint PASSED [100%]

============================= 5 passed in 23.63s ==============================
```

Executing `.\.venv\Scripts\python.exe tests/check_gpu.py` produces:

```text
PyTorch Version: 2.12.0.dev20260408+cu128
CUDA Available: True
Device Name: NVIDIA GeForce RTX 5060 Laptop GPU
Device Capability: (12, 0)
Tensor Op Result: -7.5759
GPU Hardware Gate PASSED successfully!
```

---

## 11. Known Bugs and Incomplete Components

### Known Bugs
- **None**. All 5 automated unit tests and the GPU hardware gate script execute with zero errors.

### Incomplete / Planned Components
1. `src/data/dataset.py` & `datamodule.py` (Phase 2) — PyTorch Dataset and Lightning DataModule for task filtering.
2. `src/models/baselines.py` (Phase 4) — Pretrained ResNet18 and VGG16 fine-tuning wrappers.
3. `src/tuning/optuna_tuner.py` (Phase 5) — Optuna hyperparameter search script.
4. `src/continual/` strategy suite (Phase 6) — Naive, Experience Replay, and Knowledge Distillation implementations.
5. `src/evaluation/` harness (Phase 7) — Evaluation script generating the final CSV/LaTeX comparison table.
6. `app/` (Phase 8) — Gradio interactive web interface.
7. `src/explainability/gradcam.py` & `continual_metrics.py` (Phase 9) — Grad-CAM heatmaps and formal continual learning metrics ($A_T, F_T, \text{BWT}$).

---

## 12. Technical Decisions Already Finalized

1. **Environment:** Isolated Python 3.11.9 virtual environment (`.venv`) to bypass Python 3.14 wheel incompatibilities.
2. **GPU & CUDA Build:** PyTorch `cu128` (CUDA 12.8) distribution index to ensure native `sm_120` kernel support for the RTX 5060 Blackwell GPU.
3. **Augmentations Library:** Pinned to `albumentations==1.3.1` to avoid MSVC C++ compilation dependencies.
4. **Logging System:** Local TensorBoard (`runs/`) and structured CSV logs (`artifacts/logs/`).
5. **Continual Stream:** $2 \rightarrow 1 \rightarrow 1$ task split (Task 1: 2 classes, Task 2: +1, Task 3: +1).
6. **Classification Head Strategy:** Universal dynamic head expansion via `DynamicLinearHead`.

---

## 13. Decisions That Are Still Open

1. **Replay Exemplar Selection Policy:** Simple random sampling from past task training splits vs. Herding (Mean Feature Matching) exemplar selection.
2. **Distillation Temperature Tuning:** Fixed $T=2.0$ vs. Optuna-searched temperature range $[1.5, 5.0]$.
3. **Optuna Search Trial Budget:** Recommended 20 trials vs. extended 50 trials for Phase 5.

---

## 14. Current Project Execution Instructions

### A. Environment Activation
Open PowerShell in the repository root (`c:\Users\Harsh\Documents\sem 1\AI lab\MRI classification`):

```powershell
.\.venv\Scripts\Activate.ps1
```

### B. Hardware Verification Gate
Run the GPU validation script to confirm RTX 5060 Blackwell CUDA tensor execution:

```powershell
python tests/check_gpu.py
```

### C. Run Unit Test Suite
Execute pytest across all test modules:

```powershell
pytest tests
```

---

## 15. Recommended Next Engineering Steps

1. **Phase 1 (Data Ingestion & OpenCV Skull-Stripping Validation):**
   - Download the Kaggle dataset (`masoudnickparvar/brain-tumor-mri-dataset`) into `data/raw/`.
   - Run `MRIPreprocessor` across sample scans from each class.
   - Save a visual before/after figure (`artifacts/figures/preprocessing_inspection.png`) comparing original MRI scans against contour-cropped brain tissue.
2. **Phase 2 (PyTorch DataModule & Task Splitter):**
   - Implement `MRIDataset` in `src/data/dataset.py` with dynamic task class filtering.
   - Implement `MRIDataModule` in `src/data/datamodule.py` managing stratified train/validation splits and test data loaders.
