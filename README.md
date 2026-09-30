# Multi-Class Brain Tumor Detection & Incremental Learning in MRI Scans

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA 12.8](https://img.shields.io/badge/PyTorch-CUDA%2012.8%20(cu128)-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic Notice:** This software is an experimental research prototype for deep learning evaluation and catastrophic forgetting mitigation. Model predictions are strictly intended for research and educational purposes and are **NOT** intended to replace certified radiological or clinical diagnosis.

---

## 1. Project Overview

This repository implements an academic deep learning framework for 4-class intracranial tumor classification in Brain MRI scans:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor (Normal)**

Using the curated **Masoud Nickparvar Brain Tumor MRI Dataset** (7,023 images from Figshare, SARTAJ, and Br35H), the project addresses:
- **Parameter Efficiency:** A custom convolutional architecture maintaining fewer than 5M parameters ($\approx 0.29\text{M}$ params) with Global Average Pooling (GAP) and Post-Batch Normalization.
- **Class-Incremental Learning (CIL):** A testbed simulating real-world medical data streaming where new tumor categories arrive sequentially ($2 \rightarrow 1 \rightarrow 1$ task stream), mitigating **Catastrophic Forgetting** via:
  - **Naive Fine-Tuning** (Lower-bound baseline)
  - **Experience Replay** (Exemplar memory buffer)
  - **Knowledge Distillation / LwF** (Teacher-student logit preservation)
- **Transfer Learning Baselines:** Fine-tuned **ResNet18** and **VGG16** for parameter-vs-accuracy trade-off analysis.
- **Explainability & Metrics:** Grad-CAM visual heatmaps and formal continual learning metrics ($A_T, F_T, \text{BWT}$).
- **Interactive UI:** A lightweight Gradio web interface for localized inference.

---

## 2. Repository Structure

```text
MRI classification/
├── .venv/                            # Isolated Python 3.11 virtual environment
├── configs/                          # Experiment configuration YAMLs
│   ├── base_config.yaml              # Global seed, paths, CUDA device
│   ├── data_config.yaml              # 150x150x3 RGB, contour crop params, Albumentations
│   ├── model_custom.yaml             # Custom lightweight CNN (<5M parameters)
│   ├── model_resnet18.yaml           # ResNet18 transfer baseline
│   ├── model_vgg16.yaml              # VGG16 transfer baseline
│   └── incremental_config.yaml       # 2->1->1 Task splits, buffer size, distillation temp
├── data/
│   ├── raw/                          # Original Nickparvar dataset (gitignored)
│   └── processed/                    # Preprocessed and skull-stripped caches
├── src/
│   ├── data/                         # Preprocessing, skull-stripping, datasets, DataModules
│   ├── models/                       # Custom CNN, baselines, universal dynamic head
│   ├── continual/                    # Base, Naive, Replay, and Distillation strategies
│   ├── evaluation/                   # Metrics, continual matrices, and plotting
│   ├── explainability/               # Grad-CAM heatmap generation
│   ├── tuning/                       # Optuna hyperparameter optimization
│   └── utils/                        # Deterministic seeding, config parser
├── app/                              # Gradio web application
├── tests/                            # Automated unit test suite
├── artifacts/                        # Checkpoints, figures, benchmark CSVs
├── runs/                             # TensorBoard event logs
├── requirements.txt                  # Pinned dependency specifications
└── README.md                         # Project documentation
```

---

## 3. Installation & Hardware Environment

### Workstation Specifications
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6, Blackwell Architecture, compute capability `sm_120`)
- **CUDA:** 12.8 / Driver 610.88
- **Python:** 3.11.9 (isolated in `.venv`)

### Quick Setup
```powershell
# 1. Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies with CUDA 12.8 support
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision
pip install -r requirements.txt

# 3. Verify Blackwell GPU Hardware Gate
python tests/check_gpu.py
```

---

## 4. Benchmark Comparison Matrix

| Model / Method | Accuracy | Macro F1 | Trainable Params | Latency (ms) | Forgetting ($F$) | Backward Transfer (BWT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom Lightweight CNN** | TBD | TBD | ~0.29M | TBD | N/A | N/A |
| **ResNet18 (Transfer)** | TBD | TBD | 11.2M | TBD | N/A | N/A |
| **VGG16 (Transfer)** | TBD | TBD | 134.3M | TBD | N/A | N/A |
| **Joint Training (Upper Bound)** | TBD | TBD | ~0.29M | TBD | N/A | N/A |
| **Naive Fine-Tuning** | TBD | TBD | ~0.29M | TBD | TBD | TBD |
| **CNN + Experience Replay** | TBD | TBD | ~0.29M | TBD | TBD | TBD |
| **CNN + Knowledge Distillation (LwF)** | TBD | TBD | ~0.29M | TBD | TBD | TBD |
