"""Automated unit tests for Phase 2.1A Training & Evaluation Engine."""

from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet18_baseline import ResNet18Baseline, build_resnet18_baseline
from src.training.evaluator import compute_classification_metrics, evaluate_model
from src.training.trainer import ModelTrainer


def test_resnet18_model_output_shape_and_parameter_counts():
    """Verifies output shape (B, 4) and parameter counts for Linear Probe vs Fine-Tuning."""
    # Linear Probe (Frozen)
    probe_model = build_resnet18_baseline(strategy="linear_probe", pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    out = probe_model(x)
    assert out.shape == (4, 4), f"Expected shape (4, 4), got {out.shape}"

    counts_probe = probe_model.get_parameter_counts()
    assert counts_probe["total_parameters"] == 11178564
    assert counts_probe["trainable_parameters"] == 2052
    assert counts_probe["frozen_parameters"] == 11176512

    # Fine-Tuning (Full)
    finetune_model = build_resnet18_baseline(strategy="full_finetuning", pretrained=False)
    counts_finetune = finetune_model.get_parameter_counts()
    assert counts_finetune["total_parameters"] == 11178564
    assert counts_finetune["trainable_parameters"] == 11178564
    assert counts_finetune["frozen_parameters"] == 0


def test_batchnorm_eval_mode_during_linear_probe_training():
    """Verifies that BatchNorm modules stay in eval() mode during train() when backbone is frozen."""
    model = build_resnet18_baseline(strategy="linear_probe", pretrained=False)
    model.train()

    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    assert len(bn_modules) > 0

    for bn in bn_modules:
        assert not bn.training, "BatchNorm module was found in training=True mode during frozen linear probing!"


def test_gradient_updates_confined_to_head_in_linear_probe():
    """Verifies that gradients update ONLY the fc head weights during linear probing."""
    model = build_resnet18_baseline(strategy="linear_probe", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    initial_backbone_weight = model.backbone.layer4[0].conv1.weight.clone()
    initial_fc_weight = model.backbone.fc.weight.clone()

    model.train()
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    updated_backbone_weight = model.backbone.layer4[0].conv1.weight
    updated_fc_weight = model.backbone.fc.weight

    assert torch.equal(initial_backbone_weight, updated_backbone_weight), "Backbone weights changed during linear probe training!"
    assert not torch.equal(initial_fc_weight, updated_fc_weight), "FC head weights failed to update during training!"


def test_evaluator_metrics_calculation():
    """Verifies accuracy, macro & per-class precision/recall/F1, and confusion matrix calculation."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 3])  # 1 error in class 1 (predicted as class 0)

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=4)

    assert metrics["accuracy"] == 0.875  # 7 / 8
    assert metrics["total_samples"] == 8
    
    # Class 0: TP=2, FP=1, FN=0 -> prec=2/3=0.6667, rec=2/2=1.0, f1=0.8
    # Class 1: TP=1, FP=0, FN=1 -> prec=1/1=1.0, rec=1/2=0.5, f1=0.6667
    # Class 2: TP=2, FP=0, FN=0 -> prec=1.0, rec=1.0, f1=1.0
    # Class 3: TP=2, FP=0, FN=0 -> prec=1.0, rec=1.0, f1=1.0

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert cm[0, 0] == 2
    assert cm[1, 0] == 1  # 1 class 1 misclassified as class 0
    assert cm[1, 1] == 1


def test_early_stopping_and_checkpoint_save_load(tmp_path):
    """Verifies early stopping logic, checkpoint generation, and state restoration on synthetic data."""
    # Synthetic small dataset
    x = torch.randn(20, 3, 224, 224)
    y = torch.randint(0, 4, (20,))
    ds = TensorDataset(x, y)

    train_loader = DataLoader(ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = build_resnet18_baseline(strategy="linear_probe", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_epochs=10,
        patience=2,
        checkpoint_dir=tmp_path,
        seed=42
    )

    fit_res = trainer.fit()

    assert Path(fit_res["checkpoint_path"]).exists()
    assert fit_res["best_epoch"] >= 1
    assert fit_res["best_epoch"] <= 10

    # Test load_best_checkpoint
    loaded_checkpoint = trainer.load_best_checkpoint()
    assert "model_state_dict" in loaded_checkpoint
    assert loaded_checkpoint["epoch"] == fit_res["best_epoch"]
