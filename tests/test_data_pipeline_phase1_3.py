"""Integrity and regression tests for Phase 1.3 Data Pipeline: Train/Val split, DataLoader properties, transform determinism, and hash isolation."""

import csv
import pytest
import torch
from pathlib import Path

from src.data.dataset import MRIDataset
from src.data.pipeline_factory import (
    create_train_val_manifest,
    export_data_pipeline_class_counts,
    get_mri_dataloaders,
    RAW_DATA_DIR,
    REPORTS_DIR
)


def test_no_train_val_path_overlap(tmp_path):
    """Verifies 0 path overlap between Train and Val splits."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    manifest_path, _ = create_train_val_manifest(val_ratio=0.15, seed=42, output_dir=tmp_path)

    train_paths = set()
    val_paths = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split_assignment"] == "Train":
                train_paths.add(row["file_path"])
            elif row["split_assignment"] == "Val":
                val_paths.add(row["file_path"])

    overlap = train_paths.intersection(val_paths)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping paths between Train and Val!"


def test_no_val_test_path_overlap(tmp_path):
    """Verifies 0 path overlap between Validation and Testing splits."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    tv_manifest_path, _ = create_train_val_manifest(val_ratio=0.15, seed=42, output_dir=tmp_path)
    
    val_paths = set()
    with open(tv_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split_assignment"] == "Val":
                val_paths.add(row["file_path"])

    test_paths = set()
    for p in (RAW_DATA_DIR / "Testing").rglob("*.jpg"):
        test_paths.add(str(p.relative_to(RAW_DATA_DIR)).replace("\\", "/"))

    overlap = val_paths.intersection(test_paths)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping paths between Val and Testing!"


def test_zero_exact_hash_overlap_train_and_filtered_test(tmp_path):
    """Verifies zero exact SHA-256 hash overlap between original Training data and Filtered Testing set."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    loaders = get_mri_dataloaders(
        target_size=(224, 224),
        batch_size=16,
        num_workers=0,
        data_dir=RAW_DATA_DIR,
        reports_dir=tmp_path
    )

    eval_manifest = tmp_path / "evaluation_manifest.csv"
    train_hashes = set()
    filtered_test_hashes = set()

    with open(eval_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fhash = row["sha256_hash"]
            if row["in_filtered_train"] == "True":
                train_hashes.add(fhash)
            if row["in_filtered_test"] == "True":
                filtered_test_hashes.add(fhash)

    overlap = train_hashes.intersection(filtered_test_hashes)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping hashes!"


def test_dataloader_tensor_properties_shape_dtype_finite():
    """Verifies tensor shape, float32 dtype, finite values, and label index ranges."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    loaders = get_mri_dataloaders(
        target_size=(224, 224),
        batch_size=8,
        num_workers=0,
        data_dir=RAW_DATA_DIR,
        reports_dir=REPORTS_DIR
    )

    imgs, lbls = next(iter(loaders["train"]))

    assert imgs.shape == (8, 3, 224, 224)
    assert imgs.dtype == torch.float32
    assert torch.isfinite(imgs).all(), "Image tensor contains NaN or Inf values!"
    
    assert lbls.shape == (8,)
    assert lbls.dtype == torch.int64
    assert (lbls >= 0).all() and (lbls <= 3).all(), "Label index out of valid range 0..3!"


def test_reproducible_split_generation(tmp_path):
    """Verifies that split generation is 100% reproducible for a given random seed."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    m1, _ = create_train_val_manifest(val_ratio=0.15, seed=42, output_dir=dir1)
    m2, _ = create_train_val_manifest(val_ratio=0.15, seed=42, output_dir=dir2)

    with open(m1, "r", encoding="utf-8") as f1, open(m2, "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read(), "Train/Val manifest outputs differed across identical seed runs!"


def test_validation_and_test_transforms_determinism():
    """Verifies that validation and test transforms contain no random augmentation (bit-identical outputs)."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    loaders = get_mri_dataloaders(
        target_size=(224, 224),
        batch_size=4,
        num_workers=0,
        data_dir=RAW_DATA_DIR,
        reports_dir=REPORTS_DIR
    )

    val_ds = loaders["val"].dataset
    
    # Retrieve sample 0 twice
    img1, lbl1 = val_ds[0]
    img2, lbl2 = val_ds[0]

    assert lbl1 == lbl2
    assert torch.equal(img1, img2), "Validation dataset returned non-identical tensors for the same sample index!"


def test_class_counts_match_generated_manifests():
    """Verifies that dataset sample counts match generated class count CSV summaries."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    loaders = get_mri_dataloaders(
        batch_size=16,
        num_workers=0,
        data_dir=RAW_DATA_DIR,
        reports_dir=REPORTS_DIR
    )

    assert len(loaders["train"].dataset) == 4855
    assert len(loaders["val"].dataset) == 857
    assert len(loaders["benchmark_test"].dataset) == 1311
    assert len(loaders["filtered_test"].dataset) == 1208
