"""Automated tests for Phase 1.2D Evaluation Protocol: Manifest determinism, zero hash overlap, and counts consistency."""

import csv
import pytest
from pathlib import Path
from src.data.evaluation_protocol import generate_evaluation_protocol, RAW_DATA_DIR, REPORTS_DIR


def test_evaluation_protocol_generation_and_counts(tmp_path):
    """Verifies evaluation protocol manifest generation, total file counts, and exact exclusions."""
    if not (RAW_DATA_DIR / "Training").exists() or not (RAW_DATA_DIR / "Testing").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    manifest_path, counts_path, summary = generate_evaluation_protocol(
        data_dir=RAW_DATA_DIR,
        output_dir=tmp_path
    )

    assert manifest_path.exists()
    assert counts_path.exists()

    assert summary["total_images"] == 7023
    assert summary["raw_train_images"] == 5712
    assert summary["raw_test_images"] == 1311
    assert summary["excluded_test_images"] == 103
    assert summary["filtered_test_images"] == 1208
    assert summary["hash_overlap_between_train_and_filtered_test"] == 0


def test_zero_exact_hash_overlap_in_filtered_test(tmp_path):
    """Verifies that the filtered test set has zero SHA-256 hash overlap with the training set."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    manifest_path, _, _ = generate_evaluation_protocol(
        data_dir=RAW_DATA_DIR,
        output_dir=tmp_path
    )

    train_hashes = set()
    filtered_test_hashes = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fhash = row["sha256_hash"]
            if row["in_filtered_train"] == "True":
                train_hashes.add(fhash)
            if row["in_filtered_test"] == "True":
                filtered_test_hashes.add(fhash)

    overlap = train_hashes.intersection(filtered_test_hashes)
    assert len(overlap) == 0, f"Expected 0 overlap, found {len(overlap)} overlapping hashes: {overlap}"


def test_manifest_generation_determinism(tmp_path):
    """Verifies that manifest generation is 100% deterministic across consecutive runs."""
    if not (RAW_DATA_DIR / "Training").exists():
        pytest.skip("Extracted dataset data/raw/ not present on disk.")

    dir_run1 = tmp_path / "run1"
    dir_run2 = tmp_path / "run2"

    manifest_path1, _, _ = generate_evaluation_protocol(data_dir=RAW_DATA_DIR, output_dir=dir_run1)
    manifest_path2, _, _ = generate_evaluation_protocol(data_dir=RAW_DATA_DIR, output_dir=dir_run2)

    with open(manifest_path1, "r", encoding="utf-8") as f1, open(manifest_path2, "r", encoding="utf-8") as f2:
        content1 = f1.read()
        content2 = f2.read()

    assert content1 == content2, "Manifest outputs differed between consecutive runs!"
