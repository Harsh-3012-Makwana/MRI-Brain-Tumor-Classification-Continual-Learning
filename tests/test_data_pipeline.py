"""Comprehensive unit test suite for Phase 1.1: Preprocessor padding, clipping, shape, determinism, RGB preservation, and audit functions."""

import numpy as np
import cv2
import pytest
from pathlib import Path
from src.data.preprocessor import MRIPreprocessor
from src.data.data_audit import compute_sha256, DataIntegrityAuditor


def test_empty_contour_fallback():
    """Verifies that an all-black image with no contours returns uncropped original image."""
    preprocessor = MRIPreprocessor(target_size=(150, 150))
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)

    cropped = preprocessor.crop_brain_contour(black_img)
    assert cropped.shape == black_img.shape, "Expected uncropped fallback for empty contour"


def test_tiny_contour_fallback():
    """Verifies that a tiny artifact contour below area ratio threshold returns uncropped original image."""
    preprocessor = MRIPreprocessor(target_size=(150, 150), min_contour_area_ratio=0.05)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw tiny 2x2 dot (area = 4 pixels << 0.05 * 40000 = 2000 pixels)
    cv2.rectangle(img, (10, 10), (12, 12), (255, 255, 255), -1)

    cropped = preprocessor.crop_brain_contour(img)
    assert cropped.shape == img.shape, "Expected tiny contour to trigger fallback"


def test_padding_behavior_and_boundary_clipping():
    """Verifies configurable padding (0, 5, 10, 15 px) expands crop dimensions safely without exceeding boundaries."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (200, 200, 200), -1)

    prep0 = MRIPreprocessor(padding=0)
    prep5 = MRIPreprocessor(padding=5)
    prep10 = MRIPreprocessor(padding=10)
    prep15 = MRIPreprocessor(padding=15)

    c0 = prep0.crop_brain_contour(img)
    c5 = prep5.crop_brain_contour(img)
    c10 = prep10.crop_brain_contour(img)
    c15 = prep15.crop_brain_contour(img)

    # Padding should monotonically increase height and width up to image boundaries
    assert c0.shape[0] < c5.shape[0] <= c10.shape[0] <= c15.shape[0]
    assert c0.shape[1] < c5.shape[1] <= c10.shape[1] <= c15.shape[1]

    # Test extreme padding (500 px) clips safely to original image boundaries (200x200)
    prep_extreme = MRIPreprocessor(padding=500)
    c_extreme = prep_extreme.crop_brain_contour(img)
    assert c_extreme.shape[0] <= 200
    assert c_extreme.shape[1] <= 200


def test_output_shape_and_pixel_range():
    """Verifies final preprocessed tensor is (150, 150, 3) in range [0.0, 1.0]."""
    preprocessor = MRIPreprocessor(target_size=(150, 150))
    dummy_scan = np.random.randint(0, 256, size=(220, 220, 3), dtype=np.uint8)

    processed = preprocessor.preprocess(dummy_scan, normalize_method="zero_one")

    assert processed.shape == (150, 150, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min() <= processed.max() <= 1.0


def test_preprocessor_determinism():
    """Verifies that two consecutive preprocess calls on the same image return identical outputs."""
    preprocessor = MRIPreprocessor(target_size=(150, 150), padding=10)
    img = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)

    out1 = preprocessor.preprocess(img)
    out2 = preprocessor.preprocess(img)

    assert np.array_equal(out1, out2), "Preprocessing must be 100% deterministic"


def test_rgb_channel_preservation():
    """Verifies that 3-channel RGB images preserve all 3 channels throughout cropping."""
    preprocessor = MRIPreprocessor(padding=5)
    rgb_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(rgb_img, (100, 100), 40, (100, 150, 200), -1)  # Distinct R, G, B channels

    cropped = preprocessor.crop_brain_contour(rgb_img)
    assert len(cropped.shape) == 3
    assert cropped.shape[2] == 3


def test_duplicate_detection_hash_computation(tmp_path: Path):
    """Verifies SHA-256 duplicate detection logic on temporary test files."""
    f1 = tmp_path / "img1.png"
    f2 = tmp_path / "img2.png"
    f3 = tmp_path / "img3.png"

    f1.write_bytes(b"SAME_IMAGE_DATA_123")
    f2.write_bytes(b"SAME_IMAGE_DATA_123")
    f3.write_bytes(b"DIFFERENT_IMAGE_DATA_456")

    h1 = compute_sha256(f1)
    h2 = compute_sha256(f2)
    h3 = compute_sha256(f3)

    assert h1 == h2, "Identical content must produce matching SHA-256 hash"
    assert h1 != h3, "Different content must produce distinct SHA-256 hash"
