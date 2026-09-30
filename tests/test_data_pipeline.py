"""Unit tests for MRIPreprocessor: contour cropping, resizing, and normalization."""

import numpy as np
import cv2
import pytest
from src.data.preprocessor import MRIPreprocessor


def test_contour_crop_removes_black_border():
    """Generates a synthetic image with a thick black border and verifies cropping isolates the center object."""
    preprocessor = MRIPreprocessor(target_size=(150, 150))

    # Create a 200x200 black canvas
    synthetic_image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw a bright oval in the center (representing skull/brain)
    center = (100, 100)
    axes = (50, 60)
    cv2.ellipse(synthetic_image, center, axes, 0, 0, 360, (200, 200, 200), -1)

    # Apply contour crop
    cropped = preprocessor.crop_brain_contour(synthetic_image)

    # Assert cropped height and width are strictly smaller than the 200x200 canvas
    assert cropped.shape[0] < 200, f"Expected cropped height < 200, got {cropped.shape[0]}"
    assert cropped.shape[1] < 200, f"Expected cropped width < 200, got {cropped.shape[1]}"
    # Verify the brain region is bounded appropriately (~100x120)
    assert cropped.shape[0] >= 110
    assert cropped.shape[1] >= 90


def test_preprocess_output_shape_and_range():
    """Verifies complete preprocessing pipeline outputs (150, 150, 3) in range [0, 1]."""
    preprocessor = MRIPreprocessor(target_size=(150, 150))
    dummy_scan = np.random.randint(50, 255, size=(256, 256, 3), dtype=np.uint8)

    processed = preprocessor.preprocess(dummy_scan, normalize_method="zero_one")

    assert processed.shape == (150, 150, 3), f"Expected (150, 150, 3), got {processed.shape}"
    assert processed.dtype == np.float32, f"Expected float32, got {processed.dtype}"
    assert processed.min() >= 0.0, f"Min pixel value {processed.min()} < 0.0"
    assert processed.max() <= 1.0, f"Max pixel value {processed.max()} > 1.0"
