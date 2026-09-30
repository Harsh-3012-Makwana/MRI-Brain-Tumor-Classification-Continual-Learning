"""Unit tests for CustomLightweightCNN architecture shapes and parameter constraints."""

import torch
import pytest
from src.models.custom_cnn import CustomLightweightCNN


def test_custom_cnn_forward_shape():
    """Verifies input tensor of shape (B, 3, 150, 150) yields output shape (B, 4)."""
    batch_size = 4
    model = CustomLightweightCNN(in_channels=3, num_classes=4)
    dummy_input = torch.randn(batch_size, 3, 150, 150)
    output = model(dummy_input)

    assert output.shape == (batch_size, 4), f"Expected shape ({batch_size}, 4), got {output.shape}"


def test_custom_cnn_parameter_count_constraint():
    """Asserts that total trainable parameters stay strictly below 5,000,000."""
    model = CustomLightweightCNN(in_channels=3, num_classes=4, dense_units=384, dropout_rate=0.4)
    total_params = model.count_trainable_parameters()

    print(f"Total trainable parameters: {total_params:,}")
    assert total_params < 5_000_000, f"Parameter count {total_params:,} exceeds 5M limit!"
    # Verify it is approximately ~292k params
    assert 200_000 < total_params < 400_000, f"Unexpected parameter range: {total_params:,}"
