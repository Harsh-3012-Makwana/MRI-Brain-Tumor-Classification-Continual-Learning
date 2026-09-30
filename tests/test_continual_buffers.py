"""Unit tests for continual learning dynamic head expansion and buffer operations."""

import torch
import pytest
from src.models.dynamic_head import DynamicLinearHead


def test_dynamic_head_expansion():
    """Tests that DynamicLinearHead expands from 2 -> 3 -> 4 classes preserving existing weights."""
    in_features = 384
    head = DynamicLinearHead(in_features=in_features, initial_classes=2)

    # Initial forward test
    x = torch.randn(4, in_features)
    out1 = head(x)
    assert out1.shape == (4, 2)

    # Cache old weights
    old_weight_0 = head.classifier.weight.data[0].clone()
    old_weight_1 = head.classifier.weight.data[1].clone()

    # Expand to 3 classes (Task 2)
    head.expand_classes(3)
    out2 = head(x)
    assert out2.shape == (4, 3)
    assert torch.equal(head.classifier.weight.data[0], old_weight_0)
    assert torch.equal(head.classifier.weight.data[1], old_weight_1)

    # Expand to 4 classes (Task 3)
    old_weight_2 = head.classifier.weight.data[2].clone()
    head.expand_classes(4)
    out3 = head(x)
    assert out3.shape == (4, 4)
    assert torch.equal(head.classifier.weight.data[0], old_weight_0)
    assert torch.equal(head.classifier.weight.data[1], old_weight_1)
    assert torch.equal(head.classifier.weight.data[2], old_weight_2)
