"""Dynamic expandable classification head for class-incremental learning."""

from typing import Optional
import torch
import torch.nn as nn


class DynamicLinearHead(nn.Module):
    """Classification head that dynamically expands its output dimensions across incremental tasks.

    Preserves weights of previously learned classes when expanding from N classes to N+M classes.
    """

    def __init__(self, in_features: int, initial_classes: int = 2):
        """Initializes the dynamic classification head.

        Args:
            in_features: Number of incoming feature dimensions from penultimate layer.
            initial_classes: Initial number of output classes for Task 1.
        """
        super().__init__()
        self.in_features = in_features
        self.current_classes = initial_classes
        self.classifier = nn.Linear(in_features, initial_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear classifier."""
        return self.classifier(x)

    def expand_classes(self, new_num_classes: int, device: Optional[torch.device] = None) -> None:
        """Expands the output layer to accommodate newly introduced classes.

        Copies existing weights and biases for previously learned classes, initializing
        new class logits with standard He/Kaiming normal initialization.

        Args:
            new_num_classes: Total new number of classes (must be > current_classes).
            device: Optional torch device to place new layer on.
        """
        if new_num_classes <= self.current_classes:
            return

        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data if self.classifier.bias is not None else None

        new_classifier = nn.Linear(self.in_features, new_num_classes)
        if device is not None:
            new_classifier = new_classifier.to(device)
        else:
            new_classifier = new_classifier.to(old_weight.device)

        # Copy over old weights & biases
        with torch.no_grad():
            new_classifier.weight.data[:self.current_classes, :] = old_weight
            if old_bias is not None:
                new_classifier.bias.data[:self.current_classes] = old_bias

        self.classifier = new_classifier
        prev_classes = self.current_classes
        self.current_classes = new_num_classes
        print(f"[DynamicHead] Expanded output units from {prev_classes} -> {new_num_classes} classes.")
