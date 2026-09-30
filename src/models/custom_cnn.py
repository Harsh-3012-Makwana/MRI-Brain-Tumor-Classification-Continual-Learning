"""Custom Lightweight CNN Architecture (<5M parameters) for Brain Tumor MRI Scans."""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional Block consisting of Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,  # Bias is redundant when followed immediately by BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CustomLightweightCNN(nn.Module):
    """Ultra-lightweight CNN (<5M parameters) for 4-class Brain Tumor MRI Classification.

    Architectural stages:
    - Conv-Block 1: 3 -> 32 filters (3x3), BN, ReLU, MaxPool(2x2)
    - Conv-Block 2: 32 -> 64 filters (3x3), BN, ReLU, MaxPool(2x2)
    - Conv-Block 3: 64 -> 128 filters (3x3), BN, ReLU, MaxPool(2x2)
    - Conv-Block 4: 128 -> 128 filters (3x3), BN, ReLU, MaxPool(2x2)
    - Head: Global Average Pooling -> Dense(384) -> Dropout(0.4) -> Dense(num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dense_units: int = 384,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extractor Backbone
        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 128)

        # Classifier Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, dense_units)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(dense_units, num_classes)

        # Print parameter count on instantiation as required by specification
        total_params = self.count_trainable_parameters()
        print(f"[{self.__class__.__name__}] Instantiated. Trainable parameters: {total_params:,} (<5,000,000 constraint)")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps through convolutional blocks."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classification head."""
        feats = self.forward_features(x)
        pooled = self.gap(feats)
        flattened = torch.flatten(pooled, 1)
        x = self.fc1(flattened)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

    def count_trainable_parameters(self) -> int:
        """Calculates total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
