"""ResNet18 Baseline Model Builder supporting Frozen Linear Probe and Full Fine-Tuning strategies."""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ResNet18Baseline(nn.Module):
    """ResNet18 Transfer Learning Model for 4-class Brain Tumor MRI Classification."""

    def __init__(
        self,
        num_classes: int = 4,
        freeze_backbone: bool = False,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of output classification targets (default: 4).
            freeze_backbone: If True, freezes backbone layers for Linear Probe.
            pretrained: If True, loads IMAGENET1K_V1 pretrained weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)
        
        # Replace 1000-class ImageNet head with 4-class Linear Head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Apply freezing strategy
        self.set_freeze_strategy(freeze_backbone)

    def set_freeze_strategy(self, freeze_backbone: bool):
        """Configures parameter gradient requirements for Linear Probe or Fine-Tuning."""
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            # Freeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze new FC classification head
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            # Unfreeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        """
        Overrides nn.Module.train() to ensure BatchNorm layers remain in eval mode
        when backbone is frozen.
        """
        super().train(mode)
        if mode and self.freeze_backbone:
            # Set all BatchNorm layers to eval mode during linear probing
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet18 backbone and classification head."""
        return self.backbone(x)

    def get_parameter_counts(self) -> Dict[str, int]:
        """Returns total, trainable, and frozen parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params
        }


def build_resnet18_baseline(
    strategy: str = "linear_probe",
    num_classes: int = 4,
    pretrained: bool = True
) -> ResNet18Baseline:
    """
    Factory function for building ResNet18Baseline.
    
    Args:
        strategy: 'linear_probe' (frozen backbone) or 'full_finetuning' (unfrozen).
        num_classes: Number of output targets (default 4).
        pretrained: Whether to load ImageNet weights.
    """
    if strategy not in ["linear_probe", "full_finetuning"]:
        raise ValueError(f"Unknown strategy: '{strategy}'. Expected 'linear_probe' or 'full_finetuning'.")

    freeze_backbone = (strategy == "linear_probe")
    return ResNet18Baseline(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained
    )
