"""
Abstract base class for ISLR classifiers.

Defines the standard contract for all models to ensure the training module
can seamlessly manage forward passes and encoder freezing during two-phase
fine-tuning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ISLRModel(ABC, nn.Module):
    """
    Base class for all clip-level sign-language classifiers.

    Subclasses are required to implement:
    - forward: The standard forward pass.
    - freeze_encoder: Disables backbone gradients.
    - unfreeze_encoder: Enables backbone gradients for full fine-tuning.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W) in float32.

        Returns:
            Logits of shape (B, num_classes) in float32 (no softmax applied).
        """

    @abstractmethod
    def freeze_encoder(self) -> None:
        """Freezes all backbone/encoder parameters. Head remains trainable."""

    @abstractmethod
    def unfreeze_encoder(self) -> None:
        """Unfreezes all backbone/encoder parameters for full fine-tuning."""
