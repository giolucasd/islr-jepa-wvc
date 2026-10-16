"""
I3D Classifier for Isolated Sign Language Recognition (ISLR).

Wraps the standard InceptionI3d backbone with an ISLR-compatible head.
It collapses the temporal dimension via max-pooling to output plain
(B, num_classes) logits, making it compatible with standard CrossEntropyLoss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from islr.models.base import ISLRModel
from islr.models.i3d.inception import InceptionI3d


class I3DClassifier(ISLRModel):
    """
    InceptionI3d backbone for clip-level sign classification.

    Args:
        cfg: Dictionary containing model configuration.
             Required: "num_classes" (int).
             Optional: "pretrained_path" (str), "in_channels" (int, default 3),
             "head_dropout" (float, default 0.5), "use_gradient_checkpointing".
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        num_classes: int = cfg["num_classes"]
        in_channels: int = cfg.get("in_channels", 3)
        head_dropout: float = cfg.get("head_dropout", 0.5)
        use_gradient_ckpt: bool = cfg.get("use_gradient_checkpointing", False)
        pretrained_path: str | None = cfg.get("pretrained_path", None)

        if pretrained_path is not None:
            # Load the checkpoint first and infer its output size from the
            # logits weight shape -- works for any I3D weights file without
            # hardcoding an assumption about how many classes it has.
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            pretrained_num_classes: int = state["logits.conv3d.weight"].shape[0]

            self.backbone = InceptionI3d(
                num_classes=pretrained_num_classes,
                in_channels=in_channels,
                dropout_keep_prob=head_dropout,
                use_gradient_checkpointing=use_gradient_ckpt,
            )
            self.backbone.load_state_dict(state)
            self.backbone.replace_logits(num_classes)
        else:
            # Random init or resuming from a full Lightning checkpoint.
            self.backbone = InceptionI3d(
                num_classes=num_classes,
                in_channels=in_channels,
                dropout_keep_prob=head_dropout,
                use_gradient_checkpointing=use_gradient_ckpt,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W).

        Returns:
            Logits of shape (B, num_classes). Per-frame logits are interpolated
            back to the original temporal resolution and max-pooled over time.
        """
        t = x.size(2)

        # backbone returns (B, num_classes, T') where T' << T
        per_frame_logits: torch.Tensor = self.backbone(x)

        # upsample back to original temporal resolution
        per_frame_logits = F.interpolate(
            per_frame_logits, size=t, mode="linear", align_corners=False
        )  # (B, num_classes, T)

        # max-pool over time -> clip-level logits
        logits, _ = per_frame_logits.max(dim=2)  # (B, num_classes)

        return logits

    def freeze_encoder(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.logits.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
