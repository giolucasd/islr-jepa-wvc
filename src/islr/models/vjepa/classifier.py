"""
V-JEPA 2.1 Classifier for ISLR.

Wraps the vendored V-JEPA 2.1 ViT encoder and appends an attentive-pooling
classification head. Designed to load weights exclusively from local checkpoints
to support offline HPC environments.

Configuration Requirements:
- `pretrained_path`: Absolute path to the local .pt checkpoint.
- `num_frames`: Number of frames in the input video.
- `num_classes`: Injected at runtime by DatasetMeta.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from islr.models.base import ISLRModel
from islr.models.vjepa.vision_transformer import vit_base

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attentive Pooling Head
# ---------------------------------------------------------------------------


class AttentivePoolingHead(nn.Module):
    """
    Applies single-query cross-attention pooling over a sequence of patch tokens.

    A learnable query attends over the sequence of tokens, condensing the
    information into a single vector before applying LayerNorm and a linear
    classification layer.
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()

        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        B, _N, D = token_seq.shape
        q = self.q_proj(self.query.expand(B, -1, -1))  # (B, 1, D)
        k = self.k_proj(token_seq)  # (B, N, D)
        v = self.v_proj(token_seq)  # (B, N, D)

        attn = (torch.bmm(q, k.transpose(1, 2)) * (D**-0.5)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = self.out_proj(torch.bmm(attn, v).squeeze(1))  # (B, D)
        return self.classifier(self.norm(out))  # (B, num_classes)


# ---------------------------------------------------------------------------
# VJEPA Classifier
# ---------------------------------------------------------------------------


class VJEPAClassifier(ISLRModel):
    """V-JEPA 2.1 ViT-Base encoder + AttentivePoolingHead for ISLR."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        pretrained_path: str | None = cfg.get("pretrained_path", None)
        num_classes: int = cfg["num_classes"]
        num_frames: int = cfg.get("num_frames", 64)
        crop_size: int = cfg.get("crop_size", 384)
        head_dropout: float = cfg.get("head_dropout", 0.0)
        grad_ckpt: bool = cfg.get("use_gradient_checkpointing", False)

        # 1. Instantiate encoder
        self.encoder = vit_base(
            img_size=(crop_size, crop_size),
            patch_size=16,
            num_frames=num_frames,
            tubelet_size=2,
            use_rope=True,
            modality_embedding=True,
            img_temporal_dim_size=1,  # Ensures T>1 clips route to the video path
            use_activation_checkpointing=grad_ckpt,
        )

        # 2. Load offline pretrained weights
        if pretrained_path is not None:
            logger.info("Loading V-JEPA 2.1 weights from %s ...", pretrained_path)
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)

            # Clean state dict keys
            enc_state = state["encoder"]
            enc_state = {k.replace("module.", ""): v for k, v in enc_state.items()}
            enc_state = {k.replace("backbone.", ""): v for k, v in enc_state.items()}

            missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
            if missing:
                logger.warning("Missing keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys: %s", unexpected)
            logger.info("Encoder weights loaded.")
        else:
            logger.warning(
                "No pretrained_path provided -- encoder randomly initialized. "
                "Set model.pretrained_path in your YAML config."
            )

        # 3. Initialize classification head
        self.head = AttentivePoolingHead(
            embed_dim=self.encoder.embed_dim,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    # -----------------------------------------------------------------------
    # ISLRModel Contract
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_seq = self.encoder(x)  # (B, N_tokens, 768)
        return self.head(token_seq)  # (B, num_classes)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen -- training AttentivePoolingHead only.")

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Full fine-tuning -- all parameters trainable.")
