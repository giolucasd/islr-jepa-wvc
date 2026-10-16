"""
LightningModule for Isolated Sign Language Recognition (ISLR).

This module provides a model-agnostic training loop. It accepts any nn.Module
that maps (B, C, T, H, W) inputs to (B, num_classes) logits.

Key Features:
- Dataset Contract: Expects plain integer class indices for labels (B,).
- Gradient Accumulation: Delegated entirely to the PyTorch Lightning Trainer.
- Optimizers: Supports SGD (ideal for I3D) and AdamW (ideal for V-JEPA/VideoMAE).
- Scheduling: Uses linear warmup followed by cosine annealing.

Metrics logged per split:
  train : macro top-1 acc, micro top-1 acc
  val   : macro top-1 acc, micro top-1 acc, macro F1
  test  : macro top-1 acc, micro top-1 acc, macro top-5 acc,
          macro F1, macro recall, per-class recall distribution
          (mean / std / min), ECE (calibration quality)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassRecall,
)

from islr.models.base import ISLRModel


class ISLRLitModule(pl.LightningModule):
    """
    Standard PyTorch Lightning module for clip-level sign classification.

    Args:
        model: Any ISLRModel instance providing a forward pass and freeze contract.
        config: Dictionary containing configuration parameters.
    """

    def __init__(self, model: ISLRModel, config: dict) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)

        model_cfg = self.hparams["model"]
        if model_cfg.get("freeze_encoder", False):
            self.model.freeze_encoder()
        else:
            self.model.unfreeze_encoder()

        num_classes: int = self.hparams["dataset"]["num_classes"]
        self.loss_fn = nn.CrossEntropyLoss()

        # --- train metrics ---
        self.train_acc_macro = self._acc(num_classes, "macro")
        self.train_acc_micro = self._acc(num_classes, "micro")

        # --- val metrics ---
        self.val_acc_macro = self._acc(num_classes, "macro")
        self.val_acc_micro = self._acc(num_classes, "micro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # --- test metrics ---
        self.test_acc_macro = self._acc(num_classes, "macro")
        self.test_acc_micro = self._acc(num_classes, "micro")
        self.test_acc_top5 = MulticlassAccuracy(
            num_classes=num_classes, average="macro", top_k=5
        )
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        # average="none" -> (num_classes,) tensor; lets us log distribution stats
        self.test_recall_per_class = MulticlassRecall(
            num_classes=num_classes, average="none"
        )
        # ECE: 15 bins is a standard choice; norm="l1" gives the standard expected value
        self.test_ece = MulticlassCalibrationError(
            num_classes=num_classes, n_bins=15, norm="l1"
        )

    @staticmethod
    def _acc(num_classes: int, average: str) -> MulticlassAccuracy:
        return MulticlassAccuracy(num_classes=num_classes, average=average)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, targets = batch
        logits = self(x.float())
        probs = logits.softmax(dim=-1)
        loss = self.loss_fn(logits, targets)

        self.train_acc_macro(probs, targets)
        self.train_acc_micro(probs, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_acc_macro",
            self.train_acc_macro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc_micro",
            self.train_acc_micro,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, targets = batch
        logits = self(x.float())
        probs = logits.softmax(dim=-1)
        loss = self.loss_fn(logits, targets)

        self.val_acc_macro(probs, targets)
        self.val_acc_micro(probs, targets)
        self.val_f1(probs, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_acc_macro",
            self.val_acc_macro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_acc_micro",
            self.val_acc_micro,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log("val_f1", self.val_f1, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        x, targets = batch
        logits = self(x.float())
        probs = logits.softmax(dim=-1)
        loss = self.loss_fn(logits, targets)

        self.test_acc_macro(probs, targets)
        self.test_acc_micro(probs, targets)
        self.test_acc_top5(probs, targets)
        self.test_f1(probs, targets)
        self.test_recall_per_class(probs, targets)
        self.test_ece(probs, targets)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_acc_macro",
            self.test_acc_macro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_acc_micro",
            self.test_acc_micro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_acc_top5",
            self.test_acc_top5,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log("test_f1", self.test_f1, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "test_ece", self.test_ece, prog_bar=False, on_step=False, on_epoch=True
        )

    def on_test_epoch_end(self) -> None:
        """
        Aggregate and persist test-set metrics.

        Reads accumulated scalar metrics from `trainer.callback_metrics` (already
        populated by `self.log(...)` calls in `test_step`) plus the full per-class
        recall tensor from `test_recall_per_class`, and writes both to disk:
        - test_metrics.json: aggregate scalars (loss, macro/micro acc, top-5,
            f1, ece, recall macro/std/min).
        - test_recall_per_class.json: recall for every class, keyed by class
            name (resolved from label_map.json) or by index as a string fallback.

        Only rank 0 writes files under DDP to avoid race conditions across ranks.
        Per-class recall is not logged to the experiment logger; it's intended
        for offline analysis (e.g. correlating low-recall classes with
        phonological features) rather than dashboard monitoring.
        """
        recall_per_class: torch.Tensor = self.test_recall_per_class.compute()
        recall_macro = recall_per_class.mean()
        recall_std = recall_per_class.std()
        recall_min = recall_per_class.min()

        if self.trainer.is_global_zero:
            out_dir = Path(self.trainer.log_dir or ".") / "artifacts"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Aggregate scalar metrics
            summary = {
                "test_loss": self.trainer.callback_metrics["test_loss"].item(),
                "test_acc_macro": self.trainer.callback_metrics[
                    "test_acc_macro"
                ].item(),
                "test_acc_micro": self.trainer.callback_metrics[
                    "test_acc_micro"
                ].item(),
                "test_acc_top5": self.trainer.callback_metrics["test_acc_top5"].item(),
                "test_f1": self.trainer.callback_metrics["test_f1"].item(),
                "test_ece": self.trainer.callback_metrics["test_ece"].item(),
                "test_recall_macro": recall_macro.item(),
                "test_recall_std": recall_std.item(),
                "test_recall_min": recall_min.item(),
            }
            with open(out_dir / "test_metrics.json", "w") as f:
                json.dump(summary, f, indent=2)

            # Per-class recall
            recall_list = recall_per_class.cpu().tolist()
            label_map_path = out_dir.parent.parent.parent / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path) as f:
                    label_map = json.load(f)  # {"0": "sign_a", "1": "sign_b", ...}
                payload = {
                    label_map.get(str(i), str(i)): r for i, r in enumerate(recall_list)
                }
            else:
                payload = {str(i): r for i, r in enumerate(recall_list)}
            with open(out_dir / "test_recall_per_class.json", "w") as f:
                json.dump(payload, f, indent=2)

        self.test_recall_per_class.reset()

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg = self.hparams["optimizer"]
        optimizer_name = opt_cfg.get("name", "sgd").lower()

        if optimizer_name == "sgd":
            return self._configure_sgd(opt_cfg)
        if optimizer_name == "adamw":
            return self._configure_adamw(opt_cfg)

        raise ValueError(
            f"Unknown optimizer {optimizer_name!r}. Expected 'sgd' or 'adamw'."
        )

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """
        Builds a linear warmup and cosine annealing learning rate schedule.

        The learning rate grows linearly from (lr / warmup_epochs) to lr,
        then decays to near-zero over the remaining epochs. Durations are
        dynamically derived from `max_epochs` to support HPO trial scaling.
        """
        training_cfg = self.hparams["training"]
        if "max_epochs" not in training_cfg:
            raise ValueError(
                "learning rate scheduling requires max_epochs in training config."
            )

        max_epochs = training_cfg["max_epochs"]
        warmup_epochs = training_cfg.get("warmup_epochs", 5)
        cosine_epochs = max(int((max_epochs - warmup_epochs) * 1.1), 1)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_epochs,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {"scheduler": scheduler, "interval": "epoch"}

    def _configure_sgd(self, opt_cfg: dict):
        """
        Configures SGD with momentum (Recommended for convolutional video models like I3D).
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=opt_cfg.get("lr", 0.1),
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": self._configure_scheduler(optimizer),
        }

    def _configure_adamw(self, opt_cfg: dict):
        """
        Configures AdamW decoupled weight decay.

        Recommended for transformer-based video models like V-JEPA/VideoMAE.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 0.05),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": self._configure_scheduler(optimizer),
        }
