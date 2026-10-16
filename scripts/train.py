"""
Training entry point for LSFB-isol ISLR experiments.

Usage
-----
    python -m scripts.train \
        --config configs/i3d/hpc_train.yaml \
        --output-path /home/user/output-dir/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from islr.datasets.dataloaders import get_lsfb_dataloaders
from islr.models.utils import build_model_from_config
from islr.training import ISLRLitModule
from islr.utils import (
    GpuMemoryLogger,
    export_best_weights,
    load_config,
    resolve_precision,
    save_config,
    setup_reproducibility,
    silence_warnings_and_logs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ISLR model on LSFB-isol.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory where training outputs will be stored.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # ---- reproducibility -------------------------------------------------
    seed = config["global"].get("seed", 73)
    setup_reproducibility(seed)
    silence_warnings_and_logs()

    # ---- output directory ------------------------------------------------
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(out_dir / "config.yaml", config)

    # ---- data ------------------------------------------------------------
    train_loader, val_loader, meta = get_lsfb_dataloaders(config)

    # Save the label map next to the checkpoint for inference-time decoding.
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {int(k): v for k, v in meta.label_map.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---- model -----------------------------------------------------------
    model = build_model_from_config(config)

    lit_module = ISLRLitModule(model, config)

    pretrained = config["training"].get("load_weights_from")
    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=True)
        lit_module.load_state_dict(ckpt["state_dict"], strict=True)

    # ---- loggers ---------------------------------------------------------
    logger = TensorBoardLogger(save_dir=str(out_dir), name="tensorboard")

    # ---- callbacks -------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        filename="best-val-acc-{epoch:03d}-{val_acc_macro:.4f}",
        monitor="val_acc_macro",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="epoch"),
        GpuMemoryLogger(),
        ModelSummary(max_depth=2),
    ]

    # ---- trainer ---------------------------------------------------------
    log_every_n_steps = min(50, len(train_loader))

    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=resolve_precision(),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=config["execution"].get("accumulate_grad_batches", 1),
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
    )

    # Resume from checkpoint
    resume_from = config["training"].get("resume_from", None)

    trainer.fit(
        lit_module,
        train_loader,
        val_loader,
        ckpt_path=resume_from,
    )
    export_best_weights(checkpoint_cb)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / 1024**3
        print(f"Peak GPU memory reserved: {peak:.2f} GB")


if __name__ == "__main__":
    main()
