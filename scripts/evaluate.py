"""
Evaluation entry point for LSFB-isol ISLR experiments.

Loads a trained checkpoint and evaluates it on a specified split,
reporting micro accuracy, macro accuracy, macro F1 and complementary
metrics.

Usage
-----
    python -m scripts.evaluate \
        --run-dir /work/user/i3d/si-protocol/i3d-48f-50fps \
        --split si_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from islr.datasets.dataloaders import get_eval_lsfb_dataloader
from islr.models.utils import build_model_from_config
from islr.training import ISLRLitModule
from islr.utils import (
    load_config,
    resolve_precision,
    setup_reproducibility,
    silence_warnings_and_logs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ISLR checkpoint.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the output directory of a training run "
        "(must contain config.yaml and checkpoints/).",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Split to evaluate on (e.g. 'si_test', 'sd_test', 'test').",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best' (default), 'last', or an explicit .ckpt path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reevaluation even if the eval_<split> output directory already exists.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.yaml")

    seed = config["global"].get("seed", 73)
    setup_reproducibility(seed)
    silence_warnings_and_logs()

    # ---- resolve checkpoint path ----------------------------------------
    ckpt_dir = run_dir / "checkpoints"
    if args.checkpoint == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif args.checkpoint == "best":
        candidates = sorted(ckpt_dir.glob("best-val-acc-*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No best checkpoint found in {ckpt_dir}")
        ckpt_path = candidates[-1]  # highest epoch / val_acc in filename
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Checkpoint : {ckpt_path}")

    # ---- data ------------------------------------------------------------
    # Override both splits with the evaluation target so the dataloader
    # factory gets a consistent (train_split, val_split) pair.
    # The "train" loader is unused during evaluation but the factory
    # requires both keys -- setting them equal triggers the warning which
    # we suppress here because it is expected behaviour for evaluation.
    import warnings

    config["dataset"]["train_split"] = args.split
    config["dataset"]["val_split"] = args.split

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eval_loader, _ = get_eval_lsfb_dataloader(config)

    # ---- model -----------------------------------------------------------
    model = build_model_from_config(config)

    lit_module = ISLRLitModule(model, config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    lit_module.load_state_dict(ckpt["state_dict"], strict=True)

    # ---- logger ----------------------------------------------------------
    # Log results into the same run directory under a separate subdir
    # so test metrics appear in TensorBoard alongside training curves.
    eval_dir = run_dir / f"eval_{args.split}"
    if eval_dir.exists() and not args.force:
        print(
            "Evaluation output already exists for split '{args.split}'. "
            "Use --force to rerun evaluation."
        )
        return

    logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name=f"eval_{args.split}",
    )

    # ---- trainer ---------------------------------------------------------
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=resolve_precision(),
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.test(lit_module, eval_loader, verbose=True)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / 1024**3
        print(f"Peak GPU memory reserved: {peak:.2f} GB")


if __name__ == "__main__":
    main()
