"""
Hyperparameter search for V-JEPA 2.1 (full fine-tune) on LSFB-isol SI protocol.

Searches over the global fine-tuning hyperparameters:
    lr, weight_decay

Usage
-----
    python hpo/vjepa_finetune_study.py \
        --data-root /path/to/lsfb-isol \
        --pretrained-path /path/to/vjepa2_1_vitb_dist_vitG_384.pt \
        --phase1-checkpoint /path/to/phase1-50classes.ckpt \
        --n-trials 10 \
        --output-dir /work/user/hpo/vjepa-finetune

The Optuna study is persisted to SQLite so the job can be preempted and
resumed by re-running the same command.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
from pathlib import Path

import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer

from islr.datasets.dataloaders import get_lsfb_dataloaders
from islr.models.utils import build_model_from_config
from islr.training import ISLRLitModule
from islr.utils import BestValAcc, resolve_precision, setup_reproducibility

# ---------------------------------------------------------------------------
# Fixed config -- nothing here is searched
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "global": {
        "seed": 73,
        "model_name": "vjepa",
    },
    "execution": {
        "batch_size": 8,
        "accumulate_grad_batches": 4,  # Effective batch size of 32
        "num_workers": 4,
        "use_gradient_checkpointing": True,
    },
    "dataset": {
        "train_split": "si_train",
        "val_split": "si_val",
        "num_classes": 50,
    },
    "pipeline": {
        "max_frames": 80,
        "num_frames": 48,
        "crop_size": 384,
    },
    "model": {
        "freeze_encoder": False,
        "vjepa": {
            "head_dropout": 0.4,
        },
        # pretrained_path injected from CLI args
    },
    "optimizer": {
        "name": "adamw",
    },
    "training": {
        "max_epochs": 36,  # Global stop ceiling when resuming from epoch 26
        "warmup_epochs": 2,
        "resume_from": None,  # Handled via Trainer.fit(ckpt_path=...)
    },
}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.trial.Trial,
    data_root: str,
    pretrained_path: str,
    phase1_checkpoint: str,
) -> float:
    setup_reproducibility(BASE_CONFIG["global"]["seed"])

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["dataset"]["data_root"] = data_root
    cfg["model"]["pretrained_path"] = pretrained_path

    # Searched hyperparameters
    cfg["optimizer"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-5, log=True)
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        1e-2,
        1e-1,
        log=True,
    )

    # Data
    train_loader, val_loader, meta = get_lsfb_dataloaders(cfg)

    cfg["model"]["num_classes"] = meta.num_classes
    cfg["dataset"]["num_classes"] = meta.num_classes

    # Model
    model = build_model_from_config(cfg)
    lit_model = ISLRLitModule(model, cfg)

    best_val_acc_cb = BestValAcc()
    trainer = Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision=resolve_precision(),
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_acc_macro"),
            best_val_acc_cb,
        ],
        accumulate_grad_batches=cfg["execution"].get("accumulate_grad_batches", 1),
        log_every_n_steps=1,
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=phase1_checkpoint)

    gc.collect()
    torch.cuda.empty_cache()

    return best_val_acc_cb.best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO study for V-JEPA 2.1 (full fine-tune) on LSFB-isol SI."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument(
        "--phase1-checkpoint",
        required=True,
        help="Path to 50-class frozen phase1 .ckpt to warm-start from.",
    )
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--output-dir", default="/work/user/hpo/vjepa-finetune")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.output_dir, "study.db")

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    study = optuna.create_study(
        study_name="vjepa_finetune_si",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=73),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4,  # let 4 trials run fully before pruning
            n_warmup_steps=4,  # don't prune before epoch 4
        ),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial, args.data_root, args.pretrained_path, args.phase1_checkpoint
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\nFinished trials : {len(study.trials)}")
    print(f"Best val_acc    : {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 5:
        print("\nHyperparameter importances:")
        importances = optuna.importance.get_param_importances(study)
        for k, v in importances.items():
            print(f"  {k}: {v:.3f}")
    else:
        print(
            f"\nOnly {len(completed)} completed trials -- run more trials for importance estimates."
        )


if __name__ == "__main__":
    main()
