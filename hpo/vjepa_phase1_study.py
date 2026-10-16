"""
Hyperparameter search for V-JEPA 2.1 (frozen encoder) on LSFB-isol SI protocol.

Searches over the AttentivePoolingHead hyperparameters only:
    head_dropout, lr, weight_decay

The encoder is frozen throughout, so this is cheap -- each trial only trains
~2.9M parameters. A short epoch budget (15 epochs) is enough for the head
to converge enough to rank trials reliably.

The best hyperparameters found here should be used for Phase 1 (frozen) training
on the full label set before proceeding to Phase 2 (full fine-tune).

Usage
-----
    python hpo/vjepa_study.py \
        --data-root /path/to/lsfb-isol \
        --pretrained-path /path/to/vjepa2_1_vitb_dist_vitG_384.pt \
        --n-trials 10 \
        --output-dir /work/user/hpo/vjepa

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
        "batch_size": 16,
        "accumulate_grad_batches": 2,
        "num_workers": 2,
        "use_gradient_checkpointing": False,
    },
    "dataset": {
        "train_split": "si_train",
        "val_split": "si_val",
        "num_classes": 50,  # Proxy dataset subset size for quick ranking
    },
    "pipeline": {
        "max_frames": 80,
        "num_frames": 48,
        "crop_size": 384,
    },
    "model": {
        "freeze_encoder": True,
        # pretrained_path injected from CLI args
    },
    "optimizer": {
        "name": "adamw",
    },
    "training": {
        "max_epochs": 10,
        "warmup_epochs": 2,
        "freeze_encoder": True,  # Head-only optimization throughout HPO
        "resume_from": None,
    },
}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.trial.Trial,
    data_root: str,
    pretrained_path: str,
) -> float:
    setup_reproducibility(BASE_CONFIG["global"]["seed"])

    # Deep copy base context to build an independent trial instance
    cfg = copy.deepcopy(BASE_CONFIG)

    # Inject command line parameters into their structured new blocks
    cfg["dataset"]["data_root"] = data_root
    cfg["model"]["pretrained_path"] = pretrained_path

    # Searched hyperparameters
    cfg["model"]["head_dropout"] = trial.suggest_float("head_dropout", 0.0, 0.5)
    cfg["optimizer"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        1e-5,
        1e-3,
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
        log_every_n_steps=1,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    gc.collect()
    torch.cuda.empty_cache()

    return best_val_acc_cb.best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO study for V-JEPA 2.1 (frozen) on LSFB-isol SI."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--output-dir", default="/work/user/hpo/vjepa")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.output_dir, "study.db")

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    study = optuna.create_study(
        study_name="vjepa_frozen_si",
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
        lambda trial: objective(trial, args.data_root, args.pretrained_path),
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
