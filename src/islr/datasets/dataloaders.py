"""
DataLoader factory for the LSFB-isol pipeline.

This module wires the LSFB-isol datasets to PyTorch DataLoaders.
Split logic and label mapping are delegated to `LSFBIsolConfig` from the
`lsfb-dataset` library.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from torch.utils.data import DataLoader

from islr.datasets.datasets import LsfbVideoDataset
from islr.transforms import get_i3d_transforms, get_vjepa_transforms

# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

_TRANSFORM_REGISTRY: dict[str, Callable] = {
    "i3d": get_i3d_transforms,
    "vjepa": get_vjepa_transforms,
}


def _get_transforms(
    model_name: str,
    num_frames: int,
    crop_size: int,
    target_fps: float | None = None,
) -> tuple[Callable, Callable]:
    """
    Retrieves the (train, validation) transform pipelines for the specified model.

    Args:
        model_name: Registry key selecting the normalisation pipeline ("i3d" or "vjepa").
        num_frames: Number of frames to sample per clip.
        crop_size: Spatial crop size.
        target_fps: Target FPS for the video clips.
    """
    factory = _TRANSFORM_REGISTRY.get(model_name)
    if factory is None:
        raise ValueError(
            f"No transform registered for model '{model_name}'. "
            f"Available: {list(_TRANSFORM_REGISTRY)}"
        )

    if model_name == "i3d":
        return (
            factory(train=True, num_frames=num_frames),
            factory(train=False, num_frames=num_frames),
        )
    else:
        return (
            factory(
                train=True,
                num_frames=num_frames,
                crop_size=crop_size,
                target_fps=target_fps,
            ),
            factory(
                train=False,
                num_frames=num_frames,
                crop_size=crop_size,
                target_fps=target_fps,
            ),
        )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetMeta:
    """Read-only container for dataset statistics and label mappings."""

    num_classes: int
    label_map: dict[int, str]
    train_size: int | None = None
    val_size: int | None = None
    eval_size: int | None = None


# ---------------------------------------------------------------------------
# Reproducibility & Loader Factory Helpers
# ---------------------------------------------------------------------------


def _worker_init_fn(seed: int):
    """Ensures deterministic random sampling across multiprocessing workers."""

    def fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return fn


def _build_loader(
    dataset: LsfbVideoDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    """Auxiliary builder to construct DataLoaders with consistent configuration."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn(seed),
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_lsfb_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DatasetMeta]:
    """
    Builds training and validation DataLoaders from a unified configuration dictionary.
    Uses drop_last=True to guarantee static batch dimensions during training/val.

    Splits (`train_split` and `val_split`) are read independently, supporting
    standard folds, signer-independent splits, or custom protocols from `lsfb-dataset`.

    Args:
        cfg: The entire unified configuration dictionary.

    Returns:
        A tuple containing:
        - train_loader: DataLoader for the training set.
        - val_loader: DataLoader for the validation set.
        - meta: DatasetMeta instance with class and size information.
    """
    # 1. Unpack config blocks
    global_cfg = cfg["global"]
    dataset_cfg = cfg["dataset"]
    pipeline_cfg = cfg["pipeline"]
    execution_cfg = cfg["execution"]

    # 2. Extract specific parameters
    seed = global_cfg.get("seed", 73)
    model_name = global_cfg["model_name"]

    data_root = dataset_cfg["data_root"]
    train_split = dataset_cfg.get("train_split", "train")
    val_split = dataset_cfg.get("val_split", "test")
    num_classes = dataset_cfg.get("num_classes", 700)

    max_frames = pipeline_cfg.get("max_frames", 80)
    num_frames = pipeline_cfg.get("num_frames", 16)
    crop_size = pipeline_cfg.get("crop_size", 224)
    target_fps = pipeline_cfg.get("target_fps", None)

    batch_size = execution_cfg.get("batch_size", 16)
    num_workers = execution_cfg.get("num_workers", 4)

    if train_split == val_split:
        warnings.warn(
            f"train_split and val_split are both '{train_split}'. "
            "Validation metrics are meaningless -- use only for smoke-testing "
            "or final retraining.",
            UserWarning,
            stacklevel=2,
        )

    train_config = LSFBIsolConfig(
        root=data_root,
        split=train_split,
        n_labels=num_classes,
        show_progress=True,
    )
    val_config = LSFBIsolConfig(
        root=data_root,
        split=val_split,
        n_labels=num_classes,
        show_progress=False,
    )

    train_transform, val_transform = _get_transforms(
        model_name, num_frames, crop_size, target_fps
    )

    train_ds = LsfbVideoDataset(
        train_config,
        transforms=train_transform,
        max_frames=max_frames,
    )
    val_ds = LsfbVideoDataset(
        val_config,
        transforms=val_transform,
        max_frames=max_frames,
    )

    train_loader = _build_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        seed=seed,
    )

    meta = DatasetMeta(
        num_classes=train_ds.num_classes,
        label_map=train_ds.label_map,
        train_size=len(train_ds),
        val_size=len(val_ds),
    )

    return train_loader, val_loader, meta


def get_eval_lsfb_dataloader(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DatasetMeta]:
    """
    Builds an evaluation DataLoader ensuring NO samples are dropped (`drop_last=False`).

    Reads `test_split` (falling back to "test") from the dataset configuration block.

    Args:
        cfg: The entire unified configuration dictionary.

    Returns:
        A tuple containing:
        - eval_loader: DataLoader for evaluation/testing with drop_last=False.
        - meta: DatasetMeta instance with class and size information.
    """
    global_cfg = cfg["global"]
    dataset_cfg = cfg["dataset"]
    pipeline_cfg = cfg["pipeline"]
    execution_cfg = cfg["execution"]

    seed = global_cfg.get("seed", 73)
    model_name = global_cfg["model_name"]

    data_root = dataset_cfg["data_root"]
    eval_split = dataset_cfg.get("test_split", "test")
    num_classes = dataset_cfg.get("num_classes", 700)

    max_frames = pipeline_cfg.get("max_frames", 80)
    num_frames = pipeline_cfg.get("num_frames", 16)
    crop_size = pipeline_cfg.get("crop_size", 224)
    target_fps = pipeline_cfg.get("target_fps", None)

    batch_size = execution_cfg.get("batch_size", 16)
    num_workers = execution_cfg.get("num_workers", 4)

    eval_config = LSFBIsolConfig(
        root=data_root,
        split=eval_split,
        n_labels=num_classes,
        show_progress=False,
    )

    _, eval_transform = _get_transforms(model_name, num_frames, crop_size, target_fps)

    eval_ds = LsfbVideoDataset(
        eval_config,
        transforms=eval_transform,
        max_frames=max_frames,
    )

    eval_loader = _build_loader(
        eval_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,  # Retain all remainder samples for evaluation
        seed=seed,
    )

    meta = DatasetMeta(
        num_classes=eval_ds.num_classes,
        label_map=eval_ds.label_map,
        eval_size=len(eval_ds),
    )

    return eval_loader, meta
