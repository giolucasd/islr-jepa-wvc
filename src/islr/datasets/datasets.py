"""
PyTorch Dataset for LSFB-isol video clips.

This module is strictly responsible for video I/O and spatial/temporal transforms.
All metadata logic (split routing, label mapping, instance filtering) is
delegated to LSFBIsolBase from the official lsfb-dataset library.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import cv2
import numpy as np
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase
from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from torch.utils.data import Dataset


class LsfbVideoDataset(Dataset):
    """
    Loads video instances and applies requested transforms.

    Args:
        config: LSFBIsolConfig defining the split, dataset root, and labels.
        transforms: A callable pipeline applied to the raw (T, H, W, C) uint8 array.
        max_frames: Hard limit on frames read per video to prevent memory
                    exhaustion on unusually long clips prior to trimming.
    """

    def __init__(
        self,
        config: LSFBIsolConfig,
        transforms: Callable | None = None,
        max_frames: int = 80,
    ):
        self._base = LSFBIsolBase(config)
        self.transforms = transforms
        self.max_frames = max_frames

        # Expose for DatasetMeta and model config injection
        self.label_map: dict[int, str] = self._base.index_to_label
        self.num_classes: int = len(self._base.labels)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        """
        Retrieves a video and its label, with a fallback retry mechanism
        for corrupted video files.

        Returns:
            A tuple of (transformed_video_tensor, label_int).
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                instance_id = self._base.instances[idx]
                label: int = self._base.targets[instance_id]
                path = str(self._base.config.root) + f"/videos/{instance_id}.mp4"
                video = self._read_video(path)
                if self.transforms is not None:
                    video = self.transforms(video)
                return video, label
            except (RuntimeError, Exception) as e:  # noqa: BLE001
                warnings.warn(
                    f"Failed to read video at idx {idx} (attempt {attempt + 1}/{max_retries}): {e}",
                    UserWarning,
                    stacklevel=2,
                )
                # Try next sample in the dataset, wrapping around
                idx = (idx + 1) % len(self)
        raise RuntimeError(
            f"Failed to load {max_retries} consecutive samples -- dataset may be corrupted."
        )

    # ------------------------------------------------------------------

    def _read_video(self, path: str) -> np.ndarray:
        """
        Decodes a video from disk into a raw uint8 RGB array of shape (T, H, W, C).
        Raises FileNotFoundError or RuntimeError on I/O failures.

        NOTE: Dont try optimizing it.
        IO is already irrelevant compared to the model training time.
        The bottleneck is the model itself.
        """
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        frames = []

        try:
            while len(frames) < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame[:, :, ::-1].copy())  # BGR -> RGB
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from: {path}")

        return np.stack(frames).astype(np.uint8)  # (T, H, W, C)
