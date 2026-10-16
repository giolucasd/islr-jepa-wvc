"""
Video transforms for the LSFB-isol pipeline.

This module provides spatial and temporal transformations for isolated sign
language recognition, supporting both I3D (baseline) and V-JEPA models.

Input/Output Contract:
- Transforms receive and return a (T, H, W, C) uint8 numpy array.
- Exceptions are the final two pipeline steps:
  * Pixel normalizers return float32 arrays (I3D: [-1, 1], V-JEPA: ImageNet z-score).
  * ChangeVideoShape reorders axes to CTHW.
"""

from __future__ import annotations

import numbers
import random
from collections.abc import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class I3DPixelsValue:
    """
    Scales uint8 [0, 255] or float [0, 1] inputs to float32 [-1, 1].
    Handles both ranges to ensure robustness regardless of dataset output.
    """

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        clip = clip.astype(np.float32)
        if clip.max() > 1.0:  # uint8 path: 0–255 -> [-1, 1]
            return clip / 127.5 - 1.0
        else:  # float path: 0–1  -> [-1, 1]
            return clip * 2.0 - 1.0


class VJEPAPixelsValue:
    """
    Normalizes THWC inputs to float32 using ImageNet mean and standard deviation.
    Applied before ChangeVideoShape to maintain the THWC convention.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        video = video.astype(np.float32) / 255.0
        return (video - self._MEAN) / self._STD


# ---------------------------------------------------------------------------
# Axis reordering
# ---------------------------------------------------------------------------


class ChangeVideoShape:
    """
    Reorders video axes from (T, H, W, C) to 'CTHW' or 'TCHW'.
    """

    _ORDERS = {  # noqa: RUF012
        "CTHW": (3, 0, 1, 2),
        "TCHW": (0, 3, 1, 2),
    }

    def __init__(self, shape: str):
        if shape not in self._ORDERS:
            raise ValueError(f"shape must be 'CTHW' or 'TCHW', got {shape!r}.")
        self._axes = self._ORDERS[shape]

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(clip, self._axes))


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------


class ResizeVideo:
    """
    Resizes every frame so the shorter side equals `size` (in pixels), preserving aspect ratio.
    """

    _INTERP = {  # noqa: RUF012
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }

    def __init__(self, size: int, interpolation: str = "linear"):
        if interpolation not in self._INTERP:
            raise ValueError(f"interpolation must be one of {list(self._INTERP)}")
        self.size = size
        self.interp = self._INTERP[interpolation]

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        T, H, W, C = clip.shape
        if H <= W:
            new_h, new_w = self.size, int(W * self.size / H)
        else:
            new_h, new_w = int(H * self.size / W), self.size
        resized = np.empty((T, new_h, new_w, C), dtype=clip.dtype)
        for i, frame in enumerate(clip):
            resized[i] = cv2.resize(frame, (new_w, new_h), interpolation=self.interp)
        return resized


class RandomCropVideo:
    """
    Samples one random (h, w) crop location and applies it consistently across all frames.
    """

    def __init__(self, size: int | tuple[int, int]):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.h, self.w = size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        _, im_h, im_w, _ = clip.shape
        if self.w > im_w or self.h > im_h:
            raise ValueError(
                f"Crop ({self.h},{self.w}) larger than frame ({im_h},{im_w})."
            )
        y1 = random.randint(0, im_h - self.h)
        x1 = random.randint(0, im_w - self.w)
        return np.ascontiguousarray(clip[:, y1 : y1 + self.h, x1 : x1 + self.w, :])


class CenterCropVideo:
    """
    Applies a center (h, w) crop consistently across all frames.
    """

    def __init__(self, size: int | tuple[int, int]):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.h, self.w = size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        _, im_h, im_w, _ = clip.shape
        if self.w > im_w or self.h > im_h:
            raise ValueError(
                f"Crop ({self.h},{self.w}) larger than frame ({im_h},{im_w})."
            )
        y1 = (im_h - self.h) // 2
        x1 = (im_w - self.w) // 2
        return np.ascontiguousarray(clip[:, y1 : y1 + self.h, x1 : x1 + self.w, :])


# ---------------------------------------------------------------------------
# Temporal transforms
# ---------------------------------------------------------------------------


class TemporalSubsample:
    """
    Approximates a target frame rate via fractional-step index sampling.

    Assumes a fixed, known source fps (LSFB-isol clips are recorded at 50 fps) --
    no per-video fps metadata is read. Indices are computed as
    round(0, step, 2*step, ...) with step = source_fps / target_fps, so
    non-integer steps are supported (e.g. target_fps=30 from source_fps=50
    yields step=1.667, effectively taking 3 out of every 5 frames).

    Assumes target_fps <= source_fps (step >= 1), so rounded indices are
    strictly non-decreasing and never collide.
    """

    def __init__(self, target_fps: float, source_fps: float = 50.0):
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if target_fps > source_fps:
            raise ValueError(
                f"target_fps ({target_fps}) must be <= source_fps ({source_fps})."
            )
        self.step = source_fps / target_fps

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        if self.step == 1.0:
            return clip
        n = len(clip)
        idx = np.round(np.arange(0, n, self.step)).astype(int)
        idx = np.clip(idx, 0, n - 1)
        return clip[idx]


class TrimVideo:
    """
    Deterministically extracts a window of frames: [offset, offset + size).
    Clips shorter than the target window are returned as-is (requires PadVideo).
    """

    def __init__(self, size: int, offset: int = 0):
        self.begin = offset
        self.end = offset + size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        return clip[self.begin : self.end]


class RandomTrimVideo:
    """
    Samples a random contiguous window of `size` frames.
    Clips shorter than `size` are returned as-is (requires PadVideo).
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        if len(clip) <= self.size:
            return clip
        start = random.randint(0, len(clip) - self.size)
        return clip[start : start + self.size]


class PadVideo:
    """
    Ensures the clip has exactly `size` frames by either looping or zero-padding.
    """

    def __init__(self, size: int, loop: bool = True):
        self.size = size
        self.loop = loop

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        if len(clip) >= self.size:
            return clip[: self.size]
        return self._loop_pad(clip) if self.loop else self._zero_pad(clip)

    def _loop_pad(self, clip: np.ndarray) -> np.ndarray:
        idx = np.arange(self.size) % len(clip)
        return clip[idx]

    def _zero_pad(self, clip: np.ndarray) -> np.ndarray:
        pad_shape = (self.size - len(clip), *clip.shape[1:])
        return np.concatenate([clip, np.zeros(pad_shape, dtype=clip.dtype)], axis=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _vjepa_short_side(crop_size: int) -> int:
    """
    Derives the resize short-side target from the crop size, matching the V-JEPA recipe:
    short_side = int(crop_size * 256 / 224)
    """
    return int(crop_size * 256 / 224)


def _make_pipeline(*steps) -> Callable[[np.ndarray], np.ndarray]:
    """Composes a sequence of callables into a single transform function."""

    def transform(video: np.ndarray) -> np.ndarray:
        for step in steps:
            video = step(video)
        return video

    return transform


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------


def get_i3d_transforms(train: bool, num_frames: int = 48) -> Callable:
    """
    Builds the I3D transform pipeline.

    Setup:
    - Pixel range: float32 [-1, 1] (I3D/Kinetics statistics)
    - Crop size: 224 x 224
    - Resize target: 270 px short side

    Args:
        train: If True, applies random trim and random crop. If False, uses center equivalents.
        num_frames: Frames to sample per clip.

    Returns:
        Callable mapping (T, H, W, C) uint8 -> (C, T, H, W) float32
    """
    if train:
        return _make_pipeline(
            RandomTrimVideo(num_frames),
            PadVideo(num_frames, loop=True),
            ResizeVideo(270, interpolation="linear"),
            RandomCropVideo((224, 224)),
            I3DPixelsValue(),
            ChangeVideoShape("CTHW"),
        )
    else:
        return _make_pipeline(
            TrimVideo(num_frames),
            PadVideo(num_frames, loop=True),
            ResizeVideo(270, interpolation="linear"),
            CenterCropVideo((224, 224)),
            I3DPixelsValue(),
            ChangeVideoShape("CTHW"),
        )


def get_vjepa_transforms(
    train: bool,
    num_frames: int = 16,
    crop_size: int = 384,
    target_fps: float | None = None,
    source_fps: float = 50.0,
) -> Callable:
    """
    Builds the V-JEPA2 / V-JEPA2.1 transform pipeline.

    Spatial operations are structurally identical to I3D. Differences lie in pixel
    statistics (ImageNet z-score) and crop/resize dimensions.

    Args:
        train: If True, applies random trim and random crop. If False, uses center equivalents.
        num_frames: Frames to sample per clip (matches model pretraining config, e.g., 16).
        crop_size: Spatial crop size (e.g., 256 for large/huge/giant, 384 for 2.1 versions).

    Returns:
        Callable mapping (T, H, W, C) uint8 -> (C, T, H, W) float32

    Note: Ensure `crop_size` is injected from the model config to the data config
    in your training script before calling the dataloaders.
    """
    short_side = _vjepa_short_side(crop_size)
    temporal_subsample = (
        [TemporalSubsample(target_fps, source_fps)] if target_fps else []
    )

    if train:
        return _make_pipeline(
            *temporal_subsample,
            RandomTrimVideo(num_frames),
            PadVideo(num_frames, loop=True),
            ResizeVideo(short_side, interpolation="linear"),
            RandomCropVideo((crop_size, crop_size)),
            VJEPAPixelsValue(),
            ChangeVideoShape("CTHW"),
        )
    else:
        return _make_pipeline(
            *temporal_subsample,
            TrimVideo(num_frames),
            PadVideo(num_frames, loop=True),
            ResizeVideo(short_side, interpolation="linear"),
            CenterCropVideo((crop_size, crop_size)),
            VJEPAPixelsValue(),
            ChangeVideoShape("CTHW"),
        )
