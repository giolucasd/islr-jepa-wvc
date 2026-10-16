"""
Script to analyze the duration distribution (number of frames) of LSFB-isol videos.

Useful for determining optimal preprocessing pipeline hyperparameters such as:
    max_frames and num_frames

Usage
-----
    python scripts/analyze_frames.py --data-root /path/to/lsfb-isol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lsfb_dataset.utils.datasets import load_labels, load_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze video frame statistics for LSFB-isol using padded duration estimation."
    )
    parser.add_argument(
        "--data-root", required=True, help="Root directory of the LSFB-isol dataset."
    )
    parser.add_argument(
        "--n-labels",
        type=int,
        default=700,
        help="Number of labels to filter frequent signs.",
    )
    args = parser.parse_args()

    # Fixed to 'all' for local evaluation
    target_split = "si_train"

    print(f"Loading and filtering metadata for split '{target_split}'...")

    # 1. Load the split mapping and labels using library utilities
    try:
        instances = load_split(args.data_root, target_split)
        labels, _, _ = load_labels(args.data_root, args.n_labels)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load splits/labels. Ensure split files and 'metadata' directory exist under root. Error: {e}"
        )

    # 2. Read the local instances database file
    csv_path = Path(args.data_root) / "instances.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Apply standard library filtering mechanisms
    df = df[df["id"].isin(instances)]
    df = df[df["sign"].isin(labels)]

    if len(df) == 0:
        print("Warning: No records remaining after applying split and label filters.")
        return

    # 4. Calculate raw duration delta from CSV (represented in milliseconds)
    ms_deltas = (df["end"] - df["start"]).to_numpy()

    # 5. Convert to REAL physical video frames based on your diagnostic discovery:
    # Native FPS is 50.0. The physical video contains roughly 4 additional frames of padding (~80ms total)
    NATIVE_FPS = 50.0
    PADDING_FRAMES = 4

    # Mathematical frame calculation mimicking what is stored inside the physical files
    real_video_frames = np.round((ms_deltas / 1000.0) * NATIVE_FPS) + PADDING_FRAMES
    real_video_frames = real_video_frames.astype(int)

    print(f"Total videos analyzed: {len(real_video_frames)}")
    print("-" * 50)

    # Descriptive Statistics over actual physical frames
    min_val = int(np.min(real_video_frames))
    max_val = int(np.max(real_video_frames))
    mean_val = float(np.mean(real_video_frames))
    median_val = float(np.median(real_video_frames))
    std_val = float(np.std(real_video_frames))

    print(f"Minimum : {min_val} frames")
    print(f"Maximum : {max_val} frames")
    print(f"Mean    : {mean_val:.2f} frames (± {std_val:.2f})")
    print(f"Median  : {median_val:.1f} frames")
    print("-" * 50)

    # Percentile Metrics over physical frames
    percentiles = [1, 5, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(real_video_frames, percentiles)

    print("Actual Video Frame Distribution by Percentiles:")
    for p, val in zip(percentiles, percentile_values):
        print(f"  Percentile {p:02d}% : {int(val):3d} frames")
    print("-" * 50)

    # Pipeline Budget Decisions
    print("Real Video Pipeline Coverage Analysis:")
    for frame_threshold in [16, 24, 32, 48, 64, 80]:
        coverage = float(np.mean(real_video_frames <= frame_threshold)) * 100
        print(
            f"  If max_frames={frame_threshold:<3d} -> {coverage:.2f}% of physical videos will fit WHOLE (no temporal cropping)."
        )

    # 6. Identify and save the 2 smallest videos
    print("-" * 50)
    print("Exporting the 2 smallest and largest videos...")

    # Attach the calculated frame counts back to the filtered dataframe
    df["frame_count"] = real_video_frames

    # Extract the 2 smallest videos based on frame count
    smallest_videos = df.nsmallest(2, "frame_count")[["id", "frame_count"]]
    largest_videos = df.nlargest(2, "frame_count")[["id", "frame_count"]]

    print("\nPreview of the smallest videos:")
    print(smallest_videos.to_string(index=False))
    print("\nPreview of the biggest videos:")
    print(largest_videos.to_string(index=False))


if __name__ == "__main__":
    main()
