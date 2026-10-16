"""
Diagnostic script to verify if 'start' and 'end' from instances.csv are frames or milliseconds.
Compares CSV metadata values against real physical video files using OpenCV.

This script uses the 'mini_sample' split to allow quick local verification.

Usage
-----
    python scripts/verify_video_units.py --data-root /path/to/lsfb-isol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from lsfb_dataset.utils.datasets import load_labels, load_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify video duration units (frames vs ms) by checking raw files via OpenCV."
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

    data_root = Path(args.data_root)
    csv_path = data_root / "instances.csv"
    videos_dir = data_root / "videos"

    # Enforce local 'mini_sample' split evaluation
    target_split = "mini_sample"

    print(f"Loading and filtering metadata for split '{target_split}'...")

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    # 1. Load split boundaries using library utilities
    try:
        instances = load_split(str(data_root), target_split)
        labels, _, _ = load_labels(str(data_root), args.n_labels)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load split or label utilities: {e}")

    # 2. Read and filter instances matching the local files
    df = pd.read_csv(csv_path)
    df = df[df["id"].isin(instances)]
    df = df[df["sign"].isin(labels)]

    if len(df) == 0:
        print("Warning: No records remaining after applying split filters.")
        return

    # Take up to 5 samples from the available mini_sample to analyze
    sample_size = min(5, len(df))
    sample_df = df.sample(n=sample_size, random_state=73)

    print("\n" + "=" * 95)
    print(
        f"{'Video Instance ID':<38} | {'CSV Delta':<10} | {'FPS':<6} | {'Video Frames':<12} | {'Duration':<10}"
    )
    print("-" * 95)

    for _, row in sample_df.iterrows():
        v_id = row["id"]
        csv_delta = int(row["end"] - row["start"])

        # Check for video file either in root or grouped inside a class/sign directory
        video_path = videos_dir / f"{v_id}.mp4"
        if not video_path.exists():
            video_path = videos_dir / str(row["sign"]) / f"{v_id}.mp4"

        if not video_path.exists():
            print(f"{v_id[:36]:<38} | {csv_delta:<10} | [Video file not found on disk]")
            continue

        # Inspect the physical file properties using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"{v_id[:36]:<38} | {csv_delta:<10} | [Error opening video file]")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        # FIXED: Removed 's' from inside the format specifier constraint block
        print(
            f"{v_id[:36]:<38} | {csv_delta:<10} | {fps:<6.1f} | {total_frames:<12} | {duration_sec:.2f}s"
        )

    print("=" * 95)
    print("\nHow to interpret these diagnostic results:")
    print(
        "1. If 'CSV Delta' values match 'Video Frames', your CSV is natively stored in FRAMES."
    )
    print(
        "2. If 'CSV Delta' divided by 1000 is close to 'Duration', your CSV is stored in MILLISECONDS."
    )
    print(
        "   Example: A CSV Delta of 300 with a Duration of 0.30s confirms MILLISECONDS.\n"
    )


if __name__ == "__main__":
    main()
