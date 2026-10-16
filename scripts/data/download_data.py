"""
Download LSFB-isol data.

Usage
-----
    python -m scripts.data.download_data \
        --data-root /path/to/data \
        --splits all
"""

import argparse
import warnings
from pathlib import Path

from lsfb_dataset import Downloader
from urllib3.exceptions import InsecureRequestWarning


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the LSFB-isol dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "data" / "lsfb-isol",
        help="Root directory in which to store the dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["mini_sample"],
        help="Dataset splits to download (use 'all' to download all data).",
    )
    args = parser.parse_args()

    downloader = Downloader(
        destination=args.data_root,
        dataset="isol",
        splits=args.splits,
        include_videos=True,
        include_cleaned_poses=False,
        include_raw_poses=False,
        skip_existing_files=True,
        check_ssl=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InsecureRequestWarning)
        downloader.download()


if __name__ == "__main__":
    main()
