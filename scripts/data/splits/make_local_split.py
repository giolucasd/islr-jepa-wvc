"""
Creates a 'local_sample.json' split file for local smoke-testing.

Picks the first N instance IDs from mini_sample.json and writes them to
metadata/splits/local_sample.json so the standard pipeline can load them
with split="local_sample" without any code changes.

Usage
-----
    python scripts/make_local_split.py --data-root ~/data/lsfb-isol --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", required=True, help="Path to lsfb-isol directory."
    )
    parser.add_argument("--n", type=int, default=3, help="Number of instances to keep.")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser()
    splits_dir = root / "metadata" / "splits"

    mini_path = splits_dir / "mini_sample.json"
    if not mini_path.exists():
        raise FileNotFoundError(f"mini_sample.json not found at {mini_path}")

    with open(mini_path, encoding="utf-8") as f:
        mini_ids: list = json.load(f)

    local_ids = mini_ids[: args.n]

    out_path = splits_dir / "local_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(local_ids, f, indent=2)

    print(f"Wrote {len(local_ids)} instance IDs to {out_path}")
    for id_ in local_ids:
        print(f"  {id_}")


if __name__ == "__main__":
    main()
