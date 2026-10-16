"""
Creates explicit signer-independent (SI) split files for LSFB-isol.

The five folds are already signer-disjoint. This script simply makes
the three-way partition explicit as JSON files so the training pipeline
can use them directly without relying on the library's implicit
fold-union logic.

Split assignment
----------------
    si_train.json  ->  folds 2 + 3  (39 signers, development training)
    si_val.json    ->  fold  4      (20 signers, model selection)
    si_test.json   ->  folds 0 + 1  (40 signers, final reporting only)

The test set (folds 0+1) matches the library's built-in `split="test"`,
ensuring consistency with any runs that used the library split directly.

Usage
-----
    python scripts/scripts/make_signer_independent_splits.py \
        --data-root ~/data/lsfb-isol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_fold(splits_dir: Path, fold: int) -> list[str]:
    with open(splits_dir / f"fold_{fold}.json", encoding="utf-8") as f:
        return json.load(f)


def save_split(splits_dir: Path, name: str, ids: list[str]) -> None:
    out_path = splits_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    print(f"  {name}.json  ->  {len(ids)} clips")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create signer-independent split files for LSFB-isol."
    )
    parser.add_argument(
        "--data-root", required=True, help="Path to lsfb-isol directory."
    )
    args = parser.parse_args()

    splits_dir = Path(args.data_root).expanduser() / "metadata" / "splits"

    print("Loading folds...")
    folds = {i: load_fold(splits_dir, i) for i in range(5)}

    for i, ids in folds.items():
        print(f"  fold_{i}: {len(ids)} clips")

    print("\nWriting SI split files...")
    save_split(splits_dir, "si_train", folds[2] + folds[3])
    save_split(splits_dir, "si_val", folds[4])
    save_split(splits_dir, "si_test", folds[0] + folds[1])

    print("\nDone. Verify signer disjointness with fold_check.py.")


if __name__ == "__main__":
    main()
