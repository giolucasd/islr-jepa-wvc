"""
Creates signer-dependent (SD) split files for LSFB-isol.

Every signer appears in all three sets wherever possible. Clips are
partitioned per-signer and stratified by sign class so that class
balance and signer coverage are preserved across train / val / test.

Allocation priority: test -> train -> val.
Test is guaranteed first because it is used for final reporting.
Val receives the remainder after test and train are filled; for very
sparse (signer, sign) pairs it may receive zero clips.

Default ratios (40/20/40) are chosen to match the approximate data
volumes of the signer-independent protocol (SI), making the two
regimes directly comparable in terms of training set size.

Split files written to metadata/splits/:
    sd_train.json
    sd_val.json
    sd_test.json

Usage
-----
    python scripts/splits/make_signer_dependent_splits.py \
        --data-root ~/data/lsfb-isol \
        --train-ratio 0.4 \
        --val-ratio 0.2 \
        --seed 73
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd


def save_split(splits_dir: Path, name: str, ids: list[str]) -> None:
    out_path = splits_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    print(f"  {name}.json  ->  {len(ids)} clips")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create signer-dependent split files for LSFB-isol."
    )
    parser.add_argument(
        "--data-root", required=True, help="Path to lsfb-isol directory."
    )
    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, "
            f"got {args.train_ratio} + {args.val_ratio} = "
            f"{args.train_ratio + args.val_ratio:.3f}"
        )

    print(
        f"Target ratios -- "
        f"train: {args.train_ratio:.1%}  "
        f"val: {args.val_ratio:.1%}  "
        f"test: {test_ratio:.1%}"
    )

    root = Path(args.data_root).expanduser()
    splits_dir = root / "metadata" / "splits"
    instances = pd.read_csv(root / "instances.csv")

    random.seed(args.seed)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    signers = sorted(instances["signer"].unique())
    print(f"Partitioning {len(signers)} signers...")

    # Track sparse pairs for the summary report
    n_singleton_pairs = 0  # only 1 clip -> goes to train, missing from val+test
    n_pair_pairs = 0  # only 2 clips -> test + train, missing from val

    for signer in signers:
        signer_df = instances[instances["signer"] == signer]

        by_class: dict[str, list[str]] = defaultdict(list)
        for _, row in signer_df.iterrows():
            by_class[row["sign"]].append(row["id"])

        signer_train, signer_val, signer_test = [], [], []

        for clip_ids in by_class.values():
            shuffled = clip_ids.copy()
            random.shuffle(shuffled)
            n = len(shuffled)

            if n == 1:
                # Can only go to one set -- train has highest sample value
                signer_train.extend(shuffled)
                n_singleton_pairs += 1
                continue

            # Priority: test -> train -> val
            # Test is filled first to guarantee final-reporting coverage.
            n_test = max(1, round(n * test_ratio))
            n_test = min(n_test, n - 1)  # always leave at least 1 for train

            remaining = n - n_test

            if remaining == 1:
                # Only 1 clip left after test -- give it to train
                signer_train.extend(shuffled[:1])
                signer_test.extend(shuffled[1 : 1 + n_test])
                n_pair_pairs += 1
                continue

            # Split remaining between train and val proportionally
            train_share = args.train_ratio / (args.train_ratio + args.val_ratio)
            n_train = max(1, round(remaining * train_share))
            n_train = min(n_train, remaining)
            n_val = remaining - n_train

            signer_train.extend(shuffled[:n_train])
            signer_val.extend(shuffled[n_train : n_train + n_val])
            signer_test.extend(shuffled[n_train + n_val :])

        train_ids.extend(signer_train)
        val_ids.extend(signer_val)
        test_ids.extend(signer_test)

    # ---- sanity checks --------------------------------------------------
    assert len(set(train_ids) & set(val_ids)) == 0, "Overlap: train ∩ val"
    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap: train ∩ test"
    assert len(set(val_ids) & set(test_ids)) == 0, "Overlap: val ∩ test"

    all_assigned = set(train_ids) | set(val_ids) | set(test_ids)
    all_instances = set(instances["id"])
    assert all_assigned == all_instances, (
        f"{len(all_instances - all_assigned)} clips not assigned to any split"
    )

    def signer_set(ids: list[str]) -> set[str]:
        return set(instances[instances["id"].isin(set(ids))]["signer"])

    all_signers = set(signers)
    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:
        missing = all_signers - signer_set(split_ids)
        if missing:
            warnings.warn(
                f"{len(missing)} signer(s) missing from {split_name}: {missing}. "
                "This means all their (signer, sign) pairs had only 1 clip.",
                UserWarning,
                stacklevel=1,
            )

    # ---- write files ----------------------------------------------------
    print("\nWriting SD split files...")
    save_split(splits_dir, "sd_train", train_ids)
    save_split(splits_dir, "sd_val", val_ids)
    save_split(splits_dir, "sd_test", test_ids)

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(
        f"\nActual ratios -- "
        f"train: {len(train_ids) / total:.1%}  "
        f"val: {len(val_ids) / total:.1%}  "
        f"test: {len(test_ids) / total:.1%}"
    )

    if n_singleton_pairs > 0:
        print(
            f"\nNote: {n_singleton_pairs} (signer, sign) pairs had 1 clip "
            f"-> assigned to train only (missing from val and test)."
        )
    if n_pair_pairs > 0:
        print(
            f"Note: {n_pair_pairs} (signer, sign) pairs had 2 clips "
            f"-> split between test and train (missing from val)."
        )


if __name__ == "__main__":
    main()
