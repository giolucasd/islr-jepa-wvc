import argparse
import subprocess
from pathlib import Path

# Available models
MODELS = ["vjepa", "i3d"]

# Mapping protocol directory names to their respective evaluation splits
PROTOCOL_SPLITS = {
    "legacy-split": "test",
    "sd-split": "sd_test",
    "si-split": "si_test",
}


def batch_evaluate(output_root, dry_run=False):
    base_dir = Path(output_root)

    if not base_dir.exists():
        print(f"❌ Error: Base directory '{base_dir}' not found.")
        return

    # Count runs for a final summary
    total_runs = 0

    # 1. Iterate through architecture directories (e.g., /work/user/vjepa/, /work/user/i3d/)
    for model_type in MODELS:
        model_type_dir = base_dir / model_type
        if not model_type_dir.is_dir():
            continue

        # 2. Iterate through our expected protocol folders
        for protocol, split in PROTOCOL_SPLITS.items():
            protocol_dir = model_type_dir / protocol

            # Skip if this specific model type doesn't have this protocol folder
            if not protocol_dir.exists():
                continue

            # 3. Iterate through each model run inside the protocol directory
            for run_dir in protocol_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                total_runs += 1

                # Construct the command
                cmd = [
                    "python",
                    "-m",
                    "scripts.evaluate",
                    "--run-dir",
                    str(run_dir),
                    "--split",
                    split,
                ]

                prefix = "[DRY RUN] Would execute" if dry_run else "🚀 Executing"
                print(f"{prefix}: {' '.join(cmd)}")

                if not dry_run:
                    try:
                        # Runs the command and waits for it to finish before moving to the next
                        subprocess.run(cmd, check=True)
                        print(f"✅ Successfully evaluated: {run_dir}\n")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Error while evaluating {run_dir}: {e}\n")
                        # The script continues to the next model even if one fails
                        continue
                    except KeyboardInterrupt:
                        print("\n🛑 Evaluation batch cancelled by user.")
                        return

    print(f"--- Found and processed a total of {total_runs} runs. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch run evaluations for all trained models."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory containing model output directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print out the commands that will be run without actually executing them.",
    )
    args = parser.parse_args()

    batch_evaluate(output_root=args.output_root, dry_run=args.dry_run)
