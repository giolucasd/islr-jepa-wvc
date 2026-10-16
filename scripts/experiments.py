import copy
from pathlib import Path

import yaml

# ==========================================
# 1. BASE CONFIGURATION TEMPLATE
# ==========================================
BASE_CONFIG = {
    "global": {
        "seed": 73,
        "model_name": "vjepa",
    },
    "execution": {
        "batch_size": 32,
        "accumulate_grad_batches": 1,
        "num_workers": 4,
        "use_gradient_checkpointing": False,
    },
    "dataset": {
        "data_root": "/path/to/lsfb-isol",
        "train_split": "si_train",
        "val_split": "si_val",
        "num_classes": 700,
    },
    "pipeline": {
        "max_frames": 80,
        "num_frames": 16,
        "crop_size": 384,
        "target_fps": 50.0,
    },
    "model": {
        "pretrained_path": "/path/to/vjepa2_1_vitb_dist_vitG_384.pt",
        "freeze_encoder": True,
        "vjepa": {
            "head_dropout": 0.4,
        },
    },
    "optimizer": {
        "name": "adamw",
        "lr": 2.2e-4,
        "weight_decay": 1.0e-4,
    },
    "training": {
        "max_epochs": 50,
        "warmup_epochs": 5,
        "resume_from": None,
        "load_weights_from": None,
    },
}

# ==========================================
# 2. DEFINITIONS & MAPPINGS
# ==========================================

# Dataset Splits
SPLITS = {
    "reduced-split": {
        "num_classes": 395,
        "train_split": "train",
        "val_split": "test",
    },
    "si-split": {
        "num_classes": 700,
        "train_split": "si_train",
        "val_split": "si_val",
    },
    "sd-split": {
        "num_classes": 700,
        "train_split": "sd_train",
        "val_split": "sd_val",
    },
}

# Frame dependent execution parameters
FRAME_EXEC_MAP = {
    16: {"batch_size": 32, "accumulate_grad_batches": 1},
    32: {"batch_size": 16, "accumulate_grad_batches": 2},
    48: {"batch_size": 8, "accumulate_grad_batches": 4},
}

# Phase configurations
PHASE_CONFIGS = {
    1: {
        "freeze_encoder": True,
        "use_gradient_checkpointing": False,
        "lr": 2.2e-4,
        "weight_decay": 1.0e-4,
        "resume_from": None,
    },
    2: {
        "freeze_encoder": False,
        "use_gradient_checkpointing": True,
        "lr": 3.3e-6,
        "weight_decay": 5.6e-2,
        # Default placeholder path pointing to phase 1 checkpoint
        "resume_from": "/path/to/phase1.ckpt",
    },
}

# Single-variable Ablations wrt Base Config (16f, 50.0fps, 384px)
ABLATION_GRID = [
    # Base configuration
    {"num_frames": 16, "target_fps": 50.0, "crop_size": 384},
    # Ablations: crop_size
    {"num_frames": 16, "target_fps": 50.0, "crop_size": 224},
    # Ablations: num_frames
    {"num_frames": 48, "target_fps": 50.0, "crop_size": 384},
    # Ablations: target_fps
    {"num_frames": 16, "target_fps": 4.0, "crop_size": 384},
    {"num_frames": 16, "target_fps": 16.0, "crop_size": 384},
    {"num_frames": 16, "target_fps": 25.0, "crop_size": 384},
]


def build_filename(
    num_frames: int,
    target_fps: float,
    phase: int,
    crop_size: int,
) -> str:
    """Builds the standardized YAML filename."""
    fps_val = int(target_fps) if target_fps.is_integer() else target_fps

    # Format: vjepa-<num_frames>f-<target_fps>fps-<crop_size>px-p<stage>
    base_name = f"vjepa-{num_frames}f-{fps_val}fps-{crop_size}px-p{phase}"

    return f"{base_name}.yaml"


def generate_experiment_configs(output_dir: str = "experiments"):
    root_path = Path(output_dir) / "vjepa"
    count = 0

    for split_name, split_params in SPLITS.items():
        split_dir = root_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Restrict ablations strictly to si-split; default config for all others
        grid = ABLATION_GRID if split_name == "si-split" else [ABLATION_GRID[0]]

        for phase in [1, 2]:
            phase_params = PHASE_CONFIGS[phase]

            for params in grid:
                num_frames = params["num_frames"]
                target_fps = params["target_fps"]
                crop_size = params["crop_size"]

                # Create deepcopy of base dictionary
                cfg = copy.deepcopy(BASE_CONFIG)

                # Update Dataset parameters
                cfg["dataset"].update(split_params)

                # Update Pipeline parameters
                cfg["pipeline"]["num_frames"] = num_frames
                cfg["pipeline"]["target_fps"] = float(target_fps)
                cfg["pipeline"]["crop_size"] = crop_size

                # Update Frame-dependent Execution parameters
                exec_params = FRAME_EXEC_MAP[num_frames]
                cfg["execution"]["batch_size"] = exec_params["batch_size"]
                cfg["execution"]["accumulate_grad_batches"] = exec_params[
                    "accumulate_grad_batches"
                ]
                cfg["execution"]["use_gradient_checkpointing"] = phase_params[
                    "use_gradient_checkpointing"
                ]

                # Update Model & Optimizer Phase parameters
                cfg["model"]["freeze_encoder"] = phase_params["freeze_encoder"]
                cfg["optimizer"]["lr"] = phase_params["lr"]
                cfg["optimizer"]["weight_decay"] = phase_params["weight_decay"]

                # Update Training Phase parameters
                if phase == 2:
                    p1_filename = build_filename(
                        num_frames,
                        target_fps,
                        1,
                        crop_size,
                    )
                    cfg["training"]["resume_from"] = (
                        f"/path/to/checkpoints/{p1_filename.replace('.yaml', '.ckpt')}"
                    )
                else:
                    cfg["training"]["resume_from"] = None

                # Generate File Path and Save
                filename = build_filename(
                    num_frames,
                    target_fps,
                    phase,
                    crop_size,
                )
                file_path = split_dir / filename

                with open(file_path, "w") as f:
                    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

                count += 1
                print(f"Generated: {file_path}")

    print(f"\nSuccessfully generated {count} configuration files across all splits!")


if __name__ == "__main__":
    generate_experiment_configs()
