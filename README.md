# Sign Language Recognition with Video Foundation Models: V-JEPA Transfer and Adaptation <!-- omit from toc -->

This repository provides a reproducible and configurable training pipeline for isolated sign language recognition (ISLR) on the LSFB-isol benchmark. It is designed for video-based foundation models (e.g., V-JEPA, I3D), emphasizing clean experiment definitions, modular model integration, and straightforward evaluation.

- [1. Key Features](#1-key-features)
- [2. Project Structure](#2-project-structure)
- [3. Installation](#3-installation)
- [4. Data Preparation](#4-data-preparation)
  - [4.1. Expected Directory Layout](#41-expected-directory-layout)
- [5. Usage](#5-usage)
  - [5.1. Training](#51-training)
  - [5.2. Evaluation](#52-evaluation)
- [6. Pretrained Weights \& Experimental Artifacts](#6-pretrained-weights--experimental-artifacts)
  - [6.1. Experiment Configurations](#61-experiment-configurations)
  - [6.2. Pretrained Model Weights](#62-pretrained-model-weights)
  - [6.3. Training Artifacts \& Logs](#63-training-artifacts--logs)
- [7. Citation](#7-citation)


## 1. Key Features

- **Config-Driven Workflow**: Fully controlled via YAML configurations for dataset, model, and training hyperparameters.
- **Modular Architecture**: Model-agnostic PyTorch Lightning training module for quick integration of new encoders.
- **Reproducible Artifacts**: Automatically saves exact configs, checkpoints, TensorBoard logs, and label maps per run.
- **Dataset Tooling**: Integrated data downloading and split generation for LSFB-isol setups.

## 2. Project Structure

- `configs/`: YAML experiment configurations.
- `experiments/`: Pre-generated JSON split files and trained model configs.
- `jobs/`: Cluster job submission scripts (HPC).
- `scripts/`: CLI entry points for data downloading, split generation, training, and evaluation.
- `src/islr/`: Core codebase (models, datasets, lightnings modules, utils).

## 3. Installation

Requires Python 3.12+. We recommend using [`uv`](https://github.com/astral-sh/uv).

```bash
# Create environment and sync dependencies
uv venv
uv sync
```

`uv` creates an isolated environment and installs dependencies quickly from the checked-in `uv.lock` file. This keeps development and experiment environments consistent across machines, matching the project's focus on reproducibility.

## 4. Data Preparation

Download LSFB-isol dataset assets directly using the included utility:

```bash
python -m scripts.data.download_data \
    --data-root /path/to/data \
    --splits all
```

*Note: The default `--data-root` is `~/lsfb-isol`.*

### 4.1. Expected Directory Layout

```text
lsfb-isol/
├── download_lsfb_isol.py
├── instances.csv
├── instances_special.csv
├── requirements.txt
├── videos/
└── metadata/
    ├── sign_occurrences.csv
    ├── sign_to_index.csv
    └── splits/
        ├── all.json
        ├── mini_sample.json
        ├── si_train.json / si_val.json / si_test.json
        ├── sd_train.json / sd_val.json / sd_test.json
        └── fold_0.json ... fold_4.json
```

Custom signer-dependent (SD) and signer-independent (SI) splits can be regenerated via `scripts.data.splits.make_signer_dependent_splits` and `scripts.data.splits.make_signer_independent_splits`. The exact split JSONs used in the paper are provided in `experiments/`.

## 5. Usage

### 5.1. Training

Launch a training run using a configuration file:

```bash
python -m scripts.train \
    --config configs/untracked/si-split/vjepa-16f-25fps-384px-p1.yaml \
    --output-path /work/user/vjepa/si-split/vjepa-16f-25fps-384px-p1
```

All run artifacts (checkpoints, metrics, label maps) will be stored in `<output-path>/`.

### 5.2. Evaluation

Evaluate a trained checkpoint on a target dataset split:

```bash
python -m scripts.evaluate \
    --run-dir /work/user/vjepa/si-split/vjepa-16f-25fps-384px-p1 \
    --split si_test
```

## 6. Pretrained Weights & Experimental Artifacts

### 6.1. Experiment Configurations
Experiment configurations (`.yaml`) for all runs are version-controlled in the [`experiments/`](experiments/) directory. Note that to reproduce experiments, you have to fill `datasets.data_root`, `model.pretrained_path` and `training.resume_from`.

### 6.2. Pretrained Model Weights

Pretrained weights for our primary models evaluated on the Signer-Independent (SI) split are hosted on Hugging Face:

| Model | Split | Resolution | Frames / FPS | Micro Top-1 | Download |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **I3D (Baseline)** | SI | 224×224 | 48f / 50fps | 51.77% | [🤗 Model Card](https://huggingface.co/giolucasd/i3d-48f-50fps-224px) |
| **V-JEPA (Full FT)** | SI | 384×384 | 16f / 25fps | **57.78%** | [🤗 Model Card](https://huggingface.co/giolucasd/vjepa-16f-25fps-384px-p2) |

### 6.3. Training Artifacts & Logs

All additional experimental artifacts -- including TensorBoard event logs, metric summaries, training and validation curves -- are archived and directly downloadable via [GitHub Release v1.0.0](https://github.com/giolucasd/islr-jepa-wvc/releases/tag/v1.0.0) (`experimental-artifacts-no-weights.zip`, 1.3 MB).

## 7. Citation

If you use this repository or our findings in your research, please cite our paper:

```bibtex
@inproceedings{mariano2026sign,
  title={Sign Language Recognition with Video Foundation Models: V-JEPA Transfer and Adaptation},
  author={Giovanne Mariano and H{\'e}lio Pedrini},
  booktitle={Workshop de Vis{\~a}o Computacional (WVC)},
  year={2026}
}
```
