# Scripts <!-- omit from toc -->

CLI entry points for the LSFB-isol pipeline. Each script owns one task's command-
line parsing and orchestration; reusable scientific logic stays in `src/islr/`.
Experiment choices stay in YAML under `configs/` or `experiments/`, so scripts
remain independent and are never imported by one another.

## 1. Entry points

### 1.1. Training and evaluation

| Script | Responsibility |
|---|---|
| `train.py` | Loads a YAML experiment, trains the configured model, and saves the reproducible config, label map, logs, and resumable checkpoints. |
| `evaluate.py` | Loads one saved run and evaluates a selected checkpoint on one split. |
| `batch_evaluate.py` | Discovers standard output trees and delegates each run to the evaluation task, with a dry-run option. |
| `experiments.py` | Generates YAML configurations for the registered experiment grid; it does not run training. |

### 1.2. Data and splits

| Script | Responsibility |
|---|---|
| `data/download_data.py` | Downloads selected LSFB-isol dataset splits. |
| `data/splits/make_local_split.py` | Creates a small deterministic split for local smoke tests. |
| `data/splits/make_signer_dependent_splits.py` | Creates reproducible train/validation/test splits with signer overlap. |
| `data/splits/make_signer_independent_splits.py` | Builds SI train/validation/test splits from the existing folds. |
| `data/analyze_frames.py` | Reports physical video-frame duration statistics for pipeline configuration. |
| `data/verify_video_units.py` | Checks whether dataset timing metadata is expressed in frames or milliseconds. |
| `data/count_physical_frames.py` | Decodes one video and compares its physical frame count with the container header. |

## 2. Design intention

This layout satisfies the CLI boundary: scripts provide thin, task-specific
entry points, while models, datasets, transforms, training, metrics, and other
experiment behavior are implemented in `src/islr/` and selected through YAML.
That preserves reproducibility, configurable experimentation, and one entry
point per task without coupling CLI modules together.