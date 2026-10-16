# Architecture Requirements

These are the principles that the current structure either already satisfies or should be held accountable to going forward.

## R1 -- Reproducibility
Every run must be fully reproducible from its saved config.yaml alone. This includes the random seed (data workers, model init, augmentation), the split definition, the label map, and all hyperparameters. No value relevant to reproducing a result should be hardcoded in source.

## R2 -- Configurable experimentation
Swapping a model, a split, or a hyperparameter should require only a YAML change, not a code change. The model registry and the dataloader factory are the enforcement points for this requirement.

## R3 -- Separation of scientific and infrastructure concerns
src/islr/ contains only scientific logic (models, data, training). HPC execution details (job scripts, walltime, node allocation) live in jobs/. Environment details live in containers/. These layers must not import from each other.

## R4 -- Prefer well-tested libraries over custom implementations
Use PyTorch Lightning for training loops, torchmetrics for metrics, and TensorBoard for experiment tracking rather than reimplementing them. Custom code should exist only where the off-the-shelf solution genuinely doesn't fit (e.g. the LSFB-specific data pipeline, the I3D backbone).

## R5 -- Deterministic dependency resolution
uv.lock is the single source of truth for the environment. requirements.txt exists only as a human-readable HPC installation reference. The two must be kept in sync.

## R6 -- Data pipeline correctness across splits
The label map must be built from the full dataset before any split filtering, so class indices are identical across train_test, fold_N, and all runs. Cross-validation results are only comparable if this invariant holds.

## R7 -- Crash resilience on HPC
Training runs must be resumable from the last checkpoint by re-running the same command with the same arguments. No manual intervention or flag changes should be needed after a preempted job.

## R8 -- Clear output artifacts
Each run produces a self-contained output directory containing: the full config, the label map, checkpoints (best and last), and logs. The output directory alone should be sufficient to reproduce inference or continue training.

## R9 -- Single entry point per task
There is one training script (train.py). Future tasks (evaluation, inference, feature extraction) get their own scripts but share the same library layer. Scripts never import from each other.
