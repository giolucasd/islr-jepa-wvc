# Configuration Guide <!-- omit from toc -->

This directory contains the YAML configuration files for training models on the LSFB-isol dataset. The structure is grouped by conceptual ownership and user intent rather than internal execution mechanics, placing hardware options, input sizing, and optimization constraints into dedicated blocks.

- [1. Example Configurations](#1-example-configurations)
  - [1.1. I3D Baseline (configs/si.yaml)](#11-i3d-baseline-configssiyaml)
  - [1.2. V-JEPA Fine-Tuning (configs/si-phase2.yaml)](#12-v-jepa-fine-tuning-configssi-phase2yaml)
- [2. Parameter Reference](#2-parameter-reference)
  - [2.1. Global Context (global)](#21-global-context-global)
  - [2.2. Execution Mechanics (execution)](#22-execution-mechanics-execution)
  - [2.3. Dataset Context (dataset)](#23-dataset-context-dataset)
  - [2.4. Video Pipeline \& Preprocessing (pipeline)](#24-video-pipeline--preprocessing-pipeline)
  - [2.5. Model Weights \& Layers (model)](#25-model-weights--layers-model)
  - [2.6. Optimization Algorithms (optimizer)](#26-optimization-algorithms-optimizer)
  - [2.7. Training Rules \& Scheduling (`training`)](#27-training-rules--scheduling-training)


## 1. Example Configurations

### 1.1. I3D Baseline (configs/si.yaml)

```yaml
global:
  seed: 73
  model_name: i3d

execution:
  batch_size: 32
  accumulate_grad_batches: 1
  num_workers: 8
  use_gradient_checkpointing: false

dataset:
  data_root: /path/to/lsfb-isol
  train_split: si_train
  val_split: si_val
  num_classes: 700

pipeline:
  max_frames: 80
  num_frames: 48
  crop_size: 224

model:
  pretrained_path: /path/to/i3d_rgb_charades.pt
  i3d:
    head_dropout: 0.23

optimizer:
  name: sgd
  lr: 0.032
  momentum: 0.97
  weight_decay: 0.0000033

training:
  max_epochs: 50
  warmup_epochs: 5
  resume_from: null
```

### 1.2. V-JEPA Fine-Tuning (configs/si-phase2.yaml)

```yaml
global:
  seed: 73
  model_name: vjepa

execution:
  batch_size: 16
  accumulate_grad_batches: 2
  num_workers: 4
  use_gradient_checkpointing: true

dataset:
  data_root: /path/to/lsfb-isol
  train_split: si_train
  val_split: si_val
  num_classes: 700

pipeline:
  max_frames: 80
  num_frames: 48
  crop_size: 384

model:
  pretrained_path: /path/to/vjepa2_1_vitb_dist_vitG_384.pt
  vjepa:
    head_dropout: 0.4

optimizer:
  name: adamw
  lr: 0.0000033
  weight_decay: 0.056

training:
  max_epochs: 56
  warmup_epochs: 5
  resume_from: /path/to/phase1_checkpoint.ckpt
```

## 2. Parameter Reference

### 2.1. Global Context (global)

Global session markers and orchestrator behaviors.

* **seed**: Determines random initialization behavior for data workers and reproducibility targets.
* **model_name**: The primary architecture switcher (i3d or vjepa). Serves as the key used by model builders and transform factories.

### 2.2. Execution Mechanics (execution)

Hardware-specific configuration variables grouped together to simplify fitting experiments on available nodes.

* **batch_size**: Hardware-level mini-batch size per forward step.
* **accumulate_grad_batches**: Number of backward passes before stepping the optimizer. Effective Batch Size = batch_size * accumulate_grad_batches.
* **num_workers**: Data pre-fetching process limits.
* **use_gradient_checkpointing**: Activates activation checkpointing to preserve VRAM at the cost of recomputing activations.

### 2.3. Dataset Context (dataset)

Tracks the configuration layout of files stored raw on local disk storage.

* **data_root**: Absolute path containing source datasets.
* **train_split / val_split**: Protocol partitioning json definitions.
* **num_classes**: Target index class label constraints.

### 2.4. Video Pipeline & Preprocessing (pipeline)

Structural variables shared between dataloaders and models that define the shape of tensors moving through the network.

* **max_frames**: Maximum threshold boundary read constraint for source videos.
* **num_frames**: Downsampled target frame length passed down into classification backbones.
* **crop_size**: Spatial length sizing constraint used for video bounding transforms and ViT patch grids.

### 2.5. Model Weights & Layers (model)

Architectural settings and nested specific sub-dictionaries.

* **pretrained_path**: System filepath linking to initial foundation checkpoints.
* **i3d / vjepa Sub-Blocks**: Sub-dictionaries matching your active global.model_name that contain specialized parameters:
* **head_dropout**: Regularization drop-rate applied directly before or inside classification layers.

### 2.6. Optimization Algorithms (optimizer)

* **name**: Optimizer class picker (sgd or adamw).
* **lr**: Learning rate coefficient.
* **weight_decay**: Regularization scaling penalty.
* **momentum**: Momentum factor (evaluated only if name is sgd).

### 2.7. Training Rules & Scheduling (`training`)

* **`max_epochs`**: Defines the absolute global epoch ceiling at which training terminates. When loading a checkpoint to start a new phase (e.g., resuming at epoch 26 for Phase 2), set this value to your final intended milestone (e.g., `56`). This ensures the sequential learning rate scheduler matches the absolute limits of your target timeline.
* **`warmup_epochs`**: Duration boundary for linear scheduling ramps.
* **`resume_from`**: (Optional) Path to a specific `.ckpt` file to initialize weights/states for the next phase.
* **`load_weights_from`**: (Optional) Path to a specific `.ckpt` file to initialize only the weights for the next phase.
