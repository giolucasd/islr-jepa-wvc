# V-JEPA 2.1 (Vendored)

This directory contains the vendored Vision Transformer implementation from the official V-JEPA 2.1 repository. 

## Why is this code vendored?
Our experiments are executed on High-Performance Computing (HPC) clusters that lack internet access at runtime. Because we cannot rely on `torch.hub` to dynamically download the model architectures, the required structural modules have been manually copied into this pipeline.

## Origin
The following files were extracted directly from the [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py) repository:
* `src/islr/models/vjepa/masks.py`
* `src/islr/models/vjepa/modules.py`
* `src/islr/models/vjepa/patch_embed.py`
* `src/islr/models/vjepa/tensors.py`
* `src/islr/models/vjepa/vision_transformer.py`

## Loading Checkpoints
Because internet access is restricted, weights cannot be fetched dynamically. You must:
1. Manually download the desired `.pt` checkpoint (e.g., `vjepa2_1_vitb_dist_vitG_384.pt`).
2. Provide the absolute path to this file via the `pretrained_path` parameter in your YAML model configuration.