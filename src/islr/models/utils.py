from typing import Any

from islr.models.base import ISLRModel
from islr.models.i3d.classifier import I3DClassifier
from islr.models.vjepa.classifier import VJEPAClassifier

MODEL_REGISTRY = {
    "i3d": I3DClassifier,
    "vjepa": VJEPAClassifier,
}


def build_model_from_config(config: dict[str, Any]) -> ISLRModel:
    """
    Builds an ISLR model from the unified configuration dictionary.

    Extracts global metadata, input pipeline requirements, and architecture-specific
    hyperparameters, packaging them into a flattened dictionary for the model constructor.
    """
    # 1. Identify the targeted model architecture
    model_name = config["global"]["model_name"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # 2. Extract configuration subsets
    dataset_cfg = config["dataset"]
    pipeline_cfg = config["pipeline"]
    execution_cfg = config["execution"]
    model_base_cfg = config["model"]

    # Extract nested, model-specific hyperparameters (e.g., config["model"]["vjepa"])
    model_specific_cfg = model_base_cfg.get(model_name, {})

    # 3. Flatten into a clean parameter dictionary for the model constructor
    model_args = {
        # Global & Pipeline structural constraints
        "num_classes": dataset_cfg["num_classes"],
        "num_frames": pipeline_cfg["num_frames"],
        "crop_size": pipeline_cfg["crop_size"],
        # Core shared backbone properties
        "pretrained_path": model_base_cfg.get("pretrained_path"),
        "use_gradient_checkpointing": execution_cfg.get(
            "use_gradient_checkpointing", False
        ),
        # Architecture-specific parameters (like head_dropout)
        **model_specific_cfg,
    }

    return MODEL_REGISTRY[model_name](model_args)
