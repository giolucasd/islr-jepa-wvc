from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class BestValAcc(Callback):
    def __init__(self):
        self.best = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_acc = trainer.callback_metrics.get("val_acc_macro", 0.0)
        if isinstance(val_acc, torch.Tensor):
            val_acc = val_acc.item()
        self.best = max(self.best, val_acc)


class GpuMemoryLogger(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:  # log only first batch - enough to see peak alloc
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            trainer.logger.log_metrics(
                {"gpu_mem_allocated_gb": allocated, "gpu_mem_reserved_gb": reserved},
                step=trainer.global_step,
            )


def resolve_precision() -> str:
    """
    Select the best available floating-point precision.

    - A100 / H100 (HPC): bf16-mixed  - native hardware support, stable training
    - RTX 40xx (desktop): bf16-mixed - supported but falls back gracefully
    - CPU (laptop):       32-true    - bf16 not supported on CPU
    """
    if not torch.cuda.is_available():
        return "32-true"
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    return "16-mixed"  # fp16 fallback for older GPUs


def export_best_weights(checkpoint_callback: ModelCheckpoint) -> None:
    if not checkpoint_callback.best_model_path:
        return

    ckpt = torch.load(
        checkpoint_callback.best_model_path,
        map_location="cpu",
        weights_only=True,
    )
    out_path = Path(checkpoint_callback.best_model_path).parent / "best_weights.pt"
    torch.save(ckpt["state_dict"], out_path)
