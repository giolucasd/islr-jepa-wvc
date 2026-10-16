import pytorch_lightning as pl
import torch


def setup_reproducibility(seed: int) -> None:
    """Sets up reproducibility by fixing random seeds and configuring deterministic behavior.

    Args:
        seed (int): The seed value to use for random number generators.
    """
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
