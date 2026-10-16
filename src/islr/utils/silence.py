import logging
import os
import warnings


class LitLoggerFilter(logging.Filter):
    """Filter out promotional litlogger messages while keeping INFO logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return (
            "litlogger" not in message
            and "Trainer already configured with model summary callbacks:"
            not in message
        )


def silence_warnings_and_logs() -> None:
    """
    Silence warnings and logs that are expected and not actionable.

    This function is called at the top of scripts/train.py and scripts/evaluate.py
    to suppress warnings that are known to be benign in the context of this project.
    """
    # Environment variable approach (best practice for Lightning tips)
    os.environ["LIGHTNING_DISABLE_TIPS"] = "1"

    # Silence the PyTorch Lightning num_workers bottleneck warning
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")

    # Silence the PyTorch LR scheduler epoch deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=(".*The epoch parameter in `scheduler.step\\(\\)` was not necessary.*"),
    )

    # Silence the PyTorch Lightning LeafSpec deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    )

    # Attach filter to rank_zero logger to keep INFO logs (GPU info) intact
    rank_zero_logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    rank_zero_logger.addFilter(LitLoggerFilter())
