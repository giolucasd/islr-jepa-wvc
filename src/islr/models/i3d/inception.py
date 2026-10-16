"""
Inception-I3D backbone (RGB stream).

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Carreira & Zisserman, CVPR 2017 - https://arxiv.org/abs/1705.07750
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division: equivalent to math.ceil(a / b)."""
    return (a + b - 1) // b


class MaxPool3dSamePadding(nn.MaxPool3d):
    """MaxPool3d with TensorFlow-style 'SAME' padding."""

    def _compute_pad(self, dim: int, s: int) -> int:
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()

        pad_t = self._compute_pad(0, t)
        pad_h = self._compute_pad(1, h)
        pad_w = self._compute_pad(2, w)

        x = F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_t // 2,
                pad_t - pad_t // 2,
            ),
        )
        return super().forward(x)


class Unit3D(nn.Module):
    """3-D conv + optional BN + optional activation, with 'SAME' padding."""

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding: int = 0,
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = "unit_3d",
    ):
        super().__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,  # dynamic padding applied in forward()
            bias=use_bias,
        )

        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

    def _compute_pad(self, dim: int, s: int) -> int:
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()

        pad_t = self._compute_pad(0, t)
        pad_h = self._compute_pad(1, h)
        pad_w = self._compute_pad(2, w)

        x = F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_t // 2,
                pad_t - pad_t // 2,
            ),
        )

        x = self.conv3d(x)

        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int], name: str):
        super().__init__()

        self.b0 = Unit3D(
            in_channels,
            out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels,
            out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            out_channels[1],
            out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels,
            out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            out_channels[3],
            out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels,
            out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.b0(x),
                self.b1b(self.b1a(x)),
                self.b2b(self.b2a(x)),
                self.b3b(self.b3a(x)),
            ],
            dim=1,
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D architecture.

    Parameters
    ----------
    num_classes : int
        Size of the final logit layer.
    spatial_squeeze : bool
        Squeeze spatial dims (H, W) of the logits tensor.
    final_endpoint : str
        Build the network only up to this endpoint.
    in_channels : int
        Number of input channels (3 for RGB, 2 for optical flow).
    dropout_keep_prob : float
        Dropout probability for the logits head.
    use_gradient_checkpointing : bool
        When True, applies torch gradient checkpointing to each endpoint
        module, trading ~35 % extra compute for significantly lower
        activation memory.  Leave False (default) on A100 40 GB.
    """

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        final_endpoint: str = "Logits",
        name: str = "inception_i3d",
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
        use_gradient_checkpointing: bool = False,
    ):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint: {final_endpoint!r}")

        super().__init__()

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self._use_gradient_checkpointing = use_gradient_checkpointing
        self.logits = None

        self.end_points: dict[str, nn.Module] = {}

        def _add(ep, module):
            self.end_points[ep] = module
            return ep == self._final_endpoint  # True -> stop building

        if _add(
            "Conv3d_1a_7x7",
            Unit3D(
                in_channels,
                64,
                kernel_shape=[7, 7, 7],
                stride=(2, 2, 2),
                padding=(3, 3, 3),
                name=name + "Conv3d_1a_7x7",
            ),
        ):
            return
        if _add(
            "MaxPool3d_2a_3x3",
            MaxPool3dSamePadding([1, 3, 3], stride=(1, 2, 2), padding=0),
        ):
            return
        if _add(
            "Conv3d_2b_1x1",
            Unit3D(
                64, 64, kernel_shape=[1, 1, 1], padding=0, name=name + "Conv3d_2b_1x1"
            ),
        ):
            return
        if _add(
            "Conv3d_2c_3x3",
            Unit3D(
                64, 192, kernel_shape=[3, 3, 3], padding=1, name=name + "Conv3d_2c_3x3"
            ),
        ):
            return
        if _add(
            "MaxPool3d_3a_3x3",
            MaxPool3dSamePadding([1, 3, 3], stride=(1, 2, 2), padding=0),
        ):
            return
        if _add(
            "Mixed_3b",
            InceptionModule(192, [64, 96, 128, 16, 32, 32], name + "/Mixed_3b"),
        ):
            return
        if _add(
            "Mixed_3c",
            InceptionModule(256, [128, 128, 192, 32, 96, 64], name + "/Mixed_3c"),
        ):
            return
        if _add(
            "MaxPool3d_4a_3x3",
            MaxPool3dSamePadding([3, 3, 3], stride=(2, 2, 2), padding=0),
        ):
            return
        if _add(
            "Mixed_4b",
            InceptionModule(
                128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + "/Mixed_4b"
            ),
        ):
            return
        if _add(
            "Mixed_4c",
            InceptionModule(
                192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + "/Mixed_4c"
            ),
        ):
            return
        if _add(
            "Mixed_4d",
            InceptionModule(
                160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + "/Mixed_4d"
            ),
        ):
            return
        if _add(
            "Mixed_4e",
            InceptionModule(
                128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + "/Mixed_4e"
            ),
        ):
            return
        if _add(
            "Mixed_4f",
            InceptionModule(
                112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + "/Mixed_4f"
            ),
        ):
            return
        if _add(
            "MaxPool3d_5a_2x2",
            MaxPool3dSamePadding([2, 2, 2], stride=(2, 2, 2), padding=0),
        ):
            return
        if _add(
            "Mixed_5b",
            InceptionModule(
                256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + "/Mixed_5b"
            ),
        ):
            return
        if _add(
            "Mixed_5c",
            InceptionModule(
                256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + "/Mixed_5c"
            ),
        ):
            return

        # Logits head
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()

    # ------------------------------------------------------------------

    def replace_logits(self, num_classes: int) -> None:
        """Swap the classification head; call after loading Kinetics weights."""
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def build(self) -> None:
        for k, v in self.end_points.items():
            self.add_module(k, v)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for end_point in self.VALID_ENDPOINTS:
            if end_point not in self.end_points:
                continue
            module = self._modules[end_point]
            if self._use_gradient_checkpointing:
                x = checkpoint.checkpoint(module, x, use_reentrant=False)
            else:
                x = module(x)

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x  # (B, num_classes, T)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
