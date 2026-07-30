import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .nn import normalization


class Mamba3DBlock(nn.Module):
    """
    Simple 3D Mamba block for tensors shaped [B, C, D, H, W].

    Best used at low spatial resolution, especially the U-Net bottleneck.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.norm = normalization(channels, num_groups)

        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        residual = x

        x = self.norm(x)

        b, c, d, h, w = x.shape

        # [B, C, D, H, W] -> [B, L, C]
        x_seq = x.flatten(2).transpose(1, 2).contiguous()

        # Mamba sequence modeling
        x_seq = self.mamba(x_seq)

        # [B, L, C] -> [B, C, D, H, W]
        x = x_seq.transpose(1, 2).reshape(b, c, d, h, w).contiguous()

        x = self.proj(x)

        return residual + x