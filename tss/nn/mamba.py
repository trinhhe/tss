"""
[1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao) https://arxiv.org/abs/2312.00752

Glossary:
    b: batch size ()        (`B` in [1] Algorithm 2)
    l: sequence length      (`L` in [1] Algorithm 2)
    d: channel dim          (`D` in [1] Algorithm 2)
"""

import torch
from mamba_ssm import Mamba as MambaBlock
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )
        return output


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mambablock = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand=2,
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)

        Returns:
            output: shape (b, l, d)

        """
        output = self.mambablock(self.norm(x)) + x
        return output


class Mamba(nn.Module):
    def __init__(
        self, n_layer: int, d_model: int, d_state: int, expand: int = 2
    ):
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                )
                for _ in range(n_layer)
            ]
        )
        # self.head = ?

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)

        Returns:
            logits: shape ???

        """
        for layer in self.layers:
            x = layer(x)
