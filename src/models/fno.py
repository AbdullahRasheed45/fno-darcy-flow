"""Fourier Neural Operator (2D), after Li et al., ICLR 2021.

A compact, readable implementation: lifting layer -> N spectral blocks
(spectral convolution + pointwise linear + GELU) -> projection head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """Pointwise multiplication of the lowest Fourier modes by learned weights."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (batch, in_ch, x, y) x (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_ch, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes1, : self.modes2] = self._mul(
            x_ft[:, :, : self.modes1, : self.modes2], self.w1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._mul(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.w2
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class FNO2d(nn.Module):
    """Maps an input field a(x, y) (plus grid coords) to an output field u(x, y)."""

    def __init__(self, modes: int = 12, width: int = 32, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(3, width)  # (a, x, y) -> width channels
        self.blocks = nn.ModuleList(FNOBlock(width, modes) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def _grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        b, h, w = shape[0], shape[-2], shape[-1]
        gx = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(b, h, w)
        gy = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(b, h, w)
        return torch.stack((gx, gy), dim=-1)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (batch, h, w) -> u: (batch, h, w)
        grid = self._grid(a.shape, a.device)
        x = torch.cat((a.unsqueeze(-1), grid), dim=-1)      # (b, h, w, 3)
        x = self.lift(x).permute(0, 3, 1, 2)                # (b, width, h, w)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)                           # (b, h, w, width)
        return self.head(x).squeeze(-1)
