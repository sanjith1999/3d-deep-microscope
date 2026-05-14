"""
Two-layer scalar diffractive coder with a scalar depth spread, used to cross-check the vectorial model.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from coder.diffractive import D2NNStack, angle_scan_inputs


def get_G_masked(N: int, dx_m: float, dz_m: float, lambda_m: float, w: int, mask_factor: float,
                 device="cpu") -> torch.Tensor:
    """Angular-spectrum transfer function on a w*N grid with an NA mask."""
    fx = torch.arange(-1 / (2 * dx_m), 1 / (2 * dx_m), 1 / (w * N * dx_m))
    fx = torch.tile(fx, (1, w * N)).view(w * N, w * N).to(torch.cfloat)
    fy = torch.arange(1 / (2 * dx_m), -1 / (2 * dx_m), -1 / (w * N * dx_m)).view(w * N, 1)
    fy = torch.tile(fy, (1, w * N)).view(w * N, w * N).to(torch.cfloat)
    non_ev = (abs(fx) ** 2 + abs(fy) ** 2 <= (1 / lambda_m) ** 2 * mask_factor)
    power = 1j * 2 * np.pi * torch.sqrt((1 / lambda_m ** 2) - (fx ** 2) - (fy ** 2)) * dz_m
    return (torch.exp(power * non_ev) * non_ev).to(device)


class ScalarExcitation(nn.Module):

    def __init__(self, N: int, dx_um: float, dz_um: float, nz: int, na_ex: float,
                 lambda_um: float = 0.532, w: int = 8, device="cpu"):
        super().__init__()
        self.N, self.nz, self.w, self.dev = N, nz, w, device
        dx, lam, dz = dx_um * 1e-6, lambda_um * 1e-6, dz_um * 1e-6
        # one transfer function per depth plane (dist relative to focus at nz//2), masked by NA^2
        G = [get_G_masked(N, dx, (z - nz // 2) * dz, lam, w, na_ex ** 2, device) for z in range(nz)]
        self.register_buffer("G", torch.stack(G, 0))                # (nz, wN, wN)

    def spread(self, field: torch.Tensor, chunk: int = 0, ckpt: bool = False) -> torch.Tensor:
        """field (T, N, N) complex -> H2 (T, nz, N, N) real intensity."""
        T, N, w = field.shape[0], self.N, self.w
        c = (w - 1) * N // 2
        step = chunk if chunk else T

        def plane(Gz):
            outs = []
            for s in range(0, T, step):
                pad = torch.zeros(min(step, T - s), w * N, w * N, dtype=torch.cfloat, device=field.device)
                pad[:, c:c + N, c:c + N] = field[s:s + step]
                A = torch.fft.fftshift(torch.fft.fft2(pad), dim=(-2, -1)) * Gz
                B = torch.fft.ifft2(torch.fft.ifftshift(A, dim=(-2, -1)))
                outs.append(B[:, c:c + N, c:c + N].abs().square())
            return torch.cat(outs, 0)

        planes = []
        for z in range(self.nz):
            Gz = self.G[z].view(1, w * N, w * N)
            if ckpt and field.requires_grad:
                planes.append(torch.utils.checkpoint.checkpoint(plane, Gz, use_reentrant=False))
            else:
                planes.append(plane(Gz))
        return torch.stack(planes, 1)


class ScalarDiffractiveCoder(nn.Module):
    """angle-scan inputs -> 2-layer phase D2NN -> scalar depth spread -> H2 (T, nz, N, N)."""

    def __init__(self, cfg):
        super().__init__()
        dev = cfg.device
        N, dx = cfg.n_scalar, cfg.pitch_scalar          # analog grid directly (128 @ 0.25)
        self.d2nn = D2NNStack(N, dx, cfg.wavelength_um, cfg.n_layers_scalar, cfg.delta_z_um,
                              w=cfg.d2nn_window_scalar, device=dev)
        self.register_buffer("inputs", angle_scan_inputs(
            N, dx, cfg.wavelength_um, cfg.n_alphas, cfg.n_thetas, device=dev))
        self.exc = ScalarExcitation(N, dx, cfg.dz_um, cfg.nz, cfg.na_ex,
                                    lambda_um=cfg.wavelength_um, w=cfg.exc_window_scalar, device=dev)

    @property
    def phases(self):
        return self.d2nn.phases

    @property
    def n_patterns(self) -> int:
        return self.inputs.shape[0]

    @property
    def focus_idx(self) -> int:
        return self.exc.nz // 2                          # spread centres at nz//2

    def forward(self, chunk: int = 0, ckpt: bool = False) -> torch.Tensor:
        return self.exc.spread(self.d2nn(self.inputs), chunk=chunk, ckpt=ckpt)


def build_scalar_coder(cfg) -> ScalarDiffractiveCoder:
    return ScalarDiffractiveCoder(cfg).to(cfg.device)
