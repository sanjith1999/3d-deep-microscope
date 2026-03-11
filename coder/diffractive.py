"""
Diffractive phase stack (D2NN) driven by an angle-scanned plane wave.

Angular-spectrum propagation on a padded grid; each scan direction (alpha, theta) gives one illumination pattern.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


def get_G_with_window(N: int, dx: float, delta_z: float, lambda_: float, w: int = 2,
                      device="cpu", paraxial: bool = False) -> torch.Tensor:
    """Angular-spectrum transfer function on a w*N padded grid (centered / fftshift order)."""
    M = w * N
    if delta_z == 0:
        return torch.ones((M, M), dtype=torch.cfloat, device=device)
    f = torch.fft.fftshift(torch.fft.fftfreq(M, d=dx)).to(torch.cfloat)      # centered, exactly M pts
    fx = f.view(1, M).expand(M, M)
    fy = f.view(M, 1).expand(M, M)
    if paraxial:
        f2 = (fx.abs() ** 2 + fy.abs() ** 2)
        return torch.exp(1j * 2 * np.pi * delta_z / lambda_ - 1j * np.pi * lambda_ * delta_z * f2).to(device)
    non_ev = (fx.abs() ** 2 + fy.abs() ** 2 <= (1 / lambda_) ** 2)
    power = 1j * 2 * np.pi * torch.sqrt((1 / lambda_ ** 2) - (fx ** 2) - (fy ** 2)) * delta_z
    return (torch.exp(power * non_ev) * non_ev).to(device)


class D2NNLayer(nn.Module):
    """One diffractive layer: apply a (learnable) phase mask, then AS-propagate by delta_z."""

    def __init__(self, N: int, dx: float, delta_z: float, lambda_: float,
                 learnable: bool, w: int = 2, neuron_block: int = 1, device="cpu", paraxial: bool = False):
        super().__init__()
        self.N, self.w, self.neuron_block = N, w, neuron_block
        self.register_buffer("G", get_G_with_window(N, dx, delta_z, lambda_, w, device, paraxial))
        Nn = N // neuron_block                                  # neurons per side
        if learnable:
            self.phase = nn.Parameter(0.1 * torch.randn(Nn, Nn))
        else:
            self.register_buffer("phase", torch.zeros(Nn, Nn))
        self.learnable = learnable

    def _phase_fine(self) -> torch.Tensor:
        """Upsample the neuron-grid phase to the fine grid (flat patches)."""
        p = self.phase
        if self.neuron_block > 1:
            p = p.repeat_interleave(self.neuron_block, -2).repeat_interleave(self.neuron_block, -1)
        return p

    def forward(self, field: torch.Tensor, G: torch.Tensor | None = None) -> torch.Tensor:
        B, N, w = field.shape[0], self.N, self.w
        ts = torch.exp(1j * self._phase_fine()) if self.learnable else 1.0
        x = field * ts
        pad = torch.zeros(B, w * N, w * N, dtype=torch.cfloat, device=field.device)
        c = (w - 1) * N // 2
        pad[:, c:c + N, c:c + N] = x
        Gz = self.G if G is None else G
        A = torch.fft.fftshift(torch.fft.fft2(pad), dim=(-2, -1)) * Gz.view(1, w * N, w * N)
        B_ = torch.fft.ifft2(torch.fft.ifftshift(A, dim=(-2, -1)))
        return B_[:, c:c + N, c:c + N]


class D2NNStack(nn.Module):
    """Fixed input hop + `n_layers` learnable phase layers."""

    def __init__(self, N: int, dx_um: float, lambda_um: float, n_layers: int,
                 delta_z_um: float, in_out_dz_um: float = 3.3e-3, w: int = 2,
                 neuron_block: int = 1, out_dz_um: float | None = None, device="cpu", paraxial: bool = False):
        super().__init__()
        dx, lam = dx_um * 1e-6, lambda_um * 1e-6
        dz, io = delta_z_um * 1e-6, in_out_dz_um * 1e-6
        out = (in_out_dz_um if out_dz_um is None else out_dz_um) * 1e-6
        self._N, self._dx, self._lam, self._w, self._paraxial = N, dx, lam, w, paraxial
        nb = neuron_block
        kw = dict(w=w, neuron_block=nb, device=device, paraxial=paraxial)
        layers = [D2NNLayer(N, dx, io, lam, learnable=False, **kw)]
        for _ in range(n_layers - 1):
            layers.append(D2NNLayer(N, dx, dz, lam, learnable=True, **kw))
        layers.append(D2NNLayer(N, dx, out, lam, learnable=True, **kw))
        self.layers = nn.ModuleList(layers)

    @property
    def phases(self) -> list[torch.Tensor]:
        return [l.phase for l in self.layers if l.learnable]

    def forward(self, field: torch.Tensor, out_dz_um: float | None = None) -> torch.Tensor:
        for layer in self.layers[:-1]:
            field = layer(field)
        G = None
        if out_dz_um is not None:
            G = get_G_with_window(self._N, self._dx, out_dz_um * 1e-6, self._lam, self._w, field.device, self._paraxial)
        return self.layers[-1](field, G=G)


def _angle_grid(n_alphas: int, n_thetas: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The (alpha, theta) sampling of the angle scan, in pattern order (alpha-major)."""
    alphas = torch.arange(0, math.pi, math.pi / n_alphas)
    delta = math.pi / 6
    if n_alphas == 1:
        thetas = torch.tensor([math.pi / 2]) if n_thetas == 1 else torch.linspace(delta, math.pi - delta, n_thetas)
    else:
        t1 = torch.linspace(delta, math.pi / 2 - delta, n_thetas // 2)
        t2 = torch.linspace(math.pi / 2 + delta, math.pi - delta, n_thetas - n_thetas // 2)
        thetas = torch.cat((t1, t2))
    return alphas, thetas


def angle_scan_inputs(N: int, dx_um: float, lambda_um: float, n_alphas: int, n_thetas: int,
                      device="cpu") -> torch.Tensor:
    """Input plane waves phi = k (y sin a + x cos a) cos t, swept over (alpha, theta)."""
    k = 2 * math.pi / lambda_um
    xx = (torch.arange(N).view(1, N).repeat(N, 1)) * dx_um
    yy = (torch.arange(N).view(N, 1).repeat(1, N)) * dx_um
    alphas, thetas = _angle_grid(n_alphas, n_thetas)
    fields = []
    for a in alphas:
        base = yy * math.sin(a) + xx * math.cos(a)
        for t in thetas:
            fields.append(torch.exp(1j * (k * base * math.cos(t))))
    return torch.stack(fields, 0).to(torch.cfloat).to(device)


def angle_scan_directions(n_alphas: int, n_thetas: int, device="cpu") -> torch.Tensor:
    alphas, thetas = _angle_grid(n_alphas, n_thetas)
    d = [torch.stack((math.cos(t) * torch.cos(a), math.cos(t) * torch.sin(a)))
         for a in alphas for t in thetas]
    return torch.stack(d, 0).float().to(device)
