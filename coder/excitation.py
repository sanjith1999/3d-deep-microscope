"""
Vectorial (Richards-Wolf) spread of the object-plane field into the 3D excitation intensity.
"""
from __future__ import annotations
import torch
from torch.utils.checkpoint import checkpoint

from coder.psf import psf_model


class VectorialExcitation:
    """Fixed (non-learned) vectorial exPSF spread + bin."""

    def __init__(self, n_fine: int, pitch_fine_um: float, dz_um: float, nz: int,
                 na_ex: float, n_index: float = 1.0, lambda_um: float = 0.532,
                 bin_factor: int = 2, w: int = 2, device: str = "cuda",
                 checkpoint_planes: bool = True):
        self.N = n_fine
        self.nz = nz
        self.bin = bin_factor
        self.w = w
        self.device = device
        self.checkpoint = checkpoint_planes   # recompute per-plane FFTs in backward (saves memory)
        self.focus_idx = nz // 2 - 1                      # exPSF z=0 slice (arange(-Nz//2+1, ...))

        # Vectorial excitation PSF on the fine grid: (3, nz, N, N) complex.
        exPSF = psf_model(na_ex, n_index, lambda_um, pitch_fine_um, pitch_fine_um, dz_um,
                          n_fine, n_fine, nz)().detach().permute(0, 3, 1, 2).to(device)
        # Per-(component, plane) OTFs on the w-padded grid.
        C, K, N = exPSF.shape[0], exPSF.shape[1], n_fine
        pad = torch.zeros(C, K, w * N, w * N, dtype=torch.complex64, device=device)
        c0 = (w - 1) * N // 2
        pad[:, :, c0:c0 + N, c0:c0 + N] = exPSF
        self.otf = torch.fft.fft2(torch.fft.ifftshift(pad, dim=(-2, -1)))   # (3, nz, wN, wN)
        self.c0 = c0

    def _plane(self, F: torch.Tensor, otf_i: torch.Tensor) -> torch.Tensor:
        """One depth plane: intensity of the focal field defocused by exPSF(z_i)."""
        N, c0 = self.N, self.c0
        prod = F[:, None] * otf_i[None]                                     # (B, 3, wN, wN)
        h1 = torch.fft.fftshift(torch.fft.ifft2(prod), dim=(-2, -1))[..., c0:c0 + N, c0:c0 + N]
        return h1.abs().square().sum(dim=1)

    def spread(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B, N, N) complex focal fields -> H2 (B, nz, N/bin, N/bin) real, binned to 0.25 um."""
        B, N, w, c0 = f.shape[0], self.N, self.w, self.c0
        fpad = torch.zeros(B, w * N, w * N, dtype=torch.complex64, device=f.device)
        fpad[:, c0:c0 + N, c0:c0 + N] = f
        F = torch.fft.fft2(torch.fft.ifftshift(fpad, dim=(-2, -1)))          # (B, wN, wN)
        planes = []
        for i in range(self.nz):
            if self.checkpoint and F.requires_grad:
                planes.append(checkpoint(self._plane, F, self.otf[:, i], use_reentrant=False))
            else:
                planes.append(self._plane(F, self.otf[:, i]))
        H2 = torch.stack(planes, dim=1)                                      # (B, nz, N, N)
        b = self.bin
        return H2.reshape(B, self.nz, N // b, b, N // b, b).mean(dim=(3, 5))
