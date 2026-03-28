"""
ExportForward: illumination volume x emission PSF -> pooled camera frames, and its explicit matrix A.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from coder.psf import conv_3D
from pipeline.forward.common import matched_dose


class ExportForward:
    """Matrix-free forward operator for one coder's illumination H2 (nz, N, N per pattern)."""

    def __init__(self, H2: torch.Tensor, emPSF: torch.Tensor, patch=(4, 4, 4),
                 d_size: int = 4, w: int = 2, chunk: int = 0, dose_mean: float = 1.0,
                 device: str = "cuda"):
        self.dev = device
        self.chunk = chunk                                 # patterns per conv batch (0 = all)
        # dose_mean=None uses H2 as-is (pre-normalise the FULL H2 once, then build A in pattern
        # batches with dose_mean=None so every batch shares one global scale).
        self.H2 = (H2 if dose_mean is None else matched_dose(H2, dose_mean)).to(device)
        self.emPSF = emPSF.to(device)                      # (1, nz, N, N)
        self.T, self.nz, self.N, _ = self.H2.shape
        self.pz, self.px, self.py = patch                  # analog voxels per recon voxel
        self.rz, self.rx, self.ry = self.nz // self.pz, self.N // self.px, self.N // self.py
        self.d = d_size
        self.w = w
        self.m_plane = self.nz // 2                         # detected plane (n_planes = 1)
        self.recon_shape = (self.rz, self.rx, self.ry)      # (8, 32, 32)
        self.Nd = self.N // d_size
        self.meas_shape = (self.T, self.Nd, self.Nd)

    def upsample(self, X_r: torch.Tensor) -> torch.Tensor:
        """Recon grid (rz,rx,ry) -> analog grid (nz,N,N) by constant patch replication."""
        return X_r.repeat_interleave(self.pz, 0).repeat_interleave(self.px, 1).repeat_interleave(self.py, 2)

    def forward(self, X_r: torch.Tensor) -> torch.Tensor:
        return self._measure(self.upsample(X_r.to(self.dev)))

    def forward_fine(self, X_analog: torch.Tensor) -> torch.Tensor:
        """(nz,N,N) FINE object -> (T, Nd, Nd) measurements, with NO upsampling."""
        return self._measure(X_analog.to(self.dev))

    def _measure(self, Xa: torch.Tensor) -> torch.Tensor:
        Xa = Xa.to(torch.complex64)                                  # (nz, N, N)
        step = self.chunk if self.chunk else self.T
        parts = []
        for s in range(0, self.T, step):
            X1 = self.H2[s:s + step] * Xa.unsqueeze(0)               # (b, nz, N, N) illuminated object
            emis = conv_3D(self.emPSF, X1, self.w).abs()            # (b, nz, N, N) emission volume
            plane = emis[:, self.m_plane]                           # (b, N, N) detected plane
            parts.append(F.avg_pool2d(plane.unsqueeze(1), self.d).squeeze(1))  # (b, Nd, Nd)
        return torch.cat(parts, 0)                                  # (T, Nd, Nd)

    @torch.no_grad()
    def build_A(self) -> torch.Tensor:
        """Explicit forward matrix A: (T*Nd*Nd, rz*rx*ry)."""
        rz, rx, ry = self.recon_shape
        ncols = rz * rx * ry
        A = torch.zeros(self.T * self.Nd * self.Nd, ncols, device=self.dev)
        e = torch.zeros(rz, rx, ry, device=self.dev)
        for c in range(ncols):
            iz, rem = c // (rx * ry), c % (rx * ry)
            ix, iy = rem // ry, rem % ry
            e.zero_(); e[iz, ix, iy] = 1.0
            A[:, c] = self.forward(e).reshape(-1)
        return A
